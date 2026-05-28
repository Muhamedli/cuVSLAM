
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA software released under the NVIDIA Community License is intended to be used to enable
 * the further development of AI and robotics technologies. Such software has been designed, tested,
 * and optimized for use with NVIDIA hardware, and this License grants permission to use the software
 * solely with such hardware.
 * Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
 * modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
 * outputs generated using the software or derivative works thereof. Any code contributions that you
 * share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
 * in future releases without notice or attribution.
 * By using, reproducing, modifying, distributing, performing, or displaying any portion or element
 * of the software or derivative works thereof, you agree to be bound by this License.
 */

#include "cuvslam/cuvslam2.h"

#include <algorithm>
#include <stdexcept>

#include "common/coordinate_system.h"
#include "common/error.h"
#include "common/isometry.h"
#include "math/twist_to_angle.h"
#include "odometry/increment_pose.h"
#include "odometry/mono_visual_odometry.h"
#include "odometry/multi_visual_odometry.h"
#include "odometry/rgbd_odometry.h"
#include "odometry/stereo_inertial_odometry.h"
#include "odometry/svo_config.h"
#include "slam/async_localizer/async_localizer.h"
#include "slam/async_slam/async_slam.h"
#include "slam/merge_map/merge_dbs.h"

#include "cuvslam/debug_dump.h"
#include "cuvslam/internal.h"

namespace cuvslam {

namespace pnp {
extern int g_pnp_max_iterations;
}

namespace {

// Convert translation vector from cuVSLAM to OpenCV coordinate system
Vector3f OpencvFromCuvslam(const Vector3T& cuvslam_vec) {
  return Vector3f{cuvslam_vec.x(), -cuvslam_vec.y(), -cuvslam_vec.z()};
}

Vector3T CuvslamFromOpencv(const Vector3T& opencv_vec) {
  return Vector3T{opencv_vec.x(), -opencv_vec.y(), -opencv_vec.z()};
}

Vector3f OpencvFromCuvslam(const Vector3f& cuvslam_vec) {
  return Vector3f{cuvslam_vec[0], -cuvslam_vec[1], -cuvslam_vec[2]};
}

Vector3T CuvslamFromOpencv(const Vector3f& opencv_vec) {
  return Vector3T{opencv_vec[0], -opencv_vec[1], -opencv_vec[2]};
}

// TODO(vikuznetsov): Remove camera::MulticameraMode & reuse cuvslam enum? What about Manual mode hidden from
// cuvslam API?
camera::MulticameraMode ToMulticamMode(Odometry::MulticameraMode mode) {
  switch (mode) {
    case Odometry::MulticameraMode::Performance:
      return camera::MulticameraMode::Performance;
    case Odometry::MulticameraMode::Precision:
      return camera::MulticameraMode::Precision;
    case Odometry::MulticameraMode::Moderate:
      return camera::MulticameraMode::Moderate;
  }
  throw std::invalid_argument{"Incorrect multicamera mode: " +
                              std::to_string(ToUnderlying<Odometry::MulticameraMode>(mode))};
}

ImageEncoding ToImageEncoding(Image::Encoding mode) {
  switch (mode) {
    case Image::Encoding::MONO:
      return ImageEncoding::MONO8;
    case Image::Encoding::RGB:
      return ImageEncoding::RGB8;
  }
  throw std::invalid_argument{"Incorrect image encoding: " + std::to_string(ToUnderlying<Image::Encoding>(mode))};
}

void CreateCameraModel(const Camera& camera, std::unique_ptr<camera::ICameraModel>& camera_model) {
  Vector2T resolution{camera.size[0], camera.size[1]};
  Vector2T focal{camera.focal[0], camera.focal[1]};
  Vector2T principal{camera.principal[0], camera.principal[1]};

  THROW_INVALID_ARG_IF(principal[0] < 0.0 || principal[1] < 0.0, "Principal point coords must be >= 0.0");
  THROW_INVALID_ARG_IF(focal[0] <= 0.0 || focal[1] <= 0.0, "Focal length must be > 0.0");
  THROW_INVALID_ARG_IF(resolution[0] <= 0 || resolution[1] <= 0, "Image width/height must be > 0");

  // TODO(vikuznetsov): use enum in camera::CreateCameraModel
  const std::string distortion_model = ToString(camera.distortion.model);

  camera_model = camera::CreateCameraModel(resolution, focal, principal, distortion_model,
                                           camera.distortion.parameters.data(), camera.distortion.parameters.size());
  THROW_INVALID_ARG_IF(!camera_model, "Failed to create camera model for distortion model '" + distortion_model +
                                          "' with " + std::to_string(camera.distortion.parameters.size()) +
                                          " parameters");
}

void SetTrackerRigAndIntrinsics(std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models,
                                camera::Rig& internal_rig, const std::vector<Camera>& cameras) {
  internal_rig.num_cameras = cameras.size();
  cameras_models.resize(cameras.size());

  for (size_t k = 0; k < cameras.size(); ++k) {
    CreateCameraModel(cameras[k], cameras_models[k]);
    internal_rig.intrinsics[k] = cameras_models[k].get();
    {
      // Convert from OpenCV to cuVSLAM coordinate system for input rig_from_camera
      Isometry3T rig_from_camera = cuvslam::CuvslamFromOpencv(ConvertPoseToIsometry(cameras[k].rig_from_camera));

      internal_rig.camera_from_rig[k] = rig_from_camera.inverse();
    }
  }
}

void CheckCameras(const cuvslam::Rig& rig) {
  THROW_INVALID_ARG_IF(rig.cameras.empty(), "No cameras in a rig");
  THROW_INVALID_ARG_IF(rig.cameras.size() > camera::Rig::kMaxCameras, "Number of cameras limit exceeded");
  for (const auto& cam : rig.cameras) {
    THROW_INVALID_ARG_IF(cam.size[0] != rig.cameras[0].size[0] || cam.size[1] != rig.cameras[0].size[1],
                         "All cameras resolutions must be the same");
  }
  // Check that no two cameras have identical poses
  for (size_t i = 0; i < rig.cameras.size(); i++) {
    for (size_t j = i + 1; j < rig.cameras.size(); j++) {
      const auto& p_i = rig.cameras[i].rig_from_camera;
      const auto& p_j = rig.cameras[j].rig_from_camera;
      bool same_r = std::equal(p_i.rotation.begin(), p_i.rotation.end(), p_j.rotation.begin());
      bool same_t = std::equal(p_i.translation.begin(), p_i.translation.end(), p_j.translation.begin());
      THROW_INVALID_ARG_IF(same_r && same_t,
                           "Cameras " + std::to_string(i) + " and " + std::to_string(j) + " have identical poses");
    }
  }
}

void CheckRectifiedStereoCamera(const cuvslam::Rig& rig) {
  THROW_INVALID_ARG_IF(rig.cameras.size() % 2 != 0,
                       "Rectified stereo camera mode only works with 1+ stereo cameras. "
                       "Number of cameras must be even.");
  for (size_t i = 0; i < rig.cameras.size(); ++i) {
    const auto& dist = rig.cameras[i].distortion;
    THROW_INVALID_ARG_IF(
        dist.model != Distortion::Model::Pinhole ||
            std::any_of(dist.parameters.begin(), dist.parameters.end(), [](const float& p) { return p != 0.0f; }),
        "Rectified stereo camera mode requires pinhole distortion model for camera " + std::to_string(i));
  }
  for (size_t i = 0; i < rig.cameras.size(); i += 2) {
    const auto& rot_0 = rig.cameras[i].rig_from_camera.rotation;
    const auto& rot_1 = rig.cameras[i + 1].rig_from_camera.rotation;
    THROW_INVALID_ARG_IF(
        !std::equal(rot_0.begin(), rot_0.end(), rot_1.begin(),
                    [](float a, float b) { return std::abs(a - b) < 1e-6f; }),
        "Rectified stereo camera mode requires rectified stereo cameras. Rotation matrices of cameras " +
            std::to_string(i) + " and " + std::to_string(i + 1) + " differ");
  }
}

void CheckImuCalibration(const ImuCalibration& imu_calibration) {
  Eigen::Quaternionf quat{imu_calibration.rig_from_imu.rotation.data()};
  Matrix3T rot = quat.toRotationMatrix();

  THROW_INVALID_ARG_IF(rot.array().isNaN().any() ||
                           vec<3>(imu_calibration.rig_from_imu.translation).array().isNaN().any() ||
                           !rot.inverse().isApprox(rot.transpose()) || std::abs(rot.determinant() - 1.f) > 1e-6,
                       "IMU Calibration: rig from IMU transform is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.gyroscope_noise_density),
                       "IMU Calibration: gyroscope_noise_density is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.gyroscope_random_walk),
                       "IMU Calibration: gyroscope_random_walk is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.accelerometer_noise_density),
                       "IMU Calibration: accelerometer_noise_density is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.accelerometer_random_walk),
                       "IMU Calibration: accelerometer_random_walk is not valid");
  THROW_INVALID_ARG_IF(std::isnan(imu_calibration.frequency), "IMU Calibration: IMU data frequency is not valid");
}

void CheckImages(const Odometry::ImageSet& images, int64_t frame_sync_threshold_ns,
                 const std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models) {
  THROW_INVALID_ARG_IF(images.empty(), "No images provided");
  for (size_t i = 0; i < images.size(); ++i) {
    THROW_INVALID_ARG_IF(images[i].pixels == nullptr, "No buffer provided for image " + std::to_string(i));
#ifdef USE_CUDA
    THROW_INVALID_ARG_IF(images[i].is_gpu_mem != cuda::IsGpuPointer(images[i].pixels),
                         "is_gpu_mem flag mismatch for image " + std::to_string(i));
#endif
    THROW_INVALID_ARG_IF(images[i].data_type != Image::DataType::UINT8,
                         "Image data type must be UINT8 for image " + std::to_string(i));
    THROW_INVALID_ARG_IF(images[i].camera_index >= cameras_models.size(),
                         "camera_index >= number of cameras for image " + std::to_string(i));
    const auto& resolution = cameras_models[images[i].camera_index].get()->getResolution();
    THROW_INVALID_ARG_IF(images[i].width != resolution[0] || images[i].height != resolution[1],
                         "Image dimensions (" + std::to_string(images[i].width) + "x" +
                             std::to_string(images[i].height) + ") do not correspond to camera resolution (" +
                             std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]) + ") for image " +
                             std::to_string(i));

    for (size_t j = 0; j < i; ++j) {
      THROW_INVALID_ARG_IF(std::abs(images[i].timestamp_ns - images[j].timestamp_ns) >= frame_sync_threshold_ns,
                           "Timestamps differ by more than " + std::to_string(frame_sync_threshold_ns / 1e6) + " ms" +
                               " for images " + std::to_string(j) + ", " + std::to_string(i));
      THROW_INVALID_ARG_IF(images[j].pixels == images[i].pixels,
                           "The same image buffer for images " + std::to_string(j) + ", " + std::to_string(i));
      THROW_INVALID_ARG_IF(images[j].camera_index == images[i].camera_index,
                           "The same camera index for images " + std::to_string(j) + ", " + std::to_string(i));
    }
  }
}

void FillImageSourceAndShape(const Image& image, ImageSource& source, ImageShape& shape) {
  source.data = const_cast<void*>(image.pixels);
  switch (image.data_type) {
    case Image::DataType::UINT8:
      source.type = ImageSource::U8;
      break;
    case Image::DataType::UINT16:
      source.type = ImageSource::U16;
      break;
    case Image::DataType::FLOAT32:
      source.type = ImageSource::F32;
      break;
    default:
      throw std::invalid_argument{"Unsupported image data type"};
  }
  source.memory_type = image.is_gpu_mem ? ImageSource::Device : ImageSource::Host;
  source.pitch = image.pitch;
  source.image_encoding = ToImageEncoding(image.encoding);

  shape.width = image.width;
  shape.height = image.height;
}

void CheckDepths(const Odometry::ImageSet& depths,
                 const std::vector<std::unique_ptr<camera::ICameraModel>>& cameras_models) {
  THROW_INVALID_ARG_IF(depths.empty(), "Depth images are required for RGBD odometry");
  THROW_INVALID_ARG_IF(depths.size() > 1, "Only one depth image is supported by RGBD odometry");
  for (size_t i = 0; i < depths.size(); ++i) {
    THROW_INVALID_ARG_IF(depths[i].pixels == nullptr, "No buffer provided for depth image " + std::to_string(i));
#ifdef USE_CUDA
    THROW_INVALID_ARG_IF(depths[i].is_gpu_mem != cuda::IsGpuPointer(depths[i].pixels),
                         "is_gpu_mem flag mismatch for depth image " + std::to_string(i));
#endif
    THROW_INVALID_ARG_IF(
        depths[i].data_type != Image::DataType::UINT16 && depths[i].data_type != Image::DataType::FLOAT32,
        "Depth data type must be UINT16 or FLOAT32");
    THROW_INVALID_ARG_IF(depths[i].camera_index >= cameras_models.size(),
                         "camera_index >= number of cameras for depth image " + std::to_string(i));
    const auto& resolution = cameras_models[depths[i].camera_index].get()->getResolution();
    THROW_INVALID_ARG_IF(depths[i].width != resolution[0] || depths[i].height != resolution[1],
                         "Depth image dimensions (" + std::to_string(depths[i].width) + "x" +
                             std::to_string(depths[i].height) + ") do not correspond to camera resolution (" +
                             std::to_string(resolution[0]) + "x" + std::to_string(resolution[1]) + ")");
  }
}

}  // anonymous namespace

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Public API
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SetVerbosity(int verbosity) {
  constexpr const Trace::Verbosity max_allowed =
#if defined(NDEBUG)
      Trace::Verbosity::Message;
#else
      Trace::Verbosity::Debug;
#endif
  Trace::SetVerbosity(Trace::ToVerbosity(verbosity, max_allowed));
}

void WarmUpGPU() { WarmUpGpuImpl(); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Odometry class implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// using ImageContexts = std::unordered_map<CameraId, std::shared_ptr<Odometry::State::Context>>;

class Odometry::Impl {
public:
  // objects
  camera::Rig rig;  // rig and cameras_models
  camera::FrustumIntersectionGraph fig;
  std::vector<std::unique_ptr<camera::ICameraModel>> cameras_models;
  std::unique_ptr<sof::ImageManager> image_manager;
  std::unique_ptr<odom::IVisualOdometry> visual_odometry;
  // state
  FrameId frame_id{0};
  sof::Images prev_image_ptrs;
  Isometry3T prev_abs_pose{Isometry3T::Identity()};
  int64_t last_timestamp_ns{std::numeric_limits<int64_t>::min()};
  int64_t last_frame_timestamp_ns;
  Isometry3T last_delta{Isometry3T::Identity()};
  State::ContextMap image_contexts;
  // debug
  std::string debug_dump_directory;
  int64_t max_frame_delta_ns;
  int64_t frame_sync_threshold_ns{1'000'000};  // 1 ms
  // settings
  bool imu_fusion_enabled;
  RGBDSettings rgbd_settings;
  Odometry::OdometryMode odometry_mode;
  // stats
  bool enable_final_landmarks_export{false};
  std::unordered_map<uint64_t, Vector3f> final_landmarks;

  // data helpers

  // if camera_index == std::numeric_limits<uint32_t>::max() will return observations for all cameras
  void GetLastObservations(uint32_t camera_index, std::vector<Observation>& observations) const {
    const auto& stat = visual_odometry->get_last_stat();
    THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_observations_export to true to get last observations");
    observations.clear();
    if (camera_index == std::numeric_limits<uint32_t>::max()) {
      observations.reserve(stat->tracks2d.size());
      for (const auto& t : stat->tracks2d) {
        observations.emplace_back(Observation{t.track_id, t.uv.x(), t.uv.y(), t.cam_id});
      }
    } else {
      observations.reserve(stat->tracks2d.size() / fig.primary_cameras().size());
      for (const auto& t : stat->tracks2d) {
        if (t.cam_id == camera_index) {
          observations.emplace_back(Observation{t.track_id, t.uv.x(), t.uv.y(), t.cam_id});
        }
      }
    }
  }

  void GetLastLandmarks(std::vector<Landmark>& landmarks) const {
    const auto& stat = visual_odometry->get_last_stat();
    THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_landmarks_export to true to get last landmarks");
    landmarks.clear();
    landmarks.reserve(stat->tracks3d.size());
    for (const auto& t : stat->tracks3d) {
      // Convert from cuVSLAM to OpenCV coordinate system
      landmarks.emplace_back(Landmark{t.first, OpencvFromCuvslam(t.second)});
    }
  }
};

Odometry::Odometry(Odometry&&) noexcept = default;

Odometry::~Odometry() = default;

Odometry::Odometry(const Rig& rig, const Config& cfg) {
  std::string message;
  TracePrintIf(!CheckCudaCompatibility(message), "[WARNING] %s", message.c_str());
#ifdef ENFORCE_GPU
  THROW_INVALID_ARG_IF(!cfg.use_gpu, "cfg.use_gpu must be enabled");
#endif

  CheckCameras(rig);

  THROW_INVALID_ARG_IF(cfg.odometry_mode == OdometryMode::Inertial && rig.imus.empty(),
                       "IMU fusion is enabled, but IMU calibration is not provided");
  THROW_INVALID_ARG_IF(rig.imus.size() > 1, "Only one IMU sensor is supported");

  pnp::g_pnp_max_iterations = cfg.pnp_max_iterations;

  odom::Settings svo_settings;
  svo_settings.verbose = Trace::GetVerbosity() > Trace::Verbosity::None;

  svo_settings.sof_settings.num_desired_tracks = cfg.num_desired_tracks;
  svo_settings.sof_settings.ransac_filter = cfg.ransac_filter;
  svo_settings.sba_settings.num_sba_frames = cfg.num_sba_frames;
  svo_settings.sba_settings.num_fixed_sba_frames = cfg.num_fixed_sba_frames;
  svo_settings.sba_settings.num_sba_iterations = cfg.num_sba_iterations;

  svo_settings.sba_settings.async = cfg.async_sba;
  svo_settings.sba_settings.mode =
      cfg.odometry_mode == OdometryMode::Inertial ? sba::Mode::InertialCPU : sba::Mode::OriginalGPU;

  // Use only first camera border settings. Other camera border settings are ignored now.
  svo_settings.sof_settings.multicam_mode = ToMulticamMode(cfg.multicam_mode);
  svo_settings.sof_settings.border_top = rig.cameras[0].border_top;
  svo_settings.sof_settings.border_bottom = rig.cameras[0].border_bottom;
  svo_settings.sof_settings.border_left = rig.cameras[0].border_left;
  svo_settings.sof_settings.border_right = rig.cameras[0].border_right;
  svo_settings.sof_settings.box3_prefilter = cfg.use_denoising;
  if (cfg.rectified_stereo_camera) {
    CheckRectifiedStereoCamera(rig);
    svo_settings.sof_settings.lr_tracker = "lk_horizontal";
  }

  auto tracker{std::make_unique<Odometry::Impl>()};

  SetTrackerRigAndIntrinsics(tracker->cameras_models, tracker->rig, rig.cameras);

  std::vector<CameraId> depth_ids{
      static_cast<CameraId>(cfg.odometry_mode == OdometryMode::RGBD ? cfg.rgbd_settings.depth_camera_id : 0)};
  tracker->fig = camera::FrustumIntersectionGraph(
      tracker->rig, svo_settings.sof_settings.multicam_mode, depth_ids,
      (cfg.odometry_mode == OdometryMode::RGBD) && cfg.rgbd_settings.enable_depth_stereo_tracking,
      svo_settings.sof_settings.multicam_setup);
  THROW_INVALID_ARG_IF(
      !tracker->fig.is_valid(),
      "Bad calibration. cuVSLAM needs at least one stereo pair available for multicamera/inertial modes.");

  switch (cfg.odometry_mode) {
    case OdometryMode::Multicamera: {
      svo_settings.use_prediction = cfg.use_motion_model;
      tracker->visual_odometry =
          std::make_unique<odom::MultiVisualOdometry>(tracker->rig, tracker->fig, svo_settings, cfg.use_gpu);
      break;
    }
    case OdometryMode::Inertial: {
      const auto& imu_calibration = rig.imus[0];
      CheckImuCalibration(imu_calibration);
      // Convert from OpenCV to cuVSLAM coordinate system for input rig_from_imu
      Isometry3T rig_from_imu = CuvslamFromOpencv(ConvertPoseToIsometry(imu_calibration.rig_from_imu));
      svo_settings.imu_calibration =
          imu::ImuCalibration(rig_from_imu, imu_calibration.gyroscope_noise_density,
                              imu_calibration.gyroscope_random_walk, imu_calibration.accelerometer_noise_density,
                              imu_calibration.accelerometer_random_walk, imu_calibration.frequency);
      svo_settings.use_prediction = cfg.use_motion_model;
      tracker->visual_odometry = std::make_unique<odom::StereoInertialOdometry>(
          tracker->rig, tracker->fig, svo_settings, cfg.use_gpu, cfg.debug_imu_mode);
      break;
    }
    case OdometryMode::Mono: {
      tracker->visual_odometry = std::make_unique<odom::MonoVisualOdometry>(tracker->rig, svo_settings, cfg.use_gpu);
      break;
    }
    case OdometryMode::RGBD: {
      tracker->rgbd_settings = cfg.rgbd_settings;
      tracker->visual_odometry =
          std::make_unique<odom::RGBDOdometry>(tracker->rig, tracker->fig, svo_settings, cfg.use_gpu);
      break;
    }
    default:
      throw std::invalid_argument{"Unsupported odometry mode " +
                                  std::to_string(ToUnderlying<Odometry::OdometryMode>(cfg.odometry_mode))};
  }
  tracker->visual_odometry->enable_stat(cfg.enable_observations_export || cfg.enable_landmarks_export ||
                                        cfg.enable_final_landmarks_export);
  tracker->enable_final_landmarks_export = cfg.enable_final_landmarks_export;

  tracker->imu_fusion_enabled = cfg.odometry_mode == OdometryMode::Inertial;
  tracker->debug_dump_directory = cfg.debug_dump_directory;
  tracker->max_frame_delta_ns = static_cast<int64_t>(cfg.max_frame_delta_s * 1e9);
  tracker->odometry_mode = cfg.odometry_mode;

  tracker->image_manager = std::make_unique<sof::ImageManager>();
  ImageShape shape{rig.cameras[0].size[0], rig.cameras[0].size[1]};
  const size_t cache_size = 4;
  tracker->image_manager->init(shape, rig.cameras.size() * cache_size, cfg.use_gpu,
                               cfg.odometry_mode == OdometryMode::RGBD ? cache_size : 0);

  impl = std::move(tracker);

  DumpConfiguration(impl->debug_dump_directory, rig, cfg);
}

void Odometry::RegisterImuMeasurement(uint32_t sensor_index, const ImuMeasurement& imu) {
  THROW_INVALID_ARG_IF(!impl->imu_fusion_enabled, "IMU fusion is not enabled");
  THROW_INVALID_ARG_IF(sensor_index != 0, "Only one IMU sensor is supported");
  THROW_INVALID_ARG_IF(imu.timestamp_ns < impl->last_timestamp_ns, "Timestamps are non-monotonic");
  impl->last_timestamp_ns = imu.timestamp_ns;

  DumpRegisterImuMeasurementCall(impl->debug_dump_directory, imu);

  Vector3T acc(imu.linear_accelerations[0], imu.linear_accelerations[1], imu.linear_accelerations[2]);
  Vector3T gyro(imu.angular_velocities[0], imu.angular_velocities[1], imu.angular_velocities[2]);

  const imu::ImuMeasurement m = {imu.timestamp_ns, CuvslamFromOpencv(acc), CuvslamFromOpencv(gyro)};
  static_cast<odom::StereoInertialOdometry*>(impl->visual_odometry.get())->add_imu_measurement(m);
}

PoseEstimate Odometry::Track(const ImageSet& images, const ImageSet& masks, const ImageSet& depths) {
  CheckImages(images, impl->frame_sync_threshold_ns, impl->cameras_models);
  if (impl->odometry_mode == OdometryMode::RGBD) {
    CheckDepths(depths, impl->cameras_models);
  } else {
    THROW_INVALID_ARG_IF(!depths.empty(), "Depth images are only accepted for RGBD odometry");
  }
  DumpTrackCall(impl->debug_dump_directory, impl->frame_id, images, masks, depths);

  Sources image_sources(images.size());
  Sources masks_sources(images.size());
  DepthSources depth_sources(depths.size());
  Metas image_metas(images.size());
  sof::Images cuvslam_images_ptrs;
  impl->image_contexts.clear();

  const FrameId frame_id = impl->frame_id++;
  const int64_t current_time_ns = images[0].timestamp_ns;
  THROW_INVALID_ARG_IF(current_time_ns < impl->last_timestamp_ns, "Timestamps are non-monotonic");

  if (!impl->prev_image_ptrs.empty()) {
    THROW_INVALID_ARG_IF(current_time_ns <= impl->last_frame_timestamp_ns,
                         "Frame timestamps must be strictly increasing");
    auto current_frame_delta_ns = current_time_ns - impl->last_frame_timestamp_ns;
    TraceWarningIf(current_frame_delta_ns > impl->max_frame_delta_ns,
                   "Delta between frames at frame %d is %.0f ms that is longer than desired %.0f ms. Check camera fps "
                   "and sync settings.",
                   frame_id, current_frame_delta_ns / 1e6, impl->max_frame_delta_ns / 1e6);
  }
  impl->last_timestamp_ns = current_time_ns;
  impl->last_frame_timestamp_ns = current_time_ns;

  for (const auto& image : images) {
    bool is_rgbd = false;
    auto cam_id = image.camera_index;
    ImageSource& source = image_sources[cam_id];
    ImageSource& mask_source = masks_sources[cam_id];  // a mask for each image is required even if it is empty
    ImageMeta& meta = image_metas[cam_id];

    meta.frame_id = frame_id;
    meta.camera_index = cam_id;
    meta.timestamp = current_time_ns;

    FillImageSourceAndShape(image, source, meta.shape);

    auto mask_it =
        std::find_if(masks.begin(), masks.end(), [&cam_id](const auto& mask) { return mask.camera_index == cam_id; });
    if (mask_it != masks.end()) {
      FillImageSourceAndShape(*mask_it, mask_source, meta.mask_shape);
    }

    // At this point, we know there is only one depth image for RGBD odometry
    if (impl->odometry_mode == OdometryMode::RGBD && depths[0].camera_index == cam_id) {
      auto& depth_source = depth_sources[cam_id];
      FillImageSourceAndShape(depths[0], depth_source, meta.shape);
      meta.pixel_scale_factor = impl->rgbd_settings.depth_scale_factor;
      is_rgbd = true;
    }

    sof::ImageContextPtr ptr = is_rgbd ? impl->image_manager->acquire_with_depth() : impl->image_manager->acquire();
    THROW_RUNTIME_ERROR_IF(ptr == nullptr, "Failed to acquire image context from image_manager");
    ptr->set_image_meta(meta);
    cuvslam_images_ptrs.insert({cam_id, ptr});
    impl->image_contexts.insert({cam_id, std::static_pointer_cast<Odometry::State::Context>(ptr)});
  }

  Matrix6T static_info_exp = Matrix6T::Identity();
  bool result = impl->visual_odometry->track(image_sources, depth_sources, cuvslam_images_ptrs, impl->prev_image_ptrs,
                                             masks_sources, impl->last_delta, static_info_exp);

  for (const auto& [cam_id, img] : cuvslam_images_ptrs) {
    impl->prev_image_ptrs[cam_id] = img;
  }

  if (result) {
    impl->prev_abs_pose = odom::increment_pose(impl->prev_abs_pose, impl->last_delta);
  } else {
    return PoseEstimate{};
  }
  const Isometry3T internal_pose = impl->prev_abs_pose;  // absolute position in cuVSLAM coordinates

  // static pose covariance in exponential mapping form in WCS
  const Matrix6T static_pose_covariance_exp = static_info_exp.ldlt().solve(Matrix6T::Identity());
  // static pose covariance in euler angles in WCS
  const Matrix6T static_pose_covariance_euler =
      math::PoseCovToRollPitchYawCov(static_pose_covariance_exp, internal_pose);

  PoseEstimate pose_estimate;
  pose_estimate.timestamp_ns = current_time_ns;
  PoseWithCovariance pose_with_covariance;
  // Convert from cuVSLAM to OpenCV coordinate system for output
  pose_with_covariance.pose = ConvertIsometryToPose(OpencvFromCuvslam(internal_pose));
  mat<6>(pose_with_covariance.covariance) = static_pose_covariance_euler;
  pose_estimate.world_from_rig = pose_with_covariance;

  if (impl->enable_final_landmarks_export) {
    const auto& stat = impl->visual_odometry->get_last_stat();
    assert(stat != nullptr);  // Final landmarks need to be saved, but VO stats are not enabled; this should not happen

    for (const auto& t : stat->tracks3d) {
      const TrackId id = t.first;
      // Transform landmark from cuVSLAM to OpenCV coordinate system
      const Vector3T internal_lm = internal_pose * t.second;
      impl->final_landmarks[id] = OpencvFromCuvslam(internal_lm);
    }
  }

  return pose_estimate;
}

std::vector<Observation> Odometry::GetLastObservations(uint32_t camera_index) const {
  std::vector<Observation> observations;
  THROW_INVALID_ARG_IF(camera_index > static_cast<uint32_t>(impl->rig.num_cameras), "Camera index out of range");
  impl->GetLastObservations(camera_index, observations);
  return observations;
}

std::vector<Landmark> Odometry::GetLastLandmarks() const {
  std::vector<Landmark> landmarks;
  impl->GetLastLandmarks(landmarks);
  return landmarks;
}

std::optional<Odometry::Gravity> Odometry::GetLastGravity() const {
  THROW_INVALID_ARG_IF(!impl->imu_fusion_enabled, "IMU fusion is disabled");

  const auto& gravity_estimate = static_cast<odom::StereoInertialOdometry*>(impl->visual_odometry.get())->get_gravity();
  if (!gravity_estimate.has_value()) {
    return std::nullopt;  // Gravity is not available yet
  }

  const auto& internal_g = gravity_estimate.value();
  // Convert from cuVSLAM to OpenCV coordinate system
  Vector3T opencv_g = kOpencvFromCuvslam.linear() * internal_g;
  return Gravity{opencv_g.x(), opencv_g.y(), opencv_g.z()};
}

void Odometry::GetState(Odometry::State& state) const {
  const auto& stat = impl->visual_odometry->get_last_stat();
  THROW_INVALID_ARG_IF(stat == nullptr, "Enable export of observations and/or landmarks to get state");
  state.frame_id = impl->frame_id - 1;  // frame_id is incremented after tracking, so we need to subtract 1
  state.timestamp_ns = impl->last_frame_timestamp_ns;
  state.delta = ConvertIsometryToPose(OpencvFromCuvslam(impl->last_delta));
  state.keyframe = stat->keyframe;
  state.warming_up = stat->heating;
  state.gravity = impl->imu_fusion_enabled ? GetLastGravity() : std::nullopt;
  impl->GetLastObservations(std::numeric_limits<uint32_t>::max(), state.observations);
  impl->GetLastLandmarks(state.landmarks);
  state.context = impl->image_contexts;
}

std::unordered_map<uint64_t, Vector3f> Odometry::GetFinalLandmarks() const {
  const auto& stat = impl->visual_odometry->get_last_stat();
  THROW_INVALID_ARG_IF(stat == nullptr, "Set enable_final_landmarks_export to true to get final landmarks");
  // final_landmarks are already in OpenCV coordinate system
  return impl->final_landmarks;
}

const std::vector<uint8_t>& Odometry::GetPrimaryCameras() const { return impl->fig.primary_cameras(); }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Slam class implementation
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Slam::Impl {
public:
  camera::Rig rig_;
  std::vector<std::unique_ptr<camera::ICameraModel>> cameras_models_;
  std::vector<uint8_t> primary_cameras_;
  bool use_gpu_ = true;
  bool sync_mode_ = false;
  bool gt_align_mode_ = false;
  std::unique_ptr<slam::AsyncSlam> async_slam_;
  std::unique_ptr<slam::AsyncLocalizer> async_localizer_;
  LocalizationCallback localizer_response_;
  uint64_t frame_id_ = 0;

  // views
  bool enable_reading_internals_ = false;
  static constexpr auto kMaxDataLayer = ToUnderlying(DataLayer::Max);
  std::shared_ptr<slam::ViewManager<slam::ViewLandmarks>> landmarks_views_[kMaxDataLayer];
  std::shared_ptr<Landmarks> landmarks_[kMaxDataLayer];
  std::shared_ptr<slam::ViewManager<slam::ViewPoseGraph>> pose_graph_view_;
  std::shared_ptr<PoseGraph> pose_graph_;  // return type is different, so cannot just return a pointer to a view
  std::shared_ptr<slam::ViewManager<slam::ViewLocalizerProbes>> localizer_probes_view_;
  std::shared_ptr<LocalizerProbes> localizer_probes_;

  Impl(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config)
      : primary_cameras_(primary_cameras) {
    std::string message;
    TracePrintIf(!CheckCudaCompatibility(message), "[WARNING] %s", message.c_str());

    SetTrackerRigAndIntrinsics(cameras_models_, rig_, rig.cameras);

    slam::AsyncSlamOptions options;
    options.map_cache_path = config.map_cache_path;
    options.use_gpu = config.use_gpu;
    options.reproduce_mode = config.sync_mode;
    options.pose_for_frame_required = true;  // required for localization
    options.spatial_index_options.cell_size = config.map_cell_size;
    options.max_landmarks_distance = config.max_landmarks_distance;
    options.planar_constraints = config.planar_constraints;
    options.throttling_time_ms = config.throttling_time_ms;
    use_gpu_ = config.use_gpu;
    sync_mode_ = config.sync_mode;
    gt_align_mode_ = config.gt_align_mode;
    if (config.gt_align_mode) {
      THROW_INVALID_ARG_IF(!config.sync_mode, "sync_mode should be enabled for gt_align_mode.");
      THROW_INVALID_ARG_IF(config.planar_constraints, "planar_constraints should be disabled for gt_align_mode.");
      options.loop_closure_solver_type = slam::LoopClosureSolverType::kDummy;
      options.pgo_options.type = slam::PoseGraphOptimizerType::Dummy;
    } else {
      options.max_pose_graph_nodes = config.max_map_size;
    }
    async_slam_ = std::make_unique<slam::AsyncSlam>(rig_, primary_cameras_, options);

    enable_reading_internals_ = config.enable_reading_internals;
    if (enable_reading_internals_) {
      pose_graph_view_ = std::make_shared<slam::ViewManager<slam::ViewPoseGraph>>();
      pose_graph_ = std::make_shared<PoseGraph>();

      localizer_probes_view_ = std::make_shared<slam::ViewManager<slam::ViewLocalizerProbes>>();
      localizer_probes_ = std::make_shared<LocalizerProbes>();

      for (DataLayer layer : {DataLayer::Map, DataLayer::LoopClosure, DataLayer::LocalizerMap,
                              DataLayer::LocalizerLandmarks, DataLayer::LocalizerLoopClosure, DataLayer::Landmarks}) {
        landmarks_views_[ToUnderlying(layer)] = std::make_shared<slam::ViewManager<slam::ViewLandmarks>>();
        landmarks_[ToUnderlying(layer)] = std::make_shared<Landmarks>();
      }

      async_slam_->SetLandmarksView(landmarks_views_[ToUnderlying(DataLayer::Map)]);
      async_slam_->SetLoopClosureView(landmarks_views_[ToUnderlying(DataLayer::LoopClosure)]);
      async_slam_->SetPoseGraphView(pose_graph_view_);
    }
  }

  ~Impl() {}

  void Track(FrameId frame_id, int64_t timestamp_ns, const odom::IVisualOdometry::VOFrameStat& stat,
             const Isometry3T& delta, const sof::Images& images) {
    sof::Images slam_images;
    // only pass images for primary cameras; TODO: custom primary cameras
    for (CameraId cam_id : primary_cameras_) {
      if (images.find(cam_id) != images.end()) {
        slam_images[cam_id] = images.at(cam_id);
      }
    }
    async_slam_->TrackResult(frame_id, timestamp_ns, stat, slam_images, delta, nullptr);
    // for synchronous execution:
    async_slam_->MainLoopStep();

    LocalizeNext(slam_images);
  }

  void LocalizeNext(const sof::Images& slam_images) {
    if (!async_localizer_) {
      return;
    }

    Isometry3T slam_pose;
    if (!async_slam_->GetSlamPose(slam_pose)) {
      throw std::runtime_error("Failed to get SLAM pose");
    }

    // Async localization in progress
    slam::AsyncSlam::LocalizationResult localization_result;
    if (async_localizer_->ReceiveResult(localization_result)) {
      SendLocalizationResult(localization_result, localizer_response_);
      // finished
      async_localizer_.reset();
      localizer_response_ = nullptr;
    } else {
      async_localizer_->AddNewRequest(async_slam_->GetVOTrackData(), slam_images, slam_pose);
    }
  }

  void SendLocalizationResult(const slam::AsyncSlam::LocalizationResult& result, LocalizationCallback callback) {
    if (result.is_valid()) {
      // Union
      async_slam_->LocalizedInSlam(result);
    }
    if (callback) {
      if (result.is_valid()) {
        callback(Result<Pose>::Success(ConvertIsometryToPose(OpencvFromCuvslam(result.pose_in_slam))));
      } else {
        callback(Result<Pose>::Error("Localization failed"));
      }
    }
  }
};

Slam::Slam(const Rig& rig, const std::vector<uint8_t>& primary_cameras, const Config& config)
    : impl(std::make_unique<Impl>(rig, primary_cameras, config)) {}

Slam::Slam(Slam&& other) noexcept = default;

Slam::~Slam() = default;

Pose Slam::Track(const Odometry::State& state, const Pose* gt_pose) {
  // Convert from public API types to internal types
  Isometry3T internal_delta = CuvslamFromOpencv(ConvertPoseToIsometry(state.delta));
  if (impl->gt_align_mode_) {
    THROW_INVALID_ARG_IF(gt_pose == nullptr, "gt_pose should be provided if gt_align_mode is enabled.");
    Isometry3T prev_pose{Isometry3T::Identity()};
    // if GetSlamPose returns false, prev_pose stays Identity
    impl->async_slam_->GetSlamPose(prev_pose);
    internal_delta = prev_pose.inverse() * CuvslamFromOpencv(ConvertPoseToIsometry(*gt_pose));
    internal_delta.linear() = common::CalculateRotationFromSVD(internal_delta.matrix());
  } else {
    THROW_INVALID_ARG_IF(gt_pose != nullptr, "gt_pose should be nullptr if gt_align_mode is disabled.");
  }

  odom::IVisualOdometry::VOFrameStat stat;
  stat.keyframe = state.keyframe;
  stat.heating = state.warming_up;

  // Convert observations to tracks2d
  stat.tracks2d.reserve(state.observations.size());
  for (const auto& obs : state.observations) {
    Track2D track;
    track.track_id = obs.id;
    track.uv = Vector2T(obs.u, obs.v);
    track.cam_id = obs.camera_index;
    stat.tracks2d.push_back(track);
  }

  // Convert landmarks to tracks3d
  for (const auto& lm : state.landmarks) {
    Vector3T internal_lm = CuvslamFromOpencv(lm.coords);
    stat.tracks3d[lm.id] = internal_lm;
  }

  // Convert image contexts to sof::Images
  sof::Images slam_images;
  slam_images.reserve(impl->primary_cameras_.size());
  for (CameraId cam_id : impl->primary_cameras_) {
    auto it = state.context.find(cam_id);
    if (it != state.context.end()) {
      slam_images[cam_id] = std::static_pointer_cast<sof::ImageContext>(it->second);
    }
  }

  impl->Track(state.frame_id, state.timestamp_ns, stat, internal_delta, slam_images);
  impl->frame_id_ = state.frame_id;

  Isometry3T slam_pose;
  if (!impl->async_slam_->GetSlamPose(slam_pose)) {
    throw std::runtime_error("Failed to get SLAM pose");
  }
  // copy odometry landmarks to observations view
  auto view_manager = impl->landmarks_views_[ToUnderlying(DataLayer::Landmarks)];
  auto landmarks_view = view_manager ? view_manager->acquire_earliest() : nullptr;
  if (landmarks_view) {
    for (const auto& [id, v] : stat.tracks3d) {
      if (landmarks_view->landmarks.size() >= landmarks_view->landmarks.capacity()) {
        break;
      }
      landmarks_view->landmarks.push_back({static_cast<uint64_t>(id), 1, ToArray<float, 3>(slam_pose * v)});
    }
    landmarks_view->timestamp_ns = state.timestamp_ns;
  }
  return ConvertIsometryToPose(OpencvFromCuvslam(slam_pose));
}

void Slam::SetSlamPose(const Pose& pose) {
  Isometry3T slam_pose = CuvslamFromOpencv(ConvertPoseToIsometry(pose));
  impl->async_slam_->SetAbsolutePose(slam_pose);
}

void Slam::GetAllSlamPoses(std::vector<PoseStamped>& poses, uint32_t max_poses_count) const {
  std::map<uint64_t, storage::Isometry3<float>> frames;
  if (!impl->async_slam_->GetPosesForAllFrames(frames)) {
    throw std::runtime_error("Failed to get SLAM poses");
  }

  // Resize output vector to match the number of frames, but respect max_poses_count if specified
  size_t num_poses =
      max_poses_count > 0 ? std::min(frames.size(), static_cast<size_t>(max_poses_count)) : frames.size();
  poses.resize(num_poses);

  // Convert each pose to the public API format
  size_t i = 0;
  for (const auto& [timestamp_ns, pose] : frames) {
    if (i >= num_poses) {
      break;
    }
    poses[i].timestamp_ns = static_cast<int64_t>(timestamp_ns);
    poses[i].pose = ConvertIsometryToPose(OpencvFromCuvslam(pose));
    i++;
  }
}

void Slam::SaveMap(const std::string_view& folder_name, std::function<void(bool success)> callback) const {
  impl->async_slam_->CopyToDatabase(std::string{folder_name}, callback);
}

#define CALLBACK_AND_RETURN_IF(condition, callback, type, message) \
  if (condition) {                                                 \
    callback(Result<type>::Error(message));                        \
    return;                                                        \
  }

void Slam::LocalizeInMap(const std::string_view& folder_name, const Pose& guess_pose, const ImageSet& images,
                         LocalizationSettings settings, LocalizationCallback callback) {
  CALLBACK_AND_RETURN_IF(impl->async_localizer_, callback, Pose, "Localization is already in progress.");

  slam::AsyncLocalizerOptions localizer_options;
  localizer_options.use_gpu = impl->use_gpu_;
  localizer_options.reproduce_mode = impl->sync_mode_;
  localizer_options.static_frame_calculation = false;

  localizer_options.horizontal_search_radius = settings.horizontal_search_radius;
  localizer_options.vertical_search_radius = settings.vertical_search_radius;
  localizer_options.horizontal_step = settings.horizontal_step;
  localizer_options.vertical_step = settings.vertical_step;
  localizer_options.angle_step_rads = settings.angular_step_rads;

  Isometry3T isometry_guess_pose = CuvslamFromOpencv(ConvertPoseToIsometry(guess_pose));

  Isometry3T current_pose;
  CALLBACK_AND_RETURN_IF(!impl->async_slam_->GetSlamPose(current_pose), callback, Pose,
                         "Failed to get SLAM pose. Make sure Slam::Track() is called before LocalizeInMap()");

  // AsyncLocalizer is created only for the duration of the localization
  auto localizer = std::make_unique<slam::AsyncLocalizer>();
  localizer->Init(impl->rig_, localizer_options);

  if (settings.enable_reading_internals && impl->enable_reading_internals_) {
    localizer->SetLocalizerLandmarksView(impl->landmarks_views_[ToUnderlying(DataLayer::LocalizerMap)]);
    localizer->SetLocalizerObservationView(impl->landmarks_views_[ToUnderlying(DataLayer::LocalizerLandmarks)]);
    localizer->SetLocalizerLCLandmarksView(impl->landmarks_views_[ToUnderlying(DataLayer::LocalizerLoopClosure)]);
    localizer->SetLocalizerProbesView(impl->localizer_probes_view_);
  }

  CALLBACK_AND_RETURN_IF(!localizer->OpenDatabase(std::string{folder_name}), callback, Pose,
                         "Failed to open database.");

  if (impl->sync_mode_) {
    // horrible copy-paste from C API
    const auto& primary_cameras = impl->primary_cameras_;
    Sources image_sources(images.size());
    Metas image_metas(images.size());
    sof::Images images_ptrs;
    // Get image dimensions from the first camera (assuming all cameras have same resolution)
    ImageShape shape{images[0].width, images[0].height};
    const size_t cache_size = 4;
    auto image_manager = std::make_unique<sof::ImageManager>();
    image_manager->init(shape, primary_cameras.size() * cache_size, localizer_options.use_gpu, 0);

    for (const auto& image : images) {
      auto cam_id = image.camera_index;
      if (std::none_of(primary_cameras.begin(), primary_cameras.end(), [cam_id](auto&& id) { return id == cam_id; })) {
        continue;
      }

      ImageSource& source = image_sources[cam_id];
      ImageMeta& meta = image_metas[cam_id];

      meta.frame_id = impl->frame_id_;
      meta.camera_index = cam_id;
      meta.timestamp = images[0].timestamp_ns;

      FillImageSourceAndShape(image, source, meta.shape);

      sof::ImageContextPtr ptr = image_manager->acquire();
      CALLBACK_AND_RETURN_IF(ptr == nullptr, callback, Pose, "Failed to acquire image context from image_manager");
      ptr->set_image_meta(meta);
      if (localizer_options.use_gpu) {
#ifdef USE_CUDA
        cuda::Stream s{true};
        ptr->build_gpu_image_pyramid(source, false, s.get_stream());
        ptr->build_gpu_gradient_pyramid(false, s.get_stream());
#endif
      } else {
        ptr->build_cpu_image_pyramid(source, false);
        ptr->build_cpu_gradient_pyramid(false);
      }
      images_ptrs[cam_id] = ptr;
    }
    CALLBACK_AND_RETURN_IF(images_ptrs.empty(), callback, Pose, "No valid images to localize");

    // Use current frame for localization
    const slam::AsyncSlam::VOTrackData& track_data = impl->async_slam_->GetVOTrackData();
    slam::AsyncSlam::LocalizationResult result;
    localizer->LocalizeSync(isometry_guess_pose, current_pose, track_data, images_ptrs, result);
    impl->SendLocalizationResult(result, callback);
  } else {
    // Start localization in a separate thread
    localizer->StartLocalizationThread(isometry_guess_pose, current_pose);
    impl->localizer_response_ = std::move(callback);
    impl->async_localizer_ = std::move(localizer);
  }
}

void Slam::GetSlamMetrics(Metrics& metrics) const {
  slam::AsyncSlamLCTelemetry telemetry;

  if (!impl->async_slam_->GetLastTelemetry(telemetry)) {
    throw std::runtime_error("Failed to get SLAM metrics");
  }

  // Convert metrics
  metrics.timestamp_ns = telemetry.timestamp_ns;
  metrics.lc_status = telemetry.lc_status;
  metrics.pgo_status = telemetry.pgo_status;
  metrics.lc_selected_landmarks_count = telemetry.lc_selected_landmarks_count;
  metrics.lc_tracked_landmarks_count = telemetry.lc_tracked_landmarks_count;
  metrics.lc_pnp_landmarks_count = telemetry.lc_pnp_landmarks_count;
  metrics.lc_good_landmarks_count = telemetry.lc_good_landmarks_count;
}

void Slam::GetLoopClosurePoses(std::vector<PoseStamped>& poses) const {
  // Get loop closure poses directly from AsyncSlam
  const std::list<slam::LoopClosureStamped>& last_loop_closures_stamped =
      impl->async_slam_->GetLastLoopClosuresStamped();

  // Resize output vector to match the number of loop closures
  poses.resize(last_loop_closures_stamped.size());

  // Convert each loop closure pose to the public API format
  size_t i = 0;
  for (const auto& lc_stamped : last_loop_closures_stamped) {
    poses[i].timestamp_ns = lc_stamped.timestamp_ns;
    poses[i].pose = ConvertIsometryToPose(OpencvFromCuvslam(lc_stamped.pose));
    i++;
  }
}

void Slam::EnableReadingData(DataLayer layer, uint32_t max_items_count) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  switch (layer) {
    case DataLayer::Map:
    case DataLayer::LoopClosure:
    case DataLayer::LocalizerMap:
    case DataLayer::LocalizerLandmarks:
    case DataLayer::LocalizerLoopClosure:
    case DataLayer::Landmarks:
      assert(impl->landmarks_views_[ToUnderlying(layer)]);
      impl->landmarks_views_[ToUnderlying(layer)]->init(2, max_items_count);
      break;
    case DataLayer::PoseGraph:
      assert(impl->pose_graph_view_);
      impl->pose_graph_view_->init(2, max_items_count);
      break;
    case DataLayer::LocalizerProbes:
      assert(impl->localizer_probes_view_);
      impl->localizer_probes_view_->init(2, max_items_count);
      break;
    default:
      throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  }
}

void Slam::DisableReadingData(DataLayer layer) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  switch (layer) {
    case DataLayer::Map:
    case DataLayer::LoopClosure:
    case DataLayer::LocalizerMap:
    case DataLayer::LocalizerLandmarks:
    case DataLayer::LocalizerLoopClosure:
    case DataLayer::Landmarks:
      assert(impl->landmarks_views_[ToUnderlying(layer)]);
      impl->landmarks_views_[ToUnderlying(layer)]->reset();
      break;
    case DataLayer::PoseGraph:
      assert(impl->pose_graph_view_);
      impl->pose_graph_view_->reset();
      break;
    case DataLayer::LocalizerProbes:
      assert(impl->localizer_probes_view_);
      impl->localizer_probes_view_->reset();
      break;
    default:
      throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  }
}

std::shared_ptr<const Slam::Landmarks> Slam::ReadLandmarks(DataLayer layer) {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  const auto layer_index = ToUnderlying(layer);
  if (layer_index >= Slam::Impl::kMaxDataLayer) {
    throw std::runtime_error("Invalid data layer: " + std::to_string(ToUnderlying(layer)));
  } else if (layer == DataLayer::PoseGraph) {
    throw std::runtime_error("For DataLayer::PoseGraph use ReadPoseGraph() instead");
  } else if (layer == DataLayer::LocalizerProbes) {
    throw std::runtime_error("For DataLayer::LocalizerProbes use ReadLocalizerProbes() instead");
  }
  assert(impl->landmarks_views_[layer_index]);
  auto latest = impl->landmarks_views_[layer_index]->acquire_latest();
  if (latest) {
    impl->landmarks_[layer_index]->timestamp_ns = latest->timestamp_ns;
    impl->landmarks_[layer_index]->landmarks.clear();
    impl->landmarks_[layer_index]->landmarks.reserve(latest->landmarks.size());
    for (const auto& lm : latest->landmarks) {
      impl->landmarks_[layer_index]->landmarks.push_back({lm.id, lm.weight, OpencvFromCuvslam(lm.coords)});
    }
  }
  return impl->landmarks_[layer_index];
}

std::shared_ptr<const Slam::PoseGraph> Slam::ReadPoseGraph() {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  assert(impl->pose_graph_view_);
  auto latest = impl->pose_graph_view_->acquire_latest();
  if (latest) {
    impl->pose_graph_->timestamp_ns = latest->timestamp_ns;
    impl->pose_graph_->nodes.clear();
    impl->pose_graph_->edges.clear();
    impl->pose_graph_->nodes.reserve(latest->nodes.size());
    impl->pose_graph_->edges.reserve(latest->edges.size());
    for (const auto& node : latest->nodes) {
      impl->pose_graph_->nodes.push_back({node.id, ConvertIsometryToPose(OpencvFromCuvslam(node.node_pose))});
    }
    for (const auto& edge : latest->edges) {
      PoseCovariance covariance;
      mat<6>(covariance) = edge.covariance;
      impl->pose_graph_->edges.push_back(
          {edge.node_from, edge.node_to, ConvertIsometryToPose(OpencvFromCuvslam(edge.transform)), covariance});
    }
  }
  return impl->pose_graph_;
}

std::shared_ptr<const Slam::LocalizerProbes> Slam::ReadLocalizerProbes() {
  if (!impl->enable_reading_internals_) {
    throw std::runtime_error("Reading data is not available, set enable_reading_internals flag in Slam::Config");
  }
  assert(impl->localizer_probes_view_);
  auto latest = impl->localizer_probes_view_->acquire_latest();
  if (latest) {
    impl->localizer_probes_->timestamp_ns = latest->timestamp_ns;
    impl->localizer_probes_->size = latest->size;
    impl->localizer_probes_->probes.clear();
    impl->localizer_probes_->probes.reserve(latest->probes.size());
    for (const auto& probe : latest->probes) {
      impl->localizer_probes_->probes.push_back({probe.id, ConvertIsometryToPose(OpencvFromCuvslam(probe.guess_pose)),
                                                 ConvertIsometryToPose(OpencvFromCuvslam(probe.exact_result_pose)),
                                                 probe.weight, probe.exact_result_weight, probe.solved});
    }
  }
  return impl->localizer_probes_;
}
void Slam::MergeMaps(const Rig& rig, const std::vector<std::string_view>& databases,
                     const std::string_view& output_folder) {
  // Get camera rig from tracker
  camera::Rig internal_rig;
  std::vector<std::unique_ptr<camera::ICameraModel>> cameras_models;
  SetTrackerRigAndIntrinsics(cameras_models, internal_rig, rig.cameras);

  std::vector<std::string> databases_str(databases.begin(), databases.end());
  if (!slam::MergeDatabases(internal_rig, databases_str, std::string{output_folder})) {
    throw std::runtime_error("Failed to merge maps");
  }
}

}  // namespace cuvslam
