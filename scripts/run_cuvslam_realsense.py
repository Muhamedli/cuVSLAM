import rerun.blueprint as rrb
import rerun as rr
import numpy as np
import argparse
import cuvslam
import os
import sys
import pyrealsense2 as rs

# ==============================================================================
# НАСТРОЙКИ
# ==============================================================================
BAG_PATH = "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/realsense_d405/NLK_4.bag"

# Порог джиттера (в наносекундах)
IMAGE_JITTER_THRESHOLD_NS = 40 * 1e6
# ==============================================================================

def color_from_id(identifier):
    """Генерация цвета на основе ID."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]

def quat_to_mat(q):
    """Конвертация кватерниона [x, y, z, w] в матрицу вращения 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def mat_to_quat(R):
    """Конвертация матрицы вращения 3x3 в кватернион [x, y, z, w]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])

def pose_to_mat(pose):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose.rotation)
    T[:3, 3] = pose.translation
    return T

class MockPose:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

def mat_to_pose(T):
    t = T[:3, 3]
    r = mat_to_quat(T[:3, :3])
    return MockPose(t, r)

def check_bag(bag_path):
    print(f"Проверка файла: {bag_path}")
    if not os.path.exists(bag_path):
        print("[ERROR] Файл bag не найден!")
        sys.exit(1)

def main(): 
    print("\nЗапуск CuVSLAM в Stereo режиме (чтение RealSense .bag)...\n")  

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="Включить визуализацию Rerun (1) или выключить (0)")
    args, _ = parser.parse_known_args()
    verbose = bool(args.verbose)

    check_bag(BAG_PATH)

    # ---------------------------------------------------------------------
    # Настройка Rerun
    # ---------------------------------------------------------------------
    if verbose:
        rr.init('vslam_stereo_realsense', strict=True, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(contents=[
                rrb.Vertical(contents=[
                    rrb.Spatial2DView(origin='world/camera/left', name='Left Camera (Infra 1)'),
                    rrb.Spatial2DView(origin='world/camera/right', name='Right Camera (Infra 2)')
                ]),
                rrb.Spatial3DView(
                    name="3D",
                    defaults=[rr.components.ImagePlaneDistance(0.5)]
                )
            ]),
        ), make_active=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # ---------------------------------------------------------------------
    # Инициализация RealSense Pipeline
    # ---------------------------------------------------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, BAG_PATH)
    
    # Включаем стерео-потоки (Infrared 1 - левая, Infrared 2 - правая)
    config.enable_stream(rs.stream.infrared, 1)
    config.enable_stream(rs.stream.infrared, 2)

    # Запускаем пайплайн для получения интринсиков и экстринсиков напрямую из bag
    profile = pipeline.start(config)
    
    # Отключаем RealSense Real-Time режим, чтобы читать кадры последовательно (как записано)
    playback = profile.get_device().as_playback()
    playback.set_real_time(True)

    # Получаем параметры камеры (Intrinsics) из левого инфракрасного потока
    stream_left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    stream_right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    intrinsics = stream_left.get_intrinsics()
    
    # Экстринсики (Extrinsics) для вычисления baseline
    extrinsics = stream_right.get_extrinsics_to(stream_left)
    cam_baseline = abs(extrinsics.translation[0]) # Обычно ~0.05м для D435

    print("--- Параметры камеры извлечены из bag ---")
    print(f"Разрешение: {intrinsics.width}x{intrinsics.height}")
    print(f"Focal (fx, fy): {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"Principal (cx, cy): {intrinsics.ppx:.2f}, {intrinsics.ppy:.2f}")
    print(f"Baseline: {cam_baseline:.4f} м")
    print("-----------------------------------------")

    # ---------------------------------------------------------------------
    # Настройка камер CuVSLAM 
    # ---------------------------------------------------------------------
    left_cam = cuvslam.Camera()
    left_cam.size = (intrinsics.width, intrinsics.height)
    left_cam.principal = [intrinsics.ppx, intrinsics.ppy]
    left_cam.focal = [intrinsics.fx, intrinsics.fy]

    left_pose = cuvslam.Pose()
    left_pose.translation = [0.0, 0.0, 0.0]
    left_pose.rotation = [0.0, 0.0, 0.0, 1.0] # Кватернион [x, y, z, w]
    left_cam.rig_from_camera = left_pose

    right_cam = cuvslam.Camera()
    right_cam.size = (intrinsics.width, intrinsics.height)
    right_cam.principal = [intrinsics.ppx, intrinsics.ppy]
    right_cam.focal = [intrinsics.fx, intrinsics.fy]

    right_pose = cuvslam.Pose()
    right_pose.translation = [cam_baseline, 0.0, 0.0]
    right_pose.rotation = [0.0, 0.0, 0.0, 1.0]
    right_cam.rig_from_camera = right_pose

    # ---------------------------------------------------------------------
    # Конфигурация трекера (Стерео)
    # ---------------------------------------------------------------------
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        # use_denoising=True,
        odometry_mode=cuvslam.core.Odometry.OdometryMode(0),  # 2: Stereo Mode
        multicam_mode=cuvslam.core.Odometry.MulticameraMode(0)
    )

    slam_cfg = cuvslam.Tracker.SlamConfig(
        sync_mode=True,
        enable_reading_internals=True,
        max_map_size=0,
    )
    
    rig = cuvslam.Rig([left_cam, right_cam])

    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)

    # ---------------------------------------------------------------------
    # Переменные для главного цикла
    # ---------------------------------------------------------------------
    frame_id = 0
    prev_timestamp = None
    trajectory_odom = []
    trajectory_slam = []

    last_valid_odom_pose = None
    last_valid_slam_translation = None
    is_tracking_lost = False
    offset_T = np.eye(4)
    last_valid_world_T = np.eye(4)

    print(f"\nНачинаем чтение кадров из {BAG_PATH}...")
    
    try:
        while True:
            # Чтение кадров с ожиданием. Возвращает False если bag закончился.
            success, frames = pipeline.try_wait_for_frames(timeout_ms=100)
            if not success:
                break
                
            frame_left = frames.get_infrared_frame(1)
            frame_right = frames.get_infrared_frame(2)

            if not frame_left or not frame_right:
                continue

            # Получаем timestamp от RealSense (в миллисекундах) -> переводим в наносекунды
            msg_time_ns = int(frame_left.get_timestamp() * 1e6)

            # Конвертируем 8-битные grayscale инфракрасные кадры в numpy (Y8 -> BGR для Rerun/CuVSLAM)
            img_left_bgr = np.asanyarray(frame_left.get_data())
            img_right_bgr = np.asanyarray(frame_right.get_data())

            # Проверка на джиттер (пропуски)
            if prev_timestamp is not None:
                diff_ns = msg_time_ns - prev_timestamp
                if diff_ns > IMAGE_JITTER_THRESHOLD_NS:
                    print(f"[WARN] Пропуск времени: {diff_ns/1e6:.2f} мс превышает порог.")

            # =========================================================
            # ТРЕКИНГ CUVSLAM
            # =========================================================
            odom_pose_estimate, slam_pose = tracker.track(
                msg_time_ns, images=[img_left_bgr, img_right_bgr]
            )
            
            if verbose:
                rr.set_time_sequence('frame', frame_id)
                rr.log('world/camera/left', rr.Image(img_left_bgr).compress(jpeg_quality=80))
                rr.log('world/camera/right', rr.Image(img_right_bgr).compress(jpeg_quality=80))

            if odom_pose_estimate.world_from_rig is None:
                print(f"[WARN] Потеря трекинга на кадре {frame_id}. Замораживаем позицию.")
                is_tracking_lost = True
                active_odom_pose = last_valid_odom_pose
                active_slam_translation = last_valid_slam_translation
                obs_uv_left, obs_colors_left = [], []
                obs_uv_right, obs_colors_right = [], []
            else:
                current_tracker_T = pose_to_mat(odom_pose_estimate.world_from_rig.pose)
                if is_tracking_lost:
                    if last_valid_odom_pose is not None:
                        offset_T = last_valid_world_T @ np.linalg.inv(current_tracker_T)
                    is_tracking_lost = False
                
                world_T = offset_T @ current_tracker_T
                last_valid_world_T = world_T
                active_odom_pose = mat_to_pose(world_T)
                
                active_slam_translation = slam_pose.translation if slam_pose is not None else None
                if active_slam_translation is not None:
                    pt = np.array([active_slam_translation[0], active_slam_translation[1], active_slam_translation[2], 1.0])
                    world_slam_pt = offset_T @ pt
                    active_slam_translation = world_slam_pt[:3]
                
                last_valid_odom_pose = active_odom_pose
                last_valid_slam_translation = active_slam_translation
                
                # Наблюдения (ключевые точки)
                obs_left = [tracker.get_last_observations(0)]
                obs_uv_left = [[o.u, o.v] for o in obs_left[0]]
                obs_colors_left = [color_from_id(o.id) for o in obs_left[0]]

                obs_right = [tracker.get_last_observations(1)]
                obs_uv_right = [[o.u, o.v] for o in obs_right[0]]
                obs_colors_right = [color_from_id(o.id) for o in obs_right[0]]

            # =========================================================
            # ВИЗУАЛИЗАЦИЯ
            # =========================================================
            if active_odom_pose is not None:
                trajectory_odom.append(active_odom_pose.translation)
                if active_slam_translation is not None:
                    trajectory_slam.append(active_slam_translation)

                if verbose:
                    rr.log('trajectory_odom', rr.LineStrips3D(trajectory_odom))
                    if trajectory_slam:
                        rr.log('trajectory_slam', rr.LineStrips3D(trajectory_slam))

                    rr.log(
                        "world/camera/rig",
                        rr.Transform3D(
                            translation=active_odom_pose.translation,
                            quaternion=active_odom_pose.rotation
                        ),
                        rr.Arrows3D(
                            vectors=np.eye(3) * 0.2,
                            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]] 
                        )
                    )
                    
                    if obs_uv_left:
                        rr.log('world/camera/left/observations', rr.Points2D(obs_uv_left, radii=5, colors=obs_colors_left))
                        rr.log('world/camera/right/observations', rr.Points2D(obs_uv_right, radii=5, colors=obs_colors_right))
                    else:
                        rr.log('world/camera/left/observations', rr.Clear(recursive=False))
                        rr.log('world/camera/right/observations', rr.Clear(recursive=False))
            
            frame_id += 1
            prev_timestamp = msg_time_ns

    except RuntimeError:
        pass # Исключение RuntimeError выбрасывается pipeline, когда bag-файл завершается 
    finally:
        pipeline.stop()
        print("\n[INFO] Чтение bag-файла завершено.")

if __name__ == '__main__': 
    main()