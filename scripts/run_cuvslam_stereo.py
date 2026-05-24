import rerun.blueprint as rrb
import rerun as rr
import numpy as np
import argparse
import cuvslam
import cv2
import os
import sys
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ==============================================================================
# НАСТРОЙКИ (КОНСТАНТЫ)
# ==============================================================================
BAG_PATH = "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/VO/2025_01_22-14_06_00-hall_VO_V1_0.db3"

LEFT_TOPIC = "/zedx_45327000/zed_node/left/image_rect_color/compressed"
RIGHT_TOPIC = "/zedx_45327000/zed_node/right/image_rect_color/compressed"

IMAGE_JITTER_THRESHOLD_NS = 40 * 1e6
CAM_WIDTH = 960
CAM_HEIGHT = 600
CAM_FX = 361.542236328125
CAM_FY = 361.542236328125
CAM_CX = 485.6068115234375
CAM_CY = 311.95965576171875
CAM_BASELINE = 0.12
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
    print(f"Проверка ROS 2 bag: {bag_path}")
    if not os.path.exists(bag_path):
        print("[ERROR] Файл bag не найден!")
        sys.exit(1)
    
    with Reader(bag_path) as reader:
        print("Доступные топики в bag:")
        for connection in reader.connections:
            print(f" - {connection.topic} ({connection.msgtype})")
            
def main(): 
    print("\nЗапуск CuVSLAM в Stereo режиме (чтение через rosbags)...\n")  

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=int, default=1, help="Включить визуализацию Rerun (1) или выключить (0)")
    args, _ = parser.parse_known_args()
    verbose = bool(args.verbose)

    check_bag(BAG_PATH)

    # ---------------------------------------------------------------------
    # Настройка Rerun
    # ---------------------------------------------------------------------
    if verbose:
        rr.init('vslam_stereo_rosbags', strict=True, spawn=True)
        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(contents=[
                rrb.Vertical(contents=[
                    rrb.Spatial2DView(origin='world/camera/left', name='Left Camera'),
                    rrb.Spatial2DView(origin='world/camera/right', name='Right Camera')
                ]),
                rrb.Spatial3DView(
                    name="3D",
                    defaults=[rr.components.ImagePlaneDistance(0.5)]
                )
            ]),
        ), make_active=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # ---------------------------------------------------------------------
    # Настройка камер CuVSLAM (используем константы)
    # ---------------------------------------------------------------------
    left_cam = cuvslam.Camera()
    left_cam.size = (CAM_WIDTH, CAM_HEIGHT)
    left_cam.principal = [CAM_CX, CAM_CY]
    left_cam.focal = [CAM_FX, CAM_FY]

    left_pose = cuvslam.Pose()
    left_pose.translation = [0.0, 0.0, 0.0]
    left_pose.rotation = [0.0, 0.0, 0.0, 1.0] # Кватернион [x, y, z, w]
    left_cam.rig_from_camera = left_pose

    right_cam = cuvslam.Camera()
    right_cam.size = (CAM_WIDTH, CAM_HEIGHT)
    right_cam.principal = [CAM_CX, CAM_CY]
    right_cam.focal = [CAM_FX, CAM_FY]

    right_pose = cuvslam.Pose()
    right_pose.translation = [CAM_BASELINE, 0.0, 0.0]
    right_pose.rotation = [0.0, 0.0, 0.0, 1.0]
    right_cam.rig_from_camera = right_pose

    # ---------------------------------------------------------------------
    # Конфигурация трекера (Стерео)
    # ---------------------------------------------------------------------
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        odometry_mode=cuvslam.Tracker.OdometryMode(0)  # 2: Stereo Mode
    )

    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=False)
    rig = cuvslam.Rig([left_cam, right_cam])
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)

    # ---------------------------------------------------------------------
    # Переменные для главного цикла и буфер синхронизации
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

    # Буфер для сборки левых и правых кадров с одинаковым timestamp
    sync_buffer = {}
    MAX_BUFFER_SIZE = 15

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"\nНачинаем чтение кадров из {BAG_PATH}...")
    
    with Reader(BAG_PATH) as reader:
        # Фильтруем коннекты, чтобы читать только наши картинки
        connections = [x for x in reader.connections if x.topic in [LEFT_TOPIC, RIGHT_TOPIC]]
        
        for connection, timestamp_bag_ns, rawdata in reader.messages(connections=connections):
            
            # Десериализация сообщения
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            # Вытаскиваем точный timestamp из хедера сообщения (лучше для синхронизации, чем время записи bag-а)
            msg_time_ns = msg.header.stamp.sec * int(1e9) + msg.header.stamp.nanosec

            if msg_time_ns not in sync_buffer:
                sync_buffer[msg_time_ns] = {}

            # Декодируем CompressedImage (JPEG) в NumPy BGR
            img_array = np.frombuffer(msg.data, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if connection.topic == LEFT_TOPIC:
                sync_buffer[msg_time_ns]['left'] = img_bgr
            elif connection.topic == RIGHT_TOPIC:
                sync_buffer[msg_time_ns]['right'] = img_bgr

            # Очистка старых фреймов в буфере (на случай если одна из камер пропустила кадр)
            if len(sync_buffer) > MAX_BUFFER_SIZE:
                oldest_key = min(sync_buffer.keys())
                del sync_buffer[oldest_key]

            # Если мы получили оба кадра для данного timestamp
            if 'left' in sync_buffer[msg_time_ns] and 'right' in sync_buffer[msg_time_ns]:
                left_img = sync_buffer[msg_time_ns]['left']
                right_img = sync_buffer[msg_time_ns]['right']
                
                # Удаляем из буфера, так как они нам больше не нужны
                del sync_buffer[msg_time_ns]

                # Проверка на джиттер (пропуски)
                if prev_timestamp is not None:
                    diff_ns = msg_time_ns - prev_timestamp
                    if diff_ns > IMAGE_JITTER_THRESHOLD_NS:
                        print(f"[WARN] Пропуск времени: {diff_ns/1e6:.2f} мс превышает порог.")

                # =========================================================
                # ТРЕКИНГ CUVSLAM
                # =========================================================
                odom_pose_estimate, slam_pose = tracker.track(
                    msg_time_ns, images=[left_img, right_img]
                )
                
                if verbose:
                    rr.set_time_sequence('frame', frame_id)
                    rr.log('world/camera/left', rr.Image(left_img).compress(jpeg_quality=80))
                    rr.log('world/camera/right', rr.Image(right_img).compress(jpeg_quality=80))

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

    print("\n[INFO] Чтение bag-файла завершено.")

if __name__ == '__main__': 
    main()