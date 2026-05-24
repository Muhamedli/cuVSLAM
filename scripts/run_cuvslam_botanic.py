#!/usr/bin/env python3
"""
Stereo SLAM на BotanicGarden dataset (Dalsa gray cameras, PINHOLE + Brown distortion).

Использование:
  python scripts/run_cuvslam_botanic.py
  python scripts/run_cuvslam_botanic.py --verbose 0
"""

import struct
import argparse
import bz2
import os
import sys

import cv2
import numpy as np
import yaml
import cuvslam
import rerun as rr
import rerun.blueprint as rrb

# ── Константы ────────────────────────────────────────────────────────────────
BAG_PATH = "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/Botanic_garden/1018_00_VLIO.bag"
GT_BAG_PATH = "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/Botanic_garden/gt_1018_00_qnew.bag"
CALIB_DIR = "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/Botanic_garden/BotanicGarden/calib"
LEFT_TOPIC = "/dalsa_gray/left/image_raw"
RIGHT_TOPIC = "/dalsa_gray/right/image_raw"
SYNC_TOLERANCE_NS = 1000000   # 5 мс
MAX_BUFFER_SIZE = 30

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


# ── Утилиты: кватернионы / матрицы ───────────────────────────────────────────

def mat_to_quat(R):
    """Матрица вращения 3×3 → кватернион (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s, 0.25/s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    if R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
    return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s, (R[1,0]-R[0,1])/s])

def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def pose_to_mat(pose):
    T = np.eye(4)
    T[:3,:3] = quat_to_mat(pose.rotation)
    T[:3,3] = pose.translation
    return T

class SimplePose:
    def __init__(self, t, r):
        self.translation = t
        self.rotation = r

def mat_to_simple_pose(T):
    return SimplePose(T[:3,3], mat_to_quat(T[:3,:3]))


# ── Загрузка калибровки BotanicGarden ────────────────────────────────────────

def load_opencv_yaml(path):
    """Загружает YAML с тегами opencv-matrix (пропускаем %YAML:1.0 и теги)."""
    with open(path) as f:
        text = f.read()
    # Убираем %YAML:1.0 и opencv-matrix теги для PyYAML
    text = text.replace('%YAML:1.0', '').replace('!!opencv-matrix', '')
    return yaml.safe_load(text)


def load_botanic_calibration(calib_dir):
    """
    Загружает интринсики dalsa_gray0/1, экстринсику T_gray0_gray1
    и T_xsens_gray0 (камера → IMU).
    Возвращает (left_cfg, right_cfg, T_left_from_right, T_xsens_gray0).
    """
    # Интринсики
    left_data = load_opencv_yaml(os.path.join(calib_dir, 'camera_intrinsics', 'dalsa_gray0_down.yaml'))
    right_data = load_opencv_yaml(os.path.join(calib_dir, 'camera_intrinsics', 'dalsa_gray1_down.yaml'))

    def parse_intrinsics(d):
        dp = d['distortion_parameters']
        pp = d['projection_parameters']
        return {
            'res': [d['image_width'], d['image_height']],
            'focal': [pp['fx'], pp['fy']],
            'principal': [pp['cx'], pp['cy']],
            # Brown: k1, k2, k3, p1, p2
            'distortion': [dp['k1'], dp['k2'], 0.0, dp['p1'], dp['p2']],
        }

    left_cfg = parse_intrinsics(left_data)
    right_cfg = parse_intrinsics(right_data)

    # Экстринсики
    ext_data = load_opencv_yaml(os.path.join(calib_dir, 'extrinsics', 'calib_chain.yaml'))

    # T_gray0_gray1 (right camera in left camera frame)
    T_lr = np.array(ext_data['T_gray0_gray1']['data']).reshape(4, 4)

    # T_xsens_gray0 (gray0 → Xsens/IMU frame)
    T_xsens_gray0 = np.array(ext_data['T_xsens_gray0']['data']).reshape(4, 4)

    print(f"Left:  res={left_cfg['res']}, f={left_cfg['focal']}")
    print(f"Right: res={right_cfg['res']}, f={right_cfg['focal']}")
    print(f"Baseline: {np.linalg.norm(T_lr[:3,3]):.4f} m")

    return left_cfg, right_cfg, T_lr, T_xsens_gray0


# ── ROS1 bag парсер (minimal) ────────────────────────────────────────────────

def read_header_fields(buf):
    fields = {}
    off = 0
    while off < len(buf):
        flen = struct.unpack_from('<I', buf, off)[0]; off += 4
        field = buf[off:off + flen]; off += flen
        eq = field.index(b'=')
        fields[field[:eq].decode()] = field[eq+1:]
    return fields


def parse_ros1_string(data, off):
    slen = struct.unpack_from('<I', data, off)[0]
    return data[off+4:off+4+slen].decode(), off + 4 + slen


def try_parse_image(data):
    """Парсит sensor_msgs/Image → (ts_ns, h, w, encoding, pixels) или None."""
    try:
        off = 0
        off += 4  # seq
        sec, nsec = struct.unpack_from('<II', data, off); off += 8
        _, off = parse_ros1_string(data, off)  # frame_id
        h, w = struct.unpack_from('<II', data, off); off += 8
        enc, off = parse_ros1_string(data, off)
        off += 1  # is_bigendian
        step = struct.unpack_from('<I', data, off)[0]; off += 4
        plen = struct.unpack_from('<I', data, off)[0]; off += 4
        pix = data[off:off + plen]
        if h == 0 or w == 0 or h > 10000 or w > 10000:
            return None
        return sec * 1_000_000_000 + nsec, h, w, enc, pix
    except Exception:
        return None


def read_bag_connections(path):
    """Читает Connection-записи из индексной зоны bag-файла."""
    conns = {}
    with open(path, 'rb') as f:
        f.readline()  # magic
        hdr_len = struct.unpack('<I', f.read(4))[0]
        hdr = f.read(hdr_len)
        dlen = struct.unpack('<I', f.read(4))[0]
        f.read(dlen)
        flds = read_header_fields(hdr)
        idx_pos = struct.unpack('<Q', flds['index_pos'])[0]
        f.seek(idx_pos)
        for _ in range(10000):
            b4 = f.read(4)
            if len(b4) < 4: break
            hl = struct.unpack('<I', b4)[0]
            hb = f.read(hl)
            dl = struct.unpack('<I', f.read(4))[0]
            ff = read_header_fields(hb)
            op = struct.unpack('<B', ff.get('op', b'\x00'))[0]
            if op == 7 and 'conn' in ff and 'topic' in ff:
                cid = struct.unpack('<I', ff['conn'])[0]
                conns[cid] = ff['topic'].decode()
            f.read(dl)
    return conns


def iter_bag_stereo(bag_path, left_cids, right_cids):
    """
    Итератор: выдаёт синхронные стереопары (ts_ns, left_rgb, right_rgb).
    """
    target_cids = left_cids | right_cids
    left_buf, right_buf = {}, {}

    with open(bag_path, 'rb') as f:
        f.readline()
        while True:
            b4 = f.read(4)
            if len(b4) < 4: break
            hdr_len = struct.unpack('<I', b4)[0]
            hdr = f.read(hdr_len)
            dlen = struct.unpack('<I', f.read(4))[0]
            flds = read_header_fields(hdr)
            op = struct.unpack('<B', flds.get('op', b'\x00'))[0]

            if op != 5:  # not CHUNK
                f.read(dlen)
                continue

            comp = flds.get('compression', b'none').decode()
            chunk = f.read(dlen)
            if comp == 'bz2':
                chunk = bz2.decompress(chunk)
            elif comp == 'lz4':
                if not HAS_LZ4: continue
                chunk = lz4f.decompress(chunk)
            elif comp != 'none':
                continue

            # Парсим записи внутри chunk
            coff = 0
            while coff + 8 < len(chunk):
                rhl = struct.unpack_from('<I', chunk, coff)[0]; coff += 4
                if coff + rhl + 4 > len(chunk): break
                rh = chunk[coff:coff+rhl]; coff += rhl
                rdl = struct.unpack_from('<I', chunk, coff)[0]; coff += 4
                if coff + rdl > len(chunk): break
                rd = chunk[coff:coff+rdl]; coff += rdl

                rf = read_header_fields(rh)
                rop = struct.unpack('<B', rf.get('op', b'\x00'))[0]
                if rop != 2 or 'conn' not in rf: continue
                cid = struct.unpack('<I', rf['conn'])[0]
                if cid not in target_cids: continue

                result = try_parse_image(rd)
                if result is None: continue
                ts_ns, h, w, enc, pix = result

                # Конвертация в RGB
                raw = np.frombuffer(pix, np.uint8)
                if enc in ('mono8', '8UC1'):
                    img = cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2RGB)
                elif enc in ('bgr8', '8UC3'):
                    img = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
                elif enc == 'rgb8':
                    img = raw.reshape(h, w, 3).copy()
                elif enc.startswith('bayer_'):
                    bayer = {
                        'bayer_rggb8': cv2.COLOR_BayerRG2RGB,
                        'bayer_bggr8': cv2.COLOR_BayerBG2RGB,
                        'bayer_gbrg8': cv2.COLOR_BayerGB2RGB,
                        'bayer_grbg8': cv2.COLOR_BayerGR2RGB,
                    }
                    if enc not in bayer: continue
                    img = cv2.cvtColor(raw.reshape(h, w), bayer[enc])
                else:
                    continue

                # Буферизация + синхронизация
                is_left = cid in left_cids
                if is_left:
                    left_buf[ts_ns] = img
                    other = right_buf
                else:
                    right_buf[ts_ns] = img
                    other = left_buf

                best_ts, best_d = None, SYNC_TOLERANCE_NS
                for ots in other:
                    d = abs(ts_ns - ots)
                    if d < best_d:
                        best_d, best_ts = d, ots

                for buf in (left_buf, right_buf):
                    while len(buf) > MAX_BUFFER_SIZE:
                        del buf[min(buf)]

                if best_ts is None: continue

                if is_left:
                    l = left_buf.pop(ts_ns); r = right_buf.pop(best_ts)
                else:
                    r = right_buf.pop(ts_ns); l = left_buf.pop(best_ts)

                yield min(ts_ns, best_ts), l, r


# ── Загрузка GT-траектории из bag ─────────────────────────────────────────────

def parse_pose_stamped(data):
    """
    Парсит бинарные данные geometry_msgs/PoseStamped (ROS1).
    Возвращает (ts_ns, position[3], orientation_xyzw[4]).
    """
    off = 0
    off += 4  # seq
    sec, nsec = struct.unpack_from('<II', data, off); off += 8
    _, off = parse_ros1_string(data, off)  # frame_id
    # pose.position (3 × float64)
    px, py, pz = struct.unpack_from('<ddd', data, off); off += 24
    # pose.orientation (4 × float64: x, y, z, w)
    ox, oy, oz, ow = struct.unpack_from('<dddd', data, off)
    return sec * 1_000_000_000 + nsec, [px, py, pz], [ox, oy, oz, ow]


def load_gt_trajectory(gt_bag_path):
    """
    Читает GT-траекторию (PoseStamped) из ROS1 bag.
    Возвращает list of (ts_ns, np.array[3]).
    """
    if not os.path.exists(gt_bag_path):
        print(f"[WARN] GT bag не найден: {gt_bag_path}")
        return []

    gt_points = []
    with open(gt_bag_path, 'rb') as f:
        f.readline()  # magic
        while True:
            b4 = f.read(4)
            if len(b4) < 4: break
            hdr_len = struct.unpack('<I', b4)[0]
            hdr = f.read(hdr_len)
            dlen = struct.unpack('<I', f.read(4))[0]
            flds = read_header_fields(hdr)
            op = struct.unpack('<B', flds.get('op', b'\x00'))[0]

            if op != 5:  # not CHUNK
                f.read(dlen)
                continue

            comp = flds.get('compression', b'none').decode()
            chunk = f.read(dlen)
            if comp == 'bz2':
                chunk = bz2.decompress(chunk)
            elif comp == 'lz4':
                if not HAS_LZ4: continue
                chunk = lz4f.decompress(chunk)
            elif comp != 'none':
                continue

            coff = 0
            while coff + 8 < len(chunk):
                rhl = struct.unpack_from('<I', chunk, coff)[0]; coff += 4
                if coff + rhl + 4 > len(chunk): break
                rh = chunk[coff:coff+rhl]; coff += rhl
                rdl = struct.unpack_from('<I', chunk, coff)[0]; coff += 4
                if coff + rdl > len(chunk): break
                rd = chunk[coff:coff+rdl]; coff += rdl

                rf = read_header_fields(rh)
                rop = struct.unpack('<B', rf.get('op', b'\x00'))[0]
                if rop != 2: continue

                try:
                    ts_ns, pos, _ = parse_pose_stamped(rd)
                    gt_points.append((ts_ns, np.array(pos)))
                except Exception:
                    continue

    print(f"GT: загружено {len(gt_points)} поз")
    return gt_points


# ── Визуализация ─────────────────────────────────────────────────────────────

def color_from_id(i):
    return [(i*17)%256, (i*31)%256, (i*47)%256]

def init_rerun():
    rr.init('vslam_botanic_stereo', strict=True, spawn=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Horizontal(contents=[
            rrb.Vertical(contents=[
                rrb.Spatial2DView(origin='cam/left',  name='Left'),
                rrb.Spatial2DView(origin='cam/right', name='Right'),
            ]),
            rrb.Spatial3DView(name="3D", defaults=[rr.components.ImagePlaneDistance(0.5)]),
        ]),
    ), make_active=True)
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CuVSLAM Stereo — BotanicGarden")
    parser.add_argument("--verbose", type=int, default=1, help="1=Rerun, 0=без")
    args, _ = parser.parse_known_args()
    verbose = bool(args.verbose)

    if not os.path.exists(BAG_PATH):
        print(f"[ERROR] Bag не найден: {BAG_PATH}"); sys.exit(1)

    # ── Калибровка ───────────────────────────────────────────────────────
    left_cfg, right_cfg, T_lr, T_xsens_gray0 = load_botanic_calibration(CALIB_DIR)
    right_t = T_lr[:3,3].tolist()
    right_r = mat_to_quat(T_lr[:3,:3]).tolist()

    # Для преобразования VSLAM (gray0) → GT (Xsens/IMU):
    # T_imu(t) = T_xsens_gray0 @ T_vslam(t) @ inv(T_xsens_gray0)
    T_gray0_xsens = np.linalg.inv(T_xsens_gray0)

    # ── cuvslam камеры (Brown distortion) ────────────────────────────────
    def make_cam(cfg, trans, rot):
        cam = cuvslam.Camera()
        cam.size = cfg['res']
        cam.focal = cfg['focal']
        cam.principal = cfg['principal']
        cam.distortion = cuvslam.Distortion(
            cuvslam.Distortion.Model.Brown, cfg['distortion']
        )
        p = cuvslam.Pose()
        p.translation = trans
        p.rotation = rot
        cam.rig_from_camera = p
        return cam

    left_cam  = make_cam(left_cfg,  [0.,0.,0.], [0.,0.,0.,1.])
    right_cam = make_cam(right_cfg, right_t, right_r)

    # ── Трекер ───────────────────────────────────────────────────────────
    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=False,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        odometry_mode=cuvslam.Tracker.OdometryMode(0),
    )
    slam_cfg = cuvslam.Tracker.SlamConfig(sync_mode=False)
    tracker = cuvslam.Tracker(cuvslam.Rig([left_cam, right_cam]), odom_cfg, slam_cfg)

    if verbose:
        init_rerun()

    # ── GT-траектория ────────────────────────────────────────────────────
    gt_data = load_gt_trajectory(GT_BAG_PATH)
    if verbose and gt_data:
        gt_pts = np.array([p for _, p in gt_data])
        # GT: сдвигаем начало в 0 (GT может быть в глобальной СК)
        gt_pts -= gt_pts[0]
        rr.log('trajectory_gt', rr.LineStrips3D([gt_pts.tolist()],
               colors=[[0, 200, 0]]), static=True)
        print(f"GT траектория отрисована ({len(gt_pts)} точек)")

    # ── Connections → conn_ids ───────────────────────────────────────────
    conns = read_bag_connections(BAG_PATH)
    left_cids  = {cid for cid, t in conns.items() if t == LEFT_TOPIC}
    right_cids = {cid for cid, t in conns.items() if t == RIGHT_TOPIC}
    print(f"Left cids: {left_cids}, Right cids: {right_cids}")

    if not left_cids or not right_cids:
        print("[ERROR] Не найдены image-топики в баге")
        sys.exit(1)

    # ── Состояние ────────────────────────────────────────────────────────
    frame_id = 0
    traj_odom, traj_slam = [], []
    last_pose, last_slam_t = None, None
    tracking_lost = False
    offset_T = np.eye(4)
    last_world_T = np.eye(4)

    # ── Основной цикл ───────────────────────────────────────────────────
    print(f"\nCuVSLAM Stereo — BotanicGarden\nЧтение: {BAG_PATH}\n")

    for sync_ts, left_img, right_img in iter_bag_stereo(BAG_PATH, left_cids, right_cids):
        odom_est, slam_pose = tracker.track(sync_ts, images=[left_img, right_img])

        observations = {}
        if odom_est.world_from_rig is None:
            if not tracking_lost:
                print(f"[WARN] Потеря трекинга, кадр {frame_id}")
            tracking_lost = True
            active_pose = last_pose
            active_slam_t = last_slam_t
        else:
            cur_T = pose_to_mat(odom_est.world_from_rig.pose)
            if tracking_lost and last_pose is not None:
                offset_T = last_world_T @ np.linalg.inv(cur_T)
            tracking_lost = False
            world_T = offset_T @ cur_T
            last_world_T = world_T

            # Преобразование из СК камеры (gray0) в СК IMU (Xsens)
            imu_T = T_xsens_gray0 @ world_T @ T_gray0_xsens
            active_pose = mat_to_simple_pose(imu_T)

            active_slam_t = None
            if slam_pose is not None and slam_pose.translation is not None:
                pt_cam = offset_T @ np.append(slam_pose.translation, 1.0)
                active_slam_t = (T_xsens_gray0 @ pt_cam)[:3]

            last_pose = active_pose
            last_slam_t = active_slam_t

            for idx, name in enumerate(['left', 'right']):
                obs = tracker.get_last_observations(idx)
                observations[name] = {
                    'uv': [[o.u, o.v] for o in obs],
                    'colors': [color_from_id(o.id) for o in obs],
                }

        if active_pose is not None:
            traj_odom.append(active_pose.translation)
            if active_slam_t is not None:
                traj_slam.append(active_slam_t)

        if verbose:
            rr.set_time_sequence('frame', frame_id)
            rr.log('cam/left',  rr.Image(left_img).compress(jpeg_quality=80))
            rr.log('cam/right', rr.Image(right_img).compress(jpeg_quality=80))

            if active_pose is not None:
                rr.log('trajectory_odom', rr.LineStrips3D(traj_odom))
                if traj_slam:
                    rr.log('trajectory_slam', rr.LineStrips3D(traj_slam))
                rr.log("cam/rig",
                       rr.Transform3D(translation=active_pose.translation,
                                      quaternion=active_pose.rotation),
                       rr.Arrows3D(vectors=np.eye(3)*0.2,
                                   colors=[[255,0,0],[0,255,0],[0,0,255]]))
                for cn, od in observations.items():
                    p = f'cam/{cn}/observations'
                    if od['uv']:
                        rr.log(p, rr.Points2D(od['uv'], radii=5, colors=od['colors']))
                    else:
                        rr.log(p, rr.Clear(recursive=False))

        frame_id += 1
        if frame_id % 50 == 0:
            print(f"  Обработано стереопар: {frame_id}")

    if frame_id == 0:
        print("[WARN] Не обработано ни одной пары.")
    else:
        print(f"\n[OK] Обработано: {frame_id} стереопар")


if __name__ == '__main__':
    main()
