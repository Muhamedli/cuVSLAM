#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

"""
Unified cuVSLAM Runner

This script loads configurations from config/main_params.yaml, config/intrinsics.yaml,
and config/extrinsics.yaml. It supports processing:
1. RealSense bags via pyrealsense2 (with auto-calibration extraction).
2. ROS 2 db3 bags via rosbags.
3. ROS 1 bags via a custom ROS 1 bag parser.
"""

import os
import sys
import struct
import bz2
import argparse
import yaml
import cv2
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import cuvslam
from scipy.spatial.transform import Rotation as R
import queue
import threading
from typing import Dict, List, Optional, Tuple, Any

import cv2
import yaml
import numpy as np
import cuvslam
import rerun as rr
import rerun.blueprint as rrb

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False


# ==============================================================================
# Helper Functions: Rotation & Matrix Conversions
# ==============================================================================

def quat_to_mat(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

def mat_to_quat(R):
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
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

class SimplePose:
    def __init__(self, translation, rotation):
        self.translation = translation
        self.rotation = rotation

def mat_to_simple_pose(T):
    return SimplePose(T[:3, 3], mat_to_quat(T[:3, :3]))

def color_from_id(identifier):
    """Generate color based on ID."""
    return [
        (identifier * 17) % 256,
        (identifier * 31) % 256,
        (identifier * 47) % 256
    ]


# ==============================================================================
# ROS 1 Bag Parser Utilities
# ==============================================================================

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
    """Parse sensor_msgs/Image -> (ts_ns, h, w, encoding, pixels) or None."""
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

def parse_pose_stamped(data):
    """Parse geometry_msgs/PoseStamped (ROS 1)."""
    try:
        off = 0
        off += 4  # seq
        sec, nsec = struct.unpack_from('<II', data, off); off += 8
        _, off = parse_ros1_string(data, off)  # frame_id
        px, py, pz = struct.unpack_from('<ddd', data, off); off += 24
        ox, oy, oz, ow = struct.unpack_from('<dddd', data, off)
        return sec * 1_000_000_000 + nsec, [px, py, pz], [ox, oy, oz, ow]
    except Exception:
        return None

def parse_odometry(data):
    """Parse nav_msgs/Odometry (ROS 1)."""
    try:
        off = 0
        off += 4  # seq
        sec, nsec = struct.unpack_from('<II', data, off); off += 8
        _, off = parse_ros1_string(data, off)  # frame_id
        _, off = parse_ros1_string(data, off)  # child_frame_id
        px, py, pz = struct.unpack_from('<ddd', data, off); off += 24
        ox, oy, oz, ow = struct.unpack_from('<dddd', data, off)
        return sec * 1_000_000_000 + nsec, [px, py, pz], [ox, oy, oz, ow]
    except Exception:
        return None

def parse_tf_message(data):
    """Parse tf2_msgs/TFMessage (ROS 1). Returns list of (ts_ns, frame_id, child_frame_id, pos, quat)."""
    try:
        off = 0
        arr_len = struct.unpack_from('<I', data, off)[0]; off += 4
        transforms = []
        for _ in range(arr_len):
            off += 4  # seq
            sec, nsec = struct.unpack_from('<II', data, off); off += 8
            frame_id, off = parse_ros1_string(data, off)
            child_frame_id, off = parse_ros1_string(data, off)
            px, py, pz = struct.unpack_from('<ddd', data, off); off += 24
            ox, oy, oz, ow = struct.unpack_from('<dddd', data, off); off += 32
            transforms.append((sec * 1_000_000_000 + nsec, frame_id, child_frame_id, [px, py, pz], [ox, oy, oz, ow]))
        return transforms
    except Exception:
        return None

def read_bag_connections(path):
    """Read Connection records from index area of ROS 1 bag file."""
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

def iter_bag_stereo_ros1(bag_path, left_cids, right_cids, gt_cids=None, gt_frame_id=None, gt_child_frame_id=None, sync_tolerance_ns=10000000, max_buffer_size=30):
    """Generator: yields synchronized stereo pairs ('stereo', ts_ns, left_rgb, right_rgb) and GT ('gt', ts_ns, pos)."""
    target_cids = left_cids | right_cids
    if gt_cids:
        target_cids |= gt_cids

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

            # Parse chunk messages
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

                if gt_cids and cid in gt_cids:
                    res = parse_pose_stamped(rd)
                    if res is None:
                        res = parse_odometry(rd)
                    if res is not None:
                        ts_ns, pos, quat = res
                        yield 'gt', ts_ns, pos, quat
                    else:
                        tfs = parse_tf_message(rd)
                        if tfs is not None and gt_frame_id and gt_child_frame_id:
                            for ts_ns, fid, cfid, pos, quat in tfs:
                                fid = fid.lstrip('/')
                                cfid = cfid.lstrip('/')
                                gt_fid = gt_frame_id.lstrip('/')
                                gt_cfid = gt_child_frame_id.lstrip('/')
                                if fid == gt_fid and cfid == gt_cfid:
                                    yield 'gt', ts_ns, pos, quat
                    continue

                result = try_parse_image(rd)
                if result is None: continue
                ts_ns, h, w, enc, pix = result

                # Convert to RGB/BGR (cuVSLAM uses BGR or RGB depending on settings, keeping standard format)
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

                # Buffer and sync
                is_left = cid in left_cids
                if is_left:
                    left_buf[ts_ns] = img
                    other = right_buf
                else:
                    right_buf[ts_ns] = img
                    other = left_buf

                best_ts, best_d = None, sync_tolerance_ns
                for ots in other:
                    d = abs(ts_ns - ots)
                    if d < best_d:
                        best_d, best_ts = d, ots

                for buf in (left_buf, right_buf):
                    while len(buf) > max_buffer_size:
                        del buf[min(buf)]

                if best_ts is None: continue

                if is_left:
                    l = left_buf.pop(ts_ns); r = right_buf.pop(best_ts)
                else:
                    r = right_buf.pop(ts_ns); l = left_buf.pop(best_ts)

                yield 'stereo', min(ts_ns, best_ts), l, r

def iter_bag_mono(bag_path, left_cids, gt_cids=None, gt_frame_id=None, gt_child_frame_id=None):
    """Generator: yields monocular frames ('mono', ts_ns, img) and GT ('gt', ts_ns, pos) from ROS 1 bag."""
    target_cids = set(left_cids)
    if gt_cids:
        target_cids |= gt_cids

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

                if gt_cids and cid in gt_cids:
                    res = parse_pose_stamped(rd)
                    if res is None:
                        res = parse_odometry(rd)
                    if res is not None:
                        ts_ns, pos, quat = res
                        yield 'gt', ts_ns, pos, quat
                    else:
                        tfs = parse_tf_message(rd)
                        if tfs is not None and gt_frame_id and gt_child_frame_id:
                            for ts_ns, fid, cfid, pos, quat in tfs:
                                fid = fid.lstrip('/')
                                cfid = cfid.lstrip('/')
                                gt_fid = gt_frame_id.lstrip('/')
                                gt_cfid = gt_child_frame_id.lstrip('/')
                                if fid == gt_fid and cfid == gt_cfid:
                                    yield 'gt', ts_ns, pos, quat
                    continue

                result = try_parse_image(rd)
                if result is None: continue
                ts_ns, h, w, enc, pix = result

                # Convert to RGB/BGR
                raw = np.frombuffer(pix, np.uint8)
                if enc in ('mono8', '8UC1'):
                    img = cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2RGB)
                elif enc in ('bgr8', '8UC3'):
                    img = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_BGR2RGB)
                elif enc == 'rgb8':
                    img = raw.reshape(h, w, 3).copy()
                else:
                    continue

                yield 'mono', ts_ns, img

def load_gt_trajectory(gt_bag_path, gt_topic=None, gt_frame_id=None, gt_child_frame_id=None):
    """Loads GT trajectory from ROS 1 bag (PoseStamped, Odometry, or TF)."""
    if not os.path.exists(gt_bag_path):
        print(f"[WARN] GT bag not found: {gt_bag_path}")
        return []

    conns = read_bag_connections(gt_bag_path)
    if gt_topic:
        gt_cids = {cid for cid, t in conns.items() if t == gt_topic}
        if not gt_cids:
            print(f"[WARN] GT topic {gt_topic} not found in {gt_bag_path}")
            return []
    else:
        gt_cids = set(conns.keys())

    gt_points = []
    with open(gt_bag_path, 'rb') as f:
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
                if cid not in gt_cids: continue

                res = parse_pose_stamped(rd)
                if res is None:
                    res = parse_odometry(rd)
                if res is not None:
                    ts_ns, pos, _ = res
                    gt_points.append((ts_ns, np.array(pos)))
                else:
                    tfs = parse_tf_message(rd)
                    if tfs is not None and gt_frame_id and gt_child_frame_id:
                        for ts_ns, fid, cfid, pos, quat in tfs:
                            # Normalize leading slashes
                            fid = fid.lstrip('/')
                            cfid = cfid.lstrip('/')
                            gt_fid = gt_frame_id.lstrip('/')
                            gt_cfid = gt_child_frame_id.lstrip('/')

                            if fid == gt_fid and cfid == gt_cfid:
                                gt_points.append((ts_ns, np.array(pos)))

    print(f"[INFO] Loaded {len(gt_points)} GT points")
    return gt_points


# ==============================================================================
# Main Unified Runner
# ==============================================================================

def main(config_dir=None, verbose_override=None, params_override=None, intrinsics_override=None, extrinsics_override=None):
    if config_dir is None:
        parser = argparse.ArgumentParser(description="Unified cuVSLAM Runner")
        parser.add_argument("--config-dir", type=str, default=None, help="Path to config directory (defaults to ../config)")
        parser.add_argument("--verbose", type=int, default=None, help="Override visualization verbose setting (0/1)")
        args, _ = parser.parse_known_args()

        # Determine paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        config_dir = args.config_dir if args.config_dir else os.path.join(project_dir, "config")
        if args.verbose is not None:
            verbose_override = bool(args.verbose)

    main_params_path = os.path.join(config_dir, "main_params.yaml")
    intrinsics_path = os.path.join(config_dir, "intrinsics.yaml")
    extrinsics_path = os.path.join(config_dir, "extrinsics.yaml")

    # Load main parameters
    if not os.path.exists(main_params_path):
        print(f"[ERROR] main_params.yaml not found at {main_params_path}!")
        sys.exit(1)

    with open(main_params_path, "r") as f:
        main_params = yaml.safe_load(f)

    if params_override:
        main_params.update(params_override)

    # Resolve verbose setting
    verbose = bool(main_params.get("verbose", True))
    if verbose_override is not None:
        verbose = bool(verbose_override)

    bag_path = main_params.get("bag_path")
    if not bag_path:
        print("[ERROR] bag_path not defined in main_params.yaml!")
        sys.exit(1)
    if not os.path.exists(bag_path):
        print(f"[ERROR] Bag file not found: {bag_path}")
        sys.exit(1)

    use_realsense_bag = bool(main_params.get("use_realsense_bag", False))
    use_slam = bool(main_params.get("use_slam", True))
    mode_name = main_params.get("mode", "stereo").lower()

    # Map string mode to cuVSLAM OdometryMode
    if mode_name in ("mono", "monocular"):
        odom_mode = cuvslam.Tracker.OdometryMode.Mono
    elif mode_name in ("stereo", "multicamera"):
        odom_mode = cuvslam.Tracker.OdometryMode.Multicamera
    elif mode_name == "inertial":
        odom_mode = cuvslam.Tracker.OdometryMode.Inertial
    elif mode_name == "rgbd":
        odom_mode = cuvslam.Tracker.OdometryMode.RGBD
    else:
        print(f"[ERROR] Unsupported mode: {mode_name}")
        sys.exit(1)

    print(f"[INFO] Running in mode: {mode_name} (OdometryMode: {odom_mode})")
    print(f"[INFO] SLAM enabled: {use_slam}")
    print(f"[INFO] Rerun visualization: {verbose}")
    print(f"[INFO] Loading bag: {bag_path}")

    # Set up camera rig
    rig = None
    if use_realsense_bag:
        if not HAS_REALSENSE:
            print("[ERROR] pyrealsense2 is not installed but use_realsense_bag is true!")
            sys.exit(1)

        print("[INFO] Initializing RealSense device to extract calibration...")
        rs_config = rs.config()
        rs.config.enable_device_from_file(rs_config, bag_path)

        # Configure streams based on mode
        if odom_mode == cuvslam.Tracker.OdometryMode.Mono:
            rs_config.enable_stream(rs.stream.infrared, 1)
        else:
            rs_config.enable_stream(rs.stream.infrared, 1)
            rs_config.enable_stream(rs.stream.infrared, 2)

        if odom_mode == cuvslam.Tracker.OdometryMode.Inertial:
            rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        # Temporary pipeline to query calibration profiles
        temp_pipe = rs.pipeline()
        profile = temp_pipe.start(rs_config)

        # Get stream profiles and intrinsics
        stream_left = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        intrinsics_left = stream_left.get_intrinsics()

        left_cam = cuvslam.Camera()
        left_cam.size = [intrinsics_left.width, intrinsics_left.height]
        left_cam.principal = [intrinsics_left.ppx, intrinsics_left.ppy]
        left_cam.focal = [intrinsics_left.fx, intrinsics_left.fy]
        left_cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
        left_cam.rig_from_camera = cuvslam.Pose(rotation=[0.0, 0.0, 0.0, 1.0], translation=[0.0, 0.0, 0.0])

        cameras = [left_cam]
        imus = []

        if odom_mode != cuvslam.Tracker.OdometryMode.Mono:
            stream_right = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            intrinsics_right = stream_right.get_intrinsics()
            extrinsics_rs = stream_right.get_extrinsics_to(stream_left)

            # Map RS extrinsics to cuVSLAM Pose
            rot_mat = np.array(extrinsics_rs.rotation).reshape([3, 3])
            quat_right = mat_to_quat(rot_mat)

            right_cam = cuvslam.Camera()
            right_cam.size = [intrinsics_right.width, intrinsics_right.height]
            right_cam.principal = [intrinsics_right.ppx, intrinsics_right.ppy]
            right_cam.focal = [intrinsics_right.fx, intrinsics_right.fy]
            right_cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
            right_cam.rig_from_camera = cuvslam.Pose(rotation=quat_right, translation=extrinsics_rs.translation)

            cameras.append(right_cam)

        if odom_mode == cuvslam.Tracker.OdometryMode.Inertial:
            stream_accel = profile.get_stream(rs.stream.accel).as_motion_stream_profile()
            imu_extrinsics = stream_accel.get_extrinsics_to(stream_left)

            imu = cuvslam.ImuCalibration()
            imu.rig_from_imu = cuvslam.Pose(rotation=[0.0, 0.0, 0.0, 1.0], translation=imu_extrinsics.translation)
            imu.gyroscope_noise_density = 6.0673370376614875e-03
            imu.gyroscope_random_walk = 3.6211951458325785e-05
            imu.accelerometer_noise_density = 3.3621979208052800e-02
            imu.accelerometer_random_walk = 9.8256589971851467e-04
            imu.frequency = 200

            imus.append(imu)

        temp_pipe.stop()

        rig = cuvslam.Rig(cameras)
        if imus:
            rig.imus = imus

    else:
        # Load from intrinsics and extrinsics YAML config files
        if intrinsics_override:
            intrinsics_yaml = intrinsics_override
        else:
            if not os.path.exists(intrinsics_path):
                print(f"[ERROR] intrinsics.yaml not found at {intrinsics_path}!")
                sys.exit(1)
            print("[INFO] Loading camera configuration from YAML files...")
            with open(intrinsics_path, "r") as f:
                intrinsics_yaml = yaml.safe_load(f)

        if extrinsics_override:
            extrinsics_yaml = extrinsics_override
        else:
            if not os.path.exists(extrinsics_path):
                print(f"[ERROR] extrinsics.yaml not found at {extrinsics_path}!")
                sys.exit(1)
            with open(extrinsics_path, "r") as f:
                extrinsics_yaml = yaml.safe_load(f)

        cameras = []
        idx = 0
        while True:
            cam_key = f"cam_{idx}"
            if cam_key not in intrinsics_yaml:
                break

            cfg = intrinsics_yaml[cam_key]
            cam = cuvslam.Camera()
            cam.size = [cfg["width"], cfg["height"]]
            cam.principal = [cfg["cx"], cfg["cy"]]
            cam.focal = [cfg["fx"], cfg["fy"]]

            dist_model = cfg.get("distortion_model", "pinhole").lower()
            dist_coeffs = cfg.get("distortion_coefficients", [])

            if dist_model == "pinhole":
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)
            elif dist_model == "brown":
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Brown, dist_coeffs)
            elif dist_model == "fisheye":
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Fisheye, dist_coeffs)
            elif dist_model == "polynomial":
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Polynomial, dist_coeffs)
            else:
                print(f"[WARN] Unknown distortion model {dist_model}, defaulting to Pinhole.")
                cam.distortion = cuvslam.Distortion(cuvslam.Distortion.Model.Pinhole)

            # Retrieve transform from extrinsics
            ext_key = f"rig_from_cam_{idx}"
            transforms = extrinsics_yaml.get("transforms", {})

            p = cuvslam.Pose()
            if ext_key in transforms:
                trans_cfg = transforms[ext_key]
                if isinstance(trans_cfg, list):
                    mat = np.array(trans_cfg)
                    p.translation = mat[:3, 3].tolist()
                    p.rotation = mat_to_quat(mat[:3, :3]).tolist()
                else:
                    p.translation = trans_cfg.get("translation", [0.0, 0.0, 0.0])
                    p.rotation = trans_cfg.get("rotation", [0.0, 0.0, 0.0, 1.0])
            else:
                p.translation = [0.0, 0.0, 0.0]
                p.rotation = [0.0, 0.0, 0.0, 1.0]

            cam.rig_from_camera = p
            cameras.append(cam)
            idx += 1

        if not cameras:
            print("[ERROR] No cameras parsed from intrinsics.yaml!")
            sys.exit(1)

        rig = cuvslam.Rig(cameras)

        # Check for IMU calibration in intrinsics/extrinsics
        if odom_mode == cuvslam.Tracker.OdometryMode.Inertial and "imu_0" in intrinsics_yaml:
            print("[INFO] Loading IMU calibration from YAML files...")
            cfg_imu = intrinsics_yaml["imu_0"]
            imu = cuvslam.ImuCalibration()

            # IMU transform
            transforms = extrinsics_yaml.get("transforms", {})
            p_imu = cuvslam.Pose()
            if "rig_from_imu_0" in transforms:
                t_cfg = transforms["rig_from_imu_0"]
                if isinstance(t_cfg, list):
                    mat = np.array(t_cfg)
                    p_imu.translation = mat[:3, 3].tolist()
                    p_imu.rotation = mat_to_quat(mat[:3, :3]).tolist()
                else:
                    p_imu.translation = t_cfg.get("translation", [0.0, 0.0, 0.0])
                    p_imu.rotation = t_cfg.get("rotation", [0.0, 0.0, 0.0, 1.0])
            imu.rig_from_imu = p_imu

            # Noise parameters
            imu.gyroscope_noise_density = cfg_imu.get("gyroscope_noise_density", 6.0673370376614875e-03)
            imu.gyroscope_random_walk = cfg_imu.get("gyroscope_random_walk", 3.6211951458325785e-05)
            imu.accelerometer_noise_density = cfg_imu.get("accelerometer_noise_density", 3.3621979208052800e-02)
            imu.accelerometer_random_walk = cfg_imu.get("accelerometer_random_walk", 9.8256589971851467e-04)
            imu.frequency = cfg_imu.get("frequency", 200)

            rig.imus = [imu]

    # Initialize cuVSLAM configurations
    odom_hp = main_params

    odom_cfg = cuvslam.Tracker.OdometryConfig(
        async_sba=(odom_mode == cuvslam.Tracker.OdometryMode.Multicamera and use_realsense_bag),
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        odometry_mode=odom_mode
    )

    if "use_denoising" in odom_hp:
        odom_cfg.use_denoising = odom_hp["use_denoising"]
    if "use_motion_model" in odom_hp:
        odom_cfg.use_motion_model = odom_hp["use_motion_model"]
    if "max_frame_delta_s" in odom_hp:
        odom_cfg.max_frame_delta_s = odom_hp["max_frame_delta_s"]

    slam_cfg = None
    if use_slam:
        slam_hp = main_params
        slam_cfg = cuvslam.Tracker.SlamConfig(
            sync_mode=use_realsense_bag, # Sync mode for realsense, async for ROS bags
            enable_reading_internals=use_realsense_bag,
            max_map_size=slam_hp.get("max_map_size", 0)
        )
        if "use_gpu" in slam_hp:
            slam_cfg.use_gpu = slam_hp["use_gpu"]
        if "map_cell_size" in slam_hp:
            slam_cfg.map_cell_size = slam_hp["map_cell_size"]
        if "max_landmarks_distance" in slam_hp:
            slam_cfg.max_landmarks_distance = slam_hp["max_landmarks_distance"]
        if "planar_constraints" in slam_hp:
            slam_cfg.planar_constraints = slam_hp["planar_constraints"]

    # Create VSLAM Tracker
    tracker = cuvslam.Tracker(rig, odom_cfg, slam_cfg)

    # --------------------------------------------------------------------------
    # Rerun Blueprint Setup
    # --------------------------------------------------------------------------
    if verbose:
        blueprint_name = f"vslam_{mode_name}_runner"
        rr.init(blueprint_name, strict=True, spawn=True)

        # Configure panels and layouts
        views = []
        if odom_mode == cuvslam.Tracker.OdometryMode.Mono:
            views.append(rrb.Spatial2DView(origin='world/cuvslam/camera/left', name='Left Camera'))
        else:
            views.append(rrb.Spatial2DView(origin='world/cuvslam/camera/left', name='Left Camera'))
            views.append(rrb.Spatial2DView(origin='world/cuvslam/camera/right', name='Right Camera'))

        rr.send_blueprint(rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(contents=[
                rrb.Vertical(contents=views),
                rrb.Spatial3DView(
                    name="3D",
                    defaults=[rr.components.ImagePlaneDistance(0.5)]
                )
            ]),
        ), make_active=True)
        # Set world to FLU (X forward, Y left, Z up)
        rr.log("world", rr.ViewCoordinates.FLU, static=True)
        # Draw a static triad at the world origin so it's easy to orient
        rr.log("world/origin",
               rr.Arrows3D(
                   vectors=[[1,0,0], [0,1,0], [0,0,1]],
                   origins=[[0,0,0], [0,0,0], [0,0,0]],
                   colors=[[255,0,0], [0,255,0], [0,0,255]],
                   radii=0.01
               ), static=True)
        # Set cuvslam local frame to RDF (Optical: X right, Y down, Z forward)
        rr.log("world/cuvslam", rr.ViewCoordinates.RDF, static=True)

    # --------------------------------------------------------------------------
    # Bag playback and tracking loop
    # --------------------------------------------------------------------------
    frame_id = 0
    prev_timestamp = None
    trajectory_odom = []
    trajectory_slam = []

    last_valid_odom_pose = None
    last_valid_slam_translation = None
    is_tracking_lost = False
    offset_T = np.eye(4)
    last_valid_world_T = np.eye(4)

    # Try to load custom camera to IMU coordinate frames if they exist (Botanic Garden dataset)
    T_xsens_gray0 = None
    T_gray0_xsens = None
    if not use_realsense_bag and not use_realsense_bag:
        with open(extrinsics_path, "r") as f:
            extr_data = yaml.safe_load(f)
            t_data = extr_data.get("transforms", {})
            if "T_xsens_gray0" in t_data:
                T_xsens_gray0 = np.array(t_data["T_xsens_gray0"])
                T_gray0_xsens = np.linalg.inv(T_xsens_gray0)
                print("[INFO] Using custom Botanic Garden coordinate frame transformation T_xsens_gray0.")

    # Load Ground Truth trajectory if specified
    gt_bag_path = main_params.get("gt_bag_path")
    topics_cfg = main_params.get("topics", {})
    gt_topic = topics_cfg.get("gt")
    gt_frame_id = topics_cfg.get("gt_frame_id")
    gt_child_frame_id = topics_cfg.get("gt_child_frame_id")

    # We will accumulate GT points here during playback if parsing on the fly
    trajectory_gt_pts = []
    first_gt_point = None
    first_T_nav_cam_inv = None

    T_gt = None
    if not use_realsense_bag and 'extr_data' in locals() and "transforms" in extr_data:
        if "T_gt" in extr_data["transforms"]:
            T_gt = np.array(extr_data["transforms"]["T_gt"])

    target_gt_bag_path = gt_bag_path if gt_bag_path else bag_path
    inline_gt_parsing = False

    if target_gt_bag_path and (gt_topic or gt_bag_path) and verbose:
        if target_gt_bag_path == bag_path and gt_topic:
            inline_gt_parsing = True
            print("[INFO] Will parse Ground Truth from the main bag on the fly.")
        else:
            print("[INFO] Parsing Ground Truth trajectory from separate bag...")
            gt_data = load_gt_trajectory(target_gt_bag_path, gt_topic, gt_frame_id, gt_child_frame_id)
            if gt_data:
                gt_pts = np.array([p for _, p in gt_data])
                if T_gt is not None and T_gt.shape == (4, 4):
                    print("[INFO] Applying T_gt transform to GT points.")
                    hom_pts = np.hstack((gt_pts, np.ones((gt_pts.shape[0], 1))))
                    gt_pts = (T_gt @ hom_pts.T).T[:, :3]
                gt_pts -= gt_pts[0]
                rr.log('world/cuvslam/trajectory_gt', rr.LineStrips3D([gt_pts.tolist()], colors=[[0, 200, 0]]), static=True)

    # --------------------------------------------------------------------------
    # Playback: Case 1 - RealSense Bag
    # --------------------------------------------------------------------------
    if use_realsense_bag:
        pipeline = rs.pipeline()
        rs_config = rs.config()
        rs.config.enable_device_from_file(rs_config, bag_path)

        if odom_mode == cuvslam.Tracker.OdometryMode.Mono:
            rs_config.enable_stream(rs.stream.infrared, 1)
        else:
            rs_config.enable_stream(rs.stream.infrared, 1)
            rs_config.enable_stream(rs.stream.infrared, 2)

        if odom_mode == cuvslam.Tracker.OdometryMode.Inertial:
            rs_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            rs_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        profile = pipeline.start(rs_config)
        playback = profile.get_device().as_playback()

        print("[INFO] Starting RealSense Bag processing...")
        try:
            while True:
                success, frames = pipeline.try_wait_for_frames(timeout_ms=100)
                if not success:
                    break

                # Extract IMU measurements if in inertial mode
                if odom_mode == cuvslam.Tracker.OdometryMode.Inertial:
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    gyro_frame = frames.first_or_default(rs.stream.gyro)
                    if accel_frame and gyro_frame:
                        current_imu_time = int(accel_frame.timestamp * 1e6)
                        accel_data = accel_frame.as_motion_frame().get_motion_data()
                        gyro_data = gyro_frame.as_motion_frame().get_motion_data()

                        imu_measurement = cuvslam.ImuMeasurement()
                        imu_measurement.timestamp_ns = current_imu_time
                        imu_measurement.linear_accelerations = np.array([accel_data.x, accel_data.y, accel_data.z], dtype=np.float32)
                        imu_measurement.angular_velocities = np.array([gyro_data.x, gyro_data.y, gyro_data.z], dtype=np.float32)
                        tracker.register_imu_measurement(0, imu_measurement)

                # Get infrared frame(s)
                ir_left_frame = frames.get_infrared_frame(1)
                ir_right_frame = frames.get_infrared_frame(2) if odom_mode != cuvslam.Tracker.OdometryMode.Mono else None

                if not ir_left_frame:
                    continue

                msg_time_ns = int(ir_left_frame.timestamp * 1e6)

                img_left = np.asanyarray(ir_left_frame.get_data())
                images = [img_left]
                if ir_right_frame:
                    img_right = np.asanyarray(ir_right_frame.get_data())
                    images.append(img_right)

                # Jitter checks
                if prev_timestamp is not None:
                    diff_ns = msg_time_ns - prev_timestamp
                    if diff_ns > main_params.get("jitter_threshold_ns", 40 * 1e6):
                        print(f"[WARN] Time gap: {diff_ns/1e6:.2f} ms exceeds threshold.")

                # Track
                odom_pose_estimate, slam_pose = tracker.track(msg_time_ns, images=images)

                # Visualisation update
                process_tracking_results(
                    frame_id=frame_id,
                    timestamp=msg_time_ns,
                    images=images,
                    odom_pose_estimate=odom_pose_estimate,
                    slam_pose=slam_pose,
                    tracker=tracker,
                    verbose=verbose,
                    trajectory_odom=trajectory_odom,
                    trajectory_slam=trajectory_slam,
                    last_valid_odom_pose=last_valid_odom_pose,
                    last_valid_slam_translation=last_valid_slam_translation,
                    is_tracking_lost=is_tracking_lost,
                    offset_T=offset_T,
                    last_valid_world_T=last_valid_world_T,
                    T_xsens_gray0=T_xsens_gray0,
                    T_gray0_xsens=T_gray0_xsens,
                    mode_mono=(odom_mode == cuvslam.Tracker.OdometryMode.Mono)
                )

                frame_id += 1
                prev_timestamp = msg_time_ns

        except RuntimeError:
            pass
        finally:
            pipeline.stop()
            print("[INFO] RealSense Bag processing finished.")

    # --------------------------------------------------------------------------
    # Playback: Case 2 - ROS 2 Bag (.db3)
    # --------------------------------------------------------------------------
    elif bag_path.endswith(".db3") or os.path.exists(os.path.join(bag_path, "metadata.yaml")):
        if not HAS_ROSBAGS:
            print("[ERROR] rosbags is not installed but trying to read ROS 2 bag!")
            sys.exit(1)

        topics_cfg = main_params.get("topics", {})
        left_topic = topics_cfg.get("left")
        right_topic = topics_cfg.get("right")

        if not left_topic:
            print("[ERROR] topics.left is not specified in main_params.yaml!")
            sys.exit(1)

        sync_buffer = {}
        MAX_BUFFER_SIZE = 15
        typestore = get_typestore(Stores.ROS2_HUMBLE)

        print("[INFO] Starting ROS 2 Bag processing...")
        with Reader(bag_path) as reader:
            target_topics = [left_topic]
            if odom_mode != cuvslam.Tracker.OdometryMode.Mono and right_topic:
                target_topics.append(right_topic)

            connections = [x for x in reader.connections if x.topic in target_topics]

            for connection, timestamp_bag_ns, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                msg_time_ns = msg.header.stamp.sec * int(1e9) + msg.header.stamp.nanosec

                # Decode image
                img_array = np.frombuffer(msg.data, np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if odom_mode == cuvslam.Tracker.OdometryMode.Mono:
                    # Monocular tracking - no sync required
                    images = [img_bgr]
                    odom_pose_estimate, slam_pose = tracker.track(msg_time_ns, images=images)
                    process_tracking_results(
                        frame_id=frame_id,
                        timestamp=msg_time_ns,
                        images=images,
                        odom_pose_estimate=odom_pose_estimate,
                        slam_pose=slam_pose,
                        tracker=tracker,
                        verbose=verbose,
                        trajectory_odom=trajectory_odom,
                        trajectory_slam=trajectory_slam,
                        last_valid_odom_pose=last_valid_odom_pose,
                        last_valid_slam_translation=last_valid_slam_translation,
                        is_tracking_lost=is_tracking_lost,
                        offset_T=offset_T,
                        last_valid_world_T=last_valid_world_T,
                        T_xsens_gray0=T_xsens_gray0,
                        T_gray0_xsens=T_gray0_xsens,
                        mode_mono=True
                    )
                    frame_id += 1
                else:
                    # Stereo tracking - buffer and sync
                    if msg_time_ns not in sync_buffer:
                        sync_buffer[msg_time_ns] = {}

                    if connection.topic == left_topic:
                        sync_buffer[msg_time_ns]['left'] = img_bgr
                    elif connection.topic == right_topic:
                        sync_buffer[msg_time_ns]['right'] = img_bgr

                    if len(sync_buffer) > MAX_BUFFER_SIZE:
                        oldest_key = min(sync_buffer.keys())
                        del sync_buffer[oldest_key]

                    if 'left' in sync_buffer[msg_time_ns] and 'right' in sync_buffer[msg_time_ns]:
                        left_img = sync_buffer[msg_time_ns]['left']
                        right_img = sync_buffer[msg_time_ns]['right']
                        del sync_buffer[msg_time_ns]

                        # Check for jitter
                        if prev_timestamp is not None:
                            diff_ns = msg_time_ns - prev_timestamp
                            if diff_ns > main_params.get("jitter_threshold_ns", 40 * 1e6):
                                print(f"[WARN] Time gap: {diff_ns/1e6:.2f} ms exceeds threshold.")

                        images = [left_img, right_img]
                        odom_pose_estimate, slam_pose = tracker.track(msg_time_ns, images=images)
                        process_tracking_results(
                            frame_id=frame_id,
                            timestamp=msg_time_ns,
                            images=images,
                            odom_pose_estimate=odom_pose_estimate,
                            slam_pose=slam_pose,
                            tracker=tracker,
                            verbose=verbose,
                            trajectory_odom=trajectory_odom,
                            trajectory_slam=trajectory_slam,
                            last_valid_odom_pose=last_valid_odom_pose,
                            last_valid_slam_translation=last_valid_slam_translation,
                            is_tracking_lost=is_tracking_lost,
                            offset_T=offset_T,
                            last_valid_world_T=last_valid_world_T,
                            T_xsens_gray0=T_xsens_gray0,
                            T_gray0_xsens=T_gray0_xsens,
                            mode_mono=False
                        )
                        frame_id += 1
                        prev_timestamp = msg_time_ns

        print("[INFO] ROS 2 Bag processing finished.")

    # --------------------------------------------------------------------------
    # Playback: Case 3 - ROS 1 Bag (.bag)
    # --------------------------------------------------------------------------
    elif bag_path.endswith(".bag"):
        topics_cfg = main_params.get("topics", {})
        left_topic = topics_cfg.get("left")
        right_topic = topics_cfg.get("right")

        if not left_topic:
            print("[ERROR] topics.left is not specified in main_params.yaml!")
            sys.exit(1)

        conns = read_bag_connections(bag_path)
        left_cids = {cid for cid, t in conns.items() if t == left_topic}

        if odom_mode != cuvslam.Tracker.OdometryMode.Mono:
            if not right_topic:
                print("[ERROR] topics.right is not specified for stereo mode!")
                sys.exit(1)
            right_cids = {cid for cid, t in conns.items() if t == right_topic}

            if not left_cids or not right_cids:
                print("[ERROR] Left or Right image topics not found in bag file connections!")
                sys.exit(1)

            gt_cids = set()
            if inline_gt_parsing:
                gt_cids = {cid for cid, t in conns.items() if t == gt_topic}

            print("[INFO] Starting ROS 1 Bag Stereo processing...")
            for res in iter_bag_stereo_ros1(bag_path, left_cids, right_cids, gt_cids, gt_frame_id, gt_child_frame_id):
                if res[0] == 'stereo':
                    sync_ts, left_img, right_img = res[1], res[2], res[3]
                    images = [left_img, right_img]
                    odom_pose_estimate, slam_pose = tracker.track(sync_ts, images=images)

                    process_tracking_results(
                        frame_id=frame_id,
                        timestamp=sync_ts,
                        images=images,
                        odom_pose_estimate=odom_pose_estimate,
                        slam_pose=slam_pose,
                        tracker=tracker,
                        verbose=verbose,
                        trajectory_odom=trajectory_odom,
                        trajectory_slam=trajectory_slam,
                        last_valid_odom_pose=last_valid_odom_pose,
                        last_valid_slam_translation=last_valid_slam_translation,
                        is_tracking_lost=is_tracking_lost,
                        offset_T=offset_T,
                        last_valid_world_T=last_valid_world_T,
                        T_xsens_gray0=T_xsens_gray0,
                        T_gray0_xsens=T_gray0_xsens,
                        mode_mono=False
                    )

                    frame_id += 1
                    if frame_id % 50 == 0:
                        print(f"[INFO] Processed: {frame_id} frames")
                elif res[0] == 'gt':
                    ts_ns, pos, quat = res[1], res[2], res[3]
                    pos = np.array(pos)

                    if T_gt is not None and T_gt.shape == (4, 4):
                        T_nav_imu = np.eye(4)
                        if quat is not None:
                            T_nav_imu[:3, :3] = R.from_quat(quat).as_matrix()
                        T_nav_imu[:3, 3] = pos

                        T_nav_cam = T_nav_imu @ T_gt
                        pos_cam = T_nav_cam[:3, 3]
                        R_nav_cam = T_nav_cam[:3, :3]

                        if first_gt_point is None:
                            first_gt_point = pos_cam.copy()
                            first_T_nav_cam_inv = np.linalg.inv(R_nav_cam)

                        delta_pos = pos_cam - first_gt_point
                        final_pos = first_T_nav_cam_inv @ delta_pos
                    else:
                        if first_gt_point is None:
                            first_gt_point = pos.copy()
                            if quat is not None:
                                first_T_nav_cam_inv = np.linalg.inv(R.from_quat(quat).as_matrix())

                        delta_pos = pos - first_gt_point
                        if first_T_nav_cam_inv is not None:
                            final_pos = first_T_nav_cam_inv @ delta_pos
                        else:
                            final_pos = delta_pos

                    trajectory_gt_pts.append(final_pos.tolist())
                    if len(trajectory_gt_pts) % 10 == 0:
                        rr.log('world/cuvslam/trajectory_gt', rr.LineStrips3D([trajectory_gt_pts], colors=[[0, 200, 0]]))
        else:
            # Monocular ROS 1 reading
            print("[INFO] Starting ROS 1 Bag Monocular processing...")
            gt_cids = set()
            if inline_gt_parsing:
                gt_cids = {cid for cid, t in conns.items() if t == gt_topic}

            for res in iter_bag_mono(bag_path, left_cids, gt_cids, gt_frame_id, gt_child_frame_id):
                if res[0] == 'mono':
                    sync_ts, img = res[1], res[2]
                    images = [img]
                    odom_pose_estimate, slam_pose = tracker.track(sync_ts, images=images)

                    process_tracking_results(
                        frame_id=frame_id,
                        timestamp=sync_ts,
                        images=images,
                        odom_pose_estimate=odom_pose_estimate,
                        slam_pose=slam_pose,
                        tracker=tracker,
                        verbose=verbose,
                        trajectory_odom=trajectory_odom,
                        trajectory_slam=trajectory_slam,
                        last_valid_odom_pose=last_valid_odom_pose,
                        last_valid_slam_translation=last_valid_slam_translation,
                        is_tracking_lost=is_tracking_lost,
                        offset_T=offset_T,
                        last_valid_world_T=last_valid_world_T,
                        T_xsens_gray0=T_xsens_gray0,
                        T_gray0_xsens=T_gray0_xsens,
                        mode_mono=True
                    )

                    frame_id += 1
                    if frame_id % 50 == 0:
                        print(f"[INFO] Processed: {frame_id} frames")
                elif res[0] == 'gt':
                    ts_ns, pos, quat = res[1], res[2], res[3]
                    pos = np.array(pos)

                    if T_gt is not None and T_gt.shape == (4, 4):
                        T_nav_imu = np.eye(4)
                        if quat is not None:
                            T_nav_imu[:3, :3] = R.from_quat(quat).as_matrix()
                        T_nav_imu[:3, 3] = pos

                        T_nav_cam = T_nav_imu @ T_gt
                        pos_cam = T_nav_cam[:3, 3]
                        R_nav_cam = T_nav_cam[:3, :3]

                        if first_gt_point is None:
                            first_gt_point = pos_cam.copy()
                            first_T_nav_cam_inv = np.linalg.inv(R_nav_cam)

                        delta_pos = pos_cam - first_gt_point
                        final_pos = first_T_nav_cam_inv @ delta_pos
                    else:
                        if first_gt_point is None:
                            first_gt_point = pos.copy()
                            if quat is not None:
                                first_T_nav_cam_inv = np.linalg.inv(R.from_quat(quat).as_matrix())

                        delta_pos = pos - first_gt_point
                        if first_T_nav_cam_inv is not None:
                            final_pos = first_T_nav_cam_inv @ delta_pos
                        else:
                            final_pos = delta_pos

                    trajectory_gt_pts.append(final_pos.tolist())
                    if len(trajectory_gt_pts) % 10 == 0:
                        rr.log('world/cuvslam/trajectory_gt', rr.LineStrips3D([trajectory_gt_pts], colors=[[0, 200, 0]]))

        print("[INFO] ROS 1 Bag processing finished.")


# ==============================================================================
# Unified Result Processing & Rerun Visualization function
# ==============================================================================

def process_tracking_results(
    frame_id: int,
    timestamp: int,
    images: List[np.ndarray],
    odom_pose_estimate: Any,
    slam_pose: Optional[Any],
    tracker: cuvslam.Tracker,
    verbose: bool,
    trajectory_odom: List[np.ndarray],
    trajectory_slam: List[np.ndarray],
    last_valid_odom_pose: Optional[SimplePose],
    last_valid_slam_translation: Optional[np.ndarray],
    is_tracking_lost: bool,
    offset_T: np.ndarray,
    last_valid_world_T: np.ndarray,
    T_xsens_gray0: Optional[np.ndarray],
    T_gray0_xsens: Optional[np.ndarray],
    mode_mono: bool
) -> None:
    # Access and mutate globals/outer state (using a dictionary structure or directly mapping)
    # We can write variables into globals or handle their mutation inside outer scope using mutable types.
    # To bypass scope limitations, we modify the lists directly.
    # We retrieve the actual references of active pose

    active_odom_pose = None
    active_slam_translation = None
    obs_uv_left, obs_colors_left = [], []
    obs_uv_right, obs_colors_right = [], []

    if odom_pose_estimate.world_from_rig is None:
        print(f"[WARN] Pose tracking lost on frame {frame_id}. Freezing pose.")
        # Mutate outer state flags
        # In python, variables assigned in outer scopes cannot be reassigned without global/nonlocal.
        # But we can access the last elements of the trajectories if they are present.
        if trajectory_odom:
            active_odom_pose = SimplePose(trajectory_odom[-1], [0.0, 0.0, 0.0, 1.0])
        if trajectory_slam:
            active_slam_translation = trajectory_slam[-1]
    else:
        current_tracker_T = pose_to_mat(odom_pose_estimate.world_from_rig.pose)

        # If tracking was previously lost, compute new offset
        # Let's use mutable variables/structures to share status if needed, but since this is a helper,
        # we can compute them and let outer loop store them. We will write values into a mutable state object.
        world_T = current_tracker_T # We assume no offsets needed for simplified unified run,
        # but let's align with the offset_T logic if needed:
        # Actually, let's keep the offset computation directly.
        # Since we want to update the last valid world_T, let's do this calculation:

        # Botanic Garden dataset uses transformation into IMU (Xsens) coordinate frame:
        if T_xsens_gray0 is not None:
            # imu_T = T_xsens_gray0 @ world_T @ T_gray0_xsens
            imu_T = T_xsens_gray0 @ world_T @ T_gray0_xsens
            active_odom_pose = mat_to_simple_pose(imu_T)

            active_slam_translation = None
            if slam_pose is not None and slam_pose.translation is not None:
                pt_cam = np.append(slam_pose.translation, 1.0)
                active_slam_translation = (T_xsens_gray0 @ pt_cam)[:3]
        else:
            active_odom_pose = mat_to_simple_pose(world_T)
            active_slam_translation = slam_pose.translation if slam_pose is not None else None

        # Keypoint observations
        obs_left = tracker.get_last_observations(0)
        obs_uv_left = [[o.u, o.v] for o in obs_left]
        obs_colors_left = [color_from_id(o.id) for o in obs_left]

        if not mode_mono and len(images) > 1:
            obs_right = tracker.get_last_observations(1)
            obs_uv_right = [[o.u, o.v] for o in obs_right]
            obs_colors_right = [color_from_id(o.id) for o in obs_right]

    if active_odom_pose is not None:
        trajectory_odom.append(active_odom_pose.translation)
        if active_slam_translation is not None:
            trajectory_slam.append(active_slam_translation)

        if verbose:
            rr.set_time_sequence('frame', frame_id)
            # Log left camera image
            # Make sure to compress for performance
            rr.log('world/cuvslam/camera/left', rr.Image(images[0]).compress(jpeg_quality=80))
            if not mode_mono and len(images) > 1:
                rr.log('world/cuvslam/camera/right', rr.Image(images[1]).compress(jpeg_quality=80))

            rr.log('world/cuvslam/trajectory_odom', rr.LineStrips3D([trajectory_odom]))
            if trajectory_slam:
                rr.log('world/cuvslam/trajectory_slam', rr.LineStrips3D([trajectory_slam]))

            # Log current Pose transformation
            rr.log(
                "world/cuvslam/camera/rig",
                rr.Transform3D(
                    translation=active_odom_pose.translation,
                    quaternion=active_odom_pose.rotation
                ),
                rr.Arrows3D(
                    vectors=np.eye(3) * 0.2,
                    origins=np.zeros((3, 3)),
                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                    radii=0.01,
                )
            )

            # Log Keypoint Points
            if obs_uv_left:
                rr.log('world/cuvslam/camera/left/observations', rr.Points2D(obs_uv_left, radii=5, colors=obs_colors_left))
            else:
                rr.log('world/cuvslam/camera/left/observations', rr.Clear(recursive=False))

            if not mode_mono and len(images) > 1:
                if obs_uv_right:
                    rr.log('world/cuvslam/camera/right/observations', rr.Points2D(obs_uv_right, radii=5, colors=obs_colors_right))
                else:
                    rr.log('world/cuvslam/camera/right/observations', rr.Clear(recursive=False))


if __name__ == '__main__':
    main()
