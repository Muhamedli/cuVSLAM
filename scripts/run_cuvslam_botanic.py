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
Botanic Garden cuVSLAM legacy wrapper script.
Loads configuration files from the config directory, overrides options in-memory
to run with the Botanic Garden dataset parameters, and invokes the unified runner.
"""

import os
import sys

# Add script directory to sys.path to import run_cuvslam
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import run_cuvslam

def main():
    print("[INFO] Botanic Garden legacy wrapper. Redirecting to unified cuVSLAM runner...")

    # Overrides for Botanic Garden dataset parameters
    params_override = {
        "bag_path": "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/Botanic_garden/1018_00_VLIO.bag",
        "gt_bag_path": "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/Botanic_garden/gt_1018_00_qnew.bag",
        "use_realsense_bag": False,
        "mode": "stereo",
        "use_slam": True,
        "topics": {
            "left": "/dalsa_gray/left/image_raw",
            "right": "/dalsa_gray/right/image_raw"
        }
    }

    # In-memory intrinsics for Botanic Garden
    intrinsics_override = {
        "cam_0": {
            "width": 960,
            "height": 600,
            "fx": 643.5951111267952,
            "fy": 642.665612685733,
            "cx": 475.2215406900663,
            "cy": 307.4120184196884,
            "distortion_model": "brown",
            "distortion_coefficients": [-0.061606731291039, 0.10012990070893, 0.0, 0.0, 0.0]
        },
        "cam_1": {
            "width": 960,
            "height": 600,
            "fx": 645.5637158104681,
            "fy": 644.6523115974766,
            "cx": 471.2755868675698,
            "cy": 304.269842487349,
            "distortion_model": "brown",
            "distortion_coefficients": [-0.060194715488128, 0.097872361377687, 0.0, 0.0, 0.0]
        }
    }

    # In-memory extrinsics for Botanic Garden
    extrinsics_override = {
        "transforms": {
            "rig_from_cam_0": {
                "translation": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0.0, 0.0, 1.0]
            },
            "rig_from_cam_1": {
                "translation": [0.254156200727329, 0.0007626246224125, -0.0011160568800132],
                "rotation": [0.0016017540191933125, -0.00017088846568026988, -0.0010174442335515455, 0.9999981849925659]
            },
            "T_xsens_gray0": [
                [-0.00375313, 0.01609339, 0.99986345, 0.17517298],
                [-0.99997267, -0.00642959, -0.00365005, 0.13259005],
                [0.00636997, -0.99984982, 0.01611708, 0.0670137],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    }

    # Execute unified runner
    run_cuvslam.main(
        params_override=params_override,
        intrinsics_override=intrinsics_override,
        extrinsics_override=extrinsics_override
    )

if __name__ == '__main__':
    main()
