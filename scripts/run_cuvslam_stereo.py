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
ROS 2 Stereo cuVSLAM legacy wrapper script.
Loads configuration files from the config directory, overrides options to run
with the ROS 2 stereo database configuration, and invokes the unified runner.
"""

import os
import sys

# Add script directory to sys.path to import run_cuvslam
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import run_cuvslam

def main():
    print("[INFO] ROS 2 Stereo legacy wrapper. Redirecting to unified cuVSLAM runner...")

    # Overrides for ROS 2 stereo bag execution
    params_override = {
        "bag_path": "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/VO/2025_01_22-14_06_00-hall_VO_V1_0.db3",
        "use_realsense_bag": False,
        "mode": "stereo",
        "use_slam": True,
        "topics": {
            "left": "/zedx_45327000/zed_node/left/image_rect_color/compressed",
            "right": "/zedx_45327000/zed_node/right/image_rect_color/compressed"
        }
    }

    # Execute unified runner
    run_cuvslam.main(params_override=params_override)

if __name__ == '__main__':
    main()
