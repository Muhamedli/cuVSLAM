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
RealSense cuVSLAM legacy wrapper script.
Loads configuration files from the config directory, overrides options to run
with the RealSense bag playback configuration, and invokes the unified runner.
"""

import os
import sys

# Add script directory to sys.path to import run_cuvslam
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import run_cuvslam

def main():
    print("[INFO] RealSense legacy wrapper. Redirecting to unified cuVSLAM runner...")

    # Overrides for RealSense bag execution
    params_override = {
        "bag_path": "/mnt/foundation_ssd/foundation_data/recorded_data/bags/VSLAM/realsense_d405/NLK_4.bag",
        "use_realsense_bag": True,
        "mode": "stereo",
        "use_slam": True
    }

    # Execute unified runner
    run_cuvslam.main(params_override=params_override)

if __name__ == '__main__':
    main()
