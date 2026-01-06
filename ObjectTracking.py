# -*- coding: utf-8 -*-
# @FileName: ObjectTracking.py
import argparse
from pathlib import Path
import os

class ObjectTracking:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path

    def object_tracking_run_module(self):
        def _call(mod):
            import subprocess, sys
            subprocess.run([sys.executable, "-m", mod, "--mps_path", self.mps_path], check=True)

        DIFT_reference = os.path.join(str(Path(self.mps_path).parent), 'DIFT_reference')
        src_img_path = os.path.join(DIFT_reference, 'rgb.png')
        src_json_path = os.path.join(DIFT_reference, 'ot_keypoints_selector.json')
        if os.path.exists(src_img_path) and os.path.exists(src_json_path):
            _call("object_tracking.ObjectTrackingDIFT")
        else:
            print('!!! You need to run ObjectTrackingKeypointsSelector.py to get the keypoints manually as a reference!')
            _call("object_tracking.ObjectTrackingKeypointsSelector")
        _call("object_tracking.ObjectTrackingCoTracker")
        _call("object_tracking.ObjectTrackingTriangulator")

    def object_tracking(self):
        from object_tracking.ObjectTrackingKeypointsSelector import ObjectTrackingKeypointsSelector
        from object_tracking.ObjectTrackingDIFT import ObjectTrackingDIFT
        from object_tracking.ObjectTrackingCoTracker import ObjectTrackingCoTracker
        from object_tracking.ObjectTrackingTriangulator import ObjectTrackingTriangulator
        DIFT_reference = os.path.join(str(Path(self.mps_path).parent), 'DIFT_reference')
        src_img_path = os.path.join(DIFT_reference, 'rgb.png')
        src_json_path = os.path.join(DIFT_reference, 'ot_keypoints_selector.json')
        if os.path.exists(src_img_path) and os.path.exists(src_json_path):
            ot_dift = ObjectTrackingDIFT(self.mps_path)
            ot_dift.run()
        else:
            print('!!! You need to run ObjectTrackingKeypointsSelector.py to get the keypoints manually as a reference!')
            ot_kpts_selector = ObjectTrackingKeypointsSelector(self.mps_path)
            ot_kpts_selector.run()

        ot_cotracker = ObjectTrackingCoTracker(self.mps_path)
        ot_cotracker.run()
        
        ot_triangulator = ObjectTrackingTriangulator(self.mps_path)
        ot_triangulator.run()

    def run(self, run_module: bool = True):
        if run_module:
            self.object_tracking_run_module()
        else:
            self.object_tracking()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    parser.add_argument("--run_module", action='store_true', default=True, help="一个一个模块的运行，以免出现CUDA out of memory")
    args = parser.parse_args()

    ot = ObjectTracking(args.mps_path)
    ot.run(args.run_module)

# conda activate aria
# cd src
# python -m object_tracking.ObjectTracking --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_005_vrs/" 