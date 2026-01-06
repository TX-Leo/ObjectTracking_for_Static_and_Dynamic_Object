# -*- coding: utf-8 -*-
# @FileName: ObjectTracking.py
import argparse
from pathlib import Path
import os

from object_tracking.ObjectTrackingKeypointsSelector import ObjectTrackingKeypointsSelector
from object_tracking.ObjectTrackingCoTracker import ObjectTrackingCoTracker
from object_tracking.ObjectTrackingTriangulator import ObjectTrackingTriangulator
from object_tracking.ObjectTrackingDIFT import ObjectTrackingDIFT

class ObjectTracking:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        self.ot_kpts_selector = ObjectTrackingKeypointsSelector(self.mps_path)
        self.ot_cotracker = ObjectTrackingCoTracker(self.mps_path)
        self.ot_triangulator = ObjectTrackingTriangulator(self.mps_path)
        self.ot_dift = ObjectTrackingDIFT(self.mps_path)
   
    def run(self):
        DIFT_reference = os.path.join(str(Path(self.mps_path).parent), 'DIFT_reference')
        src_img_path = os.path.join(DIFT_reference, 'rgb.png')
        src_json_path = os.path.join(DIFT_reference, 'ot_keypoints_selector.json')
        if os.path.exists(src_img_path) and os.path.exists(src_json_path):
            self.ot_dift.run()
        else:
            print('You need to run ObjectTrackingKeypointsSelector.py to get the keypoints manually as a reference!')
            self.ot_kpts_selector.run()

        self.ot_cotracker.run()
        self.ot_triangulator.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()

    ot = ObjectTracking(args.mps_path)
    ot.run()

# conda activate aria
# cd src
# python -m object_tracking.ObjectTrackingKeypointsSelector --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_5_vrs/" 