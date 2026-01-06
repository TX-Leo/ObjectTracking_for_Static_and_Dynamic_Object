# -*- coding: utf-8 -*-
# @FileName: ObjectTrackingTriangulator.py

import os
import cv2
import json
import numpy as np
import open3d as o3d
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

class GeometryUtils:
    @staticmethod
    def triangulate_n_views(poses, k_matrices, obs_uv, visibilities):
        """ 多视图三角测量解算 3D 点 """
        A = []
        for i in range(len(poses)):
            if visibilities[i] == 0: continue
            u, v = obs_uv[i]
            T_w2c = np.linalg.inv(poses[i])
            P = k_matrices[i] @ T_w2c[:3, :]
            
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
            
        if len(A) < 4: return np.zeros(3)
        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        return vh[-1][:3] / vh[-1][3]

    @staticmethod
    def calculate_reprojection_error(p3d, poses, k_matrices, obs_uv, visibilities):
        """ 计算 3D 点在所有视图下的重投影误差 (RMSE) """
        errors = []
        for i in range(len(poses)):
            if visibilities[i] == 0: continue
            T_w2c = np.linalg.inv(poses[i])
            p_cam = T_w2c[:3, :3] @ p3d + T_w2c[:3, 3]
            if p_cam[2] <= 0: continue
            
            uv_h = k_matrices[i] @ p_cam
            u_proj, v_proj = uv_h[0]/uv_h[2], uv_h[1]/uv_h[2]
            
            err = np.sqrt((u_proj - obs_uv[i][0])**2 + (v_proj - obs_uv[i][1])**2)
            errors.append(err)
        return np.mean(errors) if errors else 1e6


class ObjectTrackingTriangulator:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        self.aria_dir = os.path.join(mps_path, "aria")
        self._load_data()

    def _load_data(self):
        # 1. 加载 Phase 和 Selection 配置
        sel_path = os.path.join(self.aria_dir, "ot_keypoints_selector.json")
        res_path = os.path.join(self.aria_dir, "ot_cotracker_results.json")
        cam_cfg_path = os.path.join(self.aria_dir, "aria_cam_config.json")
        
        with open(sel_path, 'r') as f: self.sel_cfg = json.load(f)
        with open(res_path, 'r') as f: self.tracker_res = json.load(f)
        with open(cam_cfg_path if os.path.exists(cam_cfg_path) else "", 'r') as f:
            self.fps = json.load(f).get("fps", 10) if os.path.exists(cam_cfg_path) else 10

        self.ref_idx = self.sel_cfg["reference_frame_idx"]
        self.split_idx = self.sel_cfg["split_frame_idx"]
        self.tracks = np.array(self.tracker_res["tracks"])
        self.visibilities = np.array(self.tracker_res["visibilities"])
        self.num_pts = self.tracks.shape[1]

    def _load_cam_data(self, idx: int):
        """ 直接从 all_data 文件夹读取位姿和内参 """
        json_path = os.path.join(self.aria_dir, "all_data", f"{idx:05d}", "aria_cam.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return np.array(data["c2w"]), np.array(data["k"])

    def solve_3d(self):
        print(f"║ [Solver] Triangulating {self.num_pts} points from frame {self.ref_idx} to {self.split_idx}...")
        pts_3d_world = []
        errors = []

        # 预加载解算区间内的相机数据以提高速度
        cam_cache = {}
        for i in range(min(self.ref_idx, self.split_idx), max(self.ref_idx, self.split_idx) + 1):
            cam_cache[i] = self._load_cam_data(i)

        for n in range(self.num_pts):
            poses, ks, obs, vis = [], [], [], []
            for i, (c2w, k) in cam_cache.items():
                poses.append(c2w)
                ks.append(k)
                obs.append(self.tracks[i, n])
                vis.append(self.visibilities[i, n])
            
            p3d = GeometryUtils.triangulate_n_views(poses, ks, obs, vis)
            err = GeometryUtils.calculate_reprojection_error(p3d, poses, ks, obs, vis)
            
            pts_3d_world.append(p3d)
            errors.append(err)
            
        return np.array(pts_3d_world), errors

    def run(self):
        # 1. 执行解算
        pts_3d_world, errors = self.solve_3d()
    
        # 2. 保存 JSON 结果
        res_save = {
            "num_points": len(pts_3d_world),
            "reprojection_errors_px": errors,
            "points_3d_world": pts_3d_world.tolist(),
            "metadata": {
                "solver": "SVD-Linear-Triangulation",
                "reference_idx": self.ref_idx,
                "split_idx": self.split_idx
            }
        }
        out_json = os.path.join(self.mps_path, "aria", "ot_triangulation_results.json")
        with open(out_json, 'w') as f:
            json.dump(res_save, f, indent=4)
        
        # 3. 报告与可视化
        ObjectTrackingTriangulatorOps.print_report(pts_3d_world, errors)
        ObjectTrackingTriangulatorOps.render_and_save(self.mps_path, pts_3d_world, self.tracks, self.visibilities, self.split_idx, self.fps)
        ObjectTrackingTriangulatorOps.export_ply(self.mps_path, pts_3d_world)

class ObjectTrackingTriangulatorOps:
    @staticmethod
    def render_and_save(mps_path, pts_3d, tracks, vis, split_idx, fps):
        aria_dir = os.path.join(mps_path, "aria")
        video_path = os.path.join(aria_dir, "ot_triangulation_video_vis.mp4")
        
        # 加载首帧获取尺寸
        sample_img = cv2.imread(os.path.join(aria_dir, "all_data", "00000", "rgb.png"))
        h_orig, w_orig = sample_img.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_orig, h_orig))
        
        cmap = plt.get_cmap('hsv')
        gif_frames = []

        for t in tqdm(range(split_idx + 1), desc="Rendering 3D Projections"):
            # 读取图片和位姿
            frame_path = os.path.join(aria_dir, "all_data", f"{t:05d}", "rgb.png")
            img = cv2.imread(frame_path)
            
            cam_json = os.path.join(aria_dir, "all_data", f"{t:05d}", "aria_cam.json")
            with open(cam_json, 'r') as f:
                cam_data = json.load(f)
            T_w2c = np.linalg.inv(np.array(cam_data["c2w"]))
            K = np.array(cam_data["k"])

            # 绘制 HUD
            ObjectTrackingTriangulatorOps._draw_hud(img, t, len(pts_3d))

            for n in range(len(pts_3d)):
                color = [int(c*255) for c in cmap(n/len(pts_3d))[:3][::-1]]
                
                # 绘制 3D 投影点
                pc = T_w2c[:3,:3] @ pts_3d[n] + T_w2c[:3,3]
                if pc[2] > 0:
                    uv_h = K @ pc
                    u, v = int(uv_h[0]/uv_h[2]), int(uv_h[1]/uv_h[2])
                    # 双圆环渲染
                    cv2.circle(img, (u, v), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, (u, v), 5, color, -1, cv2.LINE_AA)
                    cv2.putText(img, f"P{n}", (u+10, v-10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

            out.write(img)
            gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        out.release()
        imageio.mimsave(video_path.replace(".mp4", ".gif"), gif_frames, fps=fps, loop=0)

    @staticmethod
    def _draw_hud(img, t, n_pts):
        h, w = img.shape[:2]
        cv2.rectangle(img, (20, 20), (280, 80), (30, 30, 30), -1)
        cv2.rectangle(img, (20, 20), (280, 80), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "3D TRIANGULATION SOLVER", (35, 45), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"Target: {n_pts} Keypoints | Frame: {t:05d}", (35, 65), cv2.FONT_HERSHEY_DUPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    @staticmethod
    def export_ply(mps_path, pts_3d):
        save_path = os.path.join(mps_path, "aria", "ot_triangulation_pc.ply")
        scene_pc_path = os.path.join(mps_path, "aria", "aria_pc_scene_with_human_traj.ply")
        
        combined = o3d.geometry.PointCloud()
        if os.path.exists(scene_pc_path):
            scene = o3d.io.read_point_cloud(scene_pc_path)
            scene.paint_uniform_color([0.15, 0.15, 0.15]) # 灰化背景
            combined += scene
            
        cmap = plt.get_cmap('hsv')
        for i, pt in enumerate(pts_3d):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(pt)
            sphere.paint_uniform_color(cmap(i/len(pts_3d))[:3])
            combined += sphere.sample_points_uniformly(1000)
            
        o3d.io.write_point_cloud(save_path, combined)
        print(f"║ [Export] 3D Scene PLY saved to: {save_path}")

    @staticmethod
    def print_report(pts_3d, errors):
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'TRIANGULATION QUALITY REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║ [SOLVED COORDINATES (WORLD)] {'':<29}║")
        for i, (pt, err) in enumerate(zip(pts_3d, errors)):
            p_str = f"P{i}: [{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}]"
            e_str = f"RMSE: {err:.2f}px"
            print(f"║  - {p_str:<35} | {e_str:<12} ║")
        
        avg_err = np.mean(errors)
        quality = "EXCELLENT" if avg_err < 3 else "GOOD" if avg_err < 8 else "RE-TRACK REQUIRED"
        print(f"║ {'':<58} ║")
        print(f"║  - Average Reproj Error: {avg_err:>8.2f} px {'':<22}║")
        print(f"║  - Overall Geometry Qual: {quality:<33}║")
        print("╚" + "═" * 60 + "╝\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()

    ot_triangulator = ObjectTrackingTriangulator(args.mps_path)
    ot_triangulator.run()

# conda activate aria
# cd src
# python -m object_tracking.ObjectTrackingTriangulator --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_5_vrs/" 