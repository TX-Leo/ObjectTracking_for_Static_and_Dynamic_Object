# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import numpy as np
import open3d as o3d
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

# ==================== 路径修正 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from aria.AriaDataset import AriaDataset

# ==============================================================================
# [Module 1] 几何解算
# ==============================================================================
class MathUtils:
    @staticmethod
    def triangulate_n_views(poses, k_matrices, tracks_raw, visibilities):
        """ 必须在 Landscape (Raw) 坐标系下进行解算 """
        A = []
        for i in range(len(poses)):
            if visibilities[i] == 0: continue
            u, v = tracks_raw[i] # 这是 Raw 坐标
            
            T_w2c = np.linalg.inv(poses[i])
            P = k_matrices[i] @ T_w2c[:3, :]
            
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
            
        if len(A) < 4: return np.zeros(3)
        A = np.array(A)
        _, _, vh = np.linalg.svd(A)
        return vh[-1][:3] / vh[-1][3]

# ==============================================================================
# [Module 2] 核心解算器
# ==============================================================================
class AriaStaticSolver:
    def __init__(self, dataset, config_path, result_json_path):
        self.dataset = dataset
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(result_json_path, 'r') as f:
            tracker_data = json.load(f)
        
        self.ref_idx = self.config["reference_frame_idx"]
        self.split_idx = self.config["split_frame_idx"]
        
        self.tracks_vis = np.array(tracker_data["tracks_2d"])
        self.visibilities = np.array(tracker_data["visibilities"])
        self.num_pts = self.tracks_vis.shape[1]

    def solve_3d(self):
        print(f"[Solver] 正在转换坐标并解算 3D...")
        pts_3d_world = []
        h_vis, w_vis = self.dataset[0].cam.rgb.shape[:2]

        for n in range(self.num_pts):
            poses, ks, obs_raw, vis = [], [], [], []
            for i in range(min(self.ref_idx, self.split_idx), max(self.ref_idx, self.split_idx) + 1):
                s = self.dataset[i]
                
                u, v = self.tracks_vis[i, n]
                
                poses.append(s.cam.c2w)
                ks.append(s.cam.k)
                obs_raw.append([u, v])
                vis.append(self.visibilities[i, n])
            
            p3d = MathUtils.triangulate_n_views(poses, ks, obs_raw, vis)
            pts_3d_world.append(p3d)
            
        return np.array(pts_3d_world)

# ==============================================================================
# [Module 3] 渲染器
# ==============================================================================
class StaticVisualizer:
    def __init__(self, dataset, save_dir):
        self.dataset = dataset
        self.save_dir = save_dir
        self.cmap = plt.get_cmap('hsv')

    def render(self, tracks_vis, visibilities, pts_3d, split_idx):
        h_vis, w_vis = self.dataset[0].cam.rgb.shape[:2]
        video_path = os.path.join(self.save_dir, "ot_triangulation_video_vis.mp4")
        gif_path = os.path.join(self.save_dir, "ot_triangulation_video_vis.gif")
        
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.dataset.fps, (w_vis, h_vis))
        gif_frames = []
        
        for t in tqdm(range(split_idx + 1), desc="Rendering"):
            img = self.dataset[t].cam.rgb.copy()
            T_w2c = np.linalg.inv(self.dataset[t].cam.c2w)

            for n in range(len(pts_3d)):
                color = [int(c*255) for c in self.cmap(n/len(pts_3d))[:3][::-1]]
                
                # 1. 直接绘制追踪点 (它本来就是在 Vis 坐标系下的)
                curr_vis = (int(tracks_vis[t, n, 0]), int(tracks_vis[t, n, 1]))
                # if visibilities[t, n]:
                #     cv2.circle(img, curr_vis, 3, color, -1)

                # 2. 绘制 3D 投影点 (Raw -> Vis 转换)
                pc = T_w2c[:3,:3] @ pts_3d[n] + T_w2c[:3,3]
                if pc[2] > 0:
                    uv_h = self.dataset[t].cam.k @ pc
                    u, v = uv_h[0]/uv_h[2], uv_h[1]/uv_h[2]
                    
                    cv_proj = (int(u), int(v))
                    cv2.circle(img, cv_proj, 8, (255, 255, 255), -1)
                    cv2.circle(img, cv_proj, 5, color, -1)

            out.write(img)
            gif_frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        out.release()
        imageio.mimsave(gif_path, gif_frames, fps=self.dataset.fps)

    def export_check(self, pts_3d, mps_path):
        pc_path = os.path.join(mps_path, "aria", "aria_pc_scene_and_head_traj.ply")
        combined = o3d.geometry.PointCloud()
        if os.path.exists(pc_path):
            pcd = o3d.io.read_point_cloud(pc_path)
            pcd.paint_uniform_color([0.2, 0.2, 0.2])
            combined += pcd
        for i, pt in enumerate(pts_3d):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(pt)
            sphere_pcd = sphere.sample_points_uniformly(500)
            sphere_pcd.paint_uniform_color(self.cmap(i/len(pts_3d))[:3])
            combined += sphere_pcd
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "ot_triangulation_pc.ply"), combined)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    config_path = os.path.join(save_dir, "handle_selection.json")
    result_json = os.path.join(save_dir, "ot_cotracker_results.json")
    
    ds = AriaDataset(args.mps_path, 
                     os.path.join(args.mps_path, [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')][0]),
                     os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv"), 
                     save_dir)
    
    solver = AriaStaticSolver(ds, config_path, result_json)
    pts_3d_world = solver.solve_3d()
    
    # 保存结果
    with open(os.path.join(save_dir, "ot_triangulation_results.json"), 'w') as f:
        json.dump({"static_handle_points_world": pts_3d_world.tolist()}, f, indent=4)
    
    vis = StaticVisualizer(ds, save_dir)
    vis.render(solver.tracks_vis, solver.visibilities, pts_3d_world, solver.split_idx)
    vis.export_check(pts_3d_world, args.mps_path)
    print(f"完成！请检查 {save_dir} 下的视频和点云。")

if __name__ == "__main__":
    main()

