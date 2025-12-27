# -*- coding: utf-8 -*-
# @FileName: static_solver.py
# @Description: 第二阶段：静态场景 3D 点云解算与追踪评估。

import os
import cv2
import sys
import json
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ==================== 路径修正 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from aria.AriaDataset import AriaDataset

try:
    from cotracker.predictor import CoTrackerPredictor
    HAS_COTRACKER = True
except ImportError:
    HAS_COTRACKER = False
    print("Warning: CoTracker not found. tracking will fail.")

COTRACKER_RES = 640
# ==============================================================================
# [Module 1] 几何数学类
# ==============================================================================

class MathUtils:
    @staticmethod
    def triangulate_n_views(poses, k_matrices, tracks_2d):
        """ 多视图三角化 """
        A = []
        for T_c2w, K, (u, v) in zip(poses, k_matrices, tracks_2d):
            T_w2c = np.linalg.inv(T_c2w)
            P = K @ T_w2c[:3, :]
            A.append(u * P[2, :] - P[0, :])
            A.append(v * P[2, :] - P[1, :])
        _, _, vh = np.linalg.svd(np.array(A))
        X = vh[-1]
        return X[:3] / X[3]

# ==============================================================================
# [Module 2] 静态解算类
# ==============================================================================

class AriaStaticSolver:
    def __init__(self, dataset: AriaDataset, config_path: str, device="cuda"):
        self.dataset = dataset
        self.device = device
        self.img_h, self.img_w = dataset[0].cam.rgb.shape[:2]
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.ref_idx = self.config["reference_frame_idx"]
        self.split_idx = self.config["split_frame_idx"]
        self.init_pts_raw = self.config["handle_points_2d_raw"]
        self.num_pts = len(self.init_pts_raw)
        self.target_res = COTRACKER_RES

    def run_tracking(self):
        if not HAS_COTRACKER:
            raise ImportError("CoTracker is required for this step.")
            
        print(f"\n[Task 1/5] 执行 CoTracker 静态追踪 (分辨率: {self.target_res}x{self.target_res})...")
        frames = []
        # 分别计算宽和高的缩放比例
        scale_w = self.target_res / self.img_w
        scale_h = self.target_res / self.img_h

        # 1. 准备视频流
        for i in tqdm(range(self.split_idx + 1), desc="Loading Video"):
            img = self.dataset[i].cam.rgb
            # 直接 resize，不旋转
            frames.append(cv2.resize(img, (self.target_res, self.target_res)))
        
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(self.device).float()
        
        # 2. 准备查询点 (u 对应 x_scale, v 对应 y_scale)
        queries = []
        for p in self.init_pts_raw:
            queries.append([float(self.ref_idx), p[0] * scale_w, p[1] * scale_h])
        
        # 3. 推理
        ckpt = os.path.join(os.path.dirname(__file__), "checkpoints/scaled_offline.pth")
        model = CoTrackerPredictor(checkpoint=ckpt).to(self.device)
        with torch.no_grad():
            pred_tracks, _ = model(video, queries=torch.tensor([queries]).to(self.device).float())
        
        # 4. 坐标还原
        tracks_out = pred_tracks[0].cpu().numpy()
        tracks_out[:, :, 0] /= scale_w
        tracks_out[:, :, 1] /= scale_h
        return tracks_out

    def solve_3d_with_metrics(self, tracks_raw):
        print(f"[Task 2/5] 正在执行多视图三角化 (共 {self.num_pts} 个点)...")
        pts_3d = []
        errors_per_frame = []

        for pt_idx in range(self.num_pts):
            poses, ks, obs_2d = [], [], []
            for i in range(self.split_idx + 1):
                s = self.dataset[i]
                poses.append(s.cam.c2w) 
                ks.append(s.cam.k)
                obs_2d.append(tracks_raw[i, pt_idx])
            p3d = MathUtils.triangulate_n_views(poses, ks, obs_2d)
            pts_3d.append(p3d)
            
        pts_3d = np.array(pts_3d)

        # 计算重投影误差
        total_rmse = 0
        for i in range(self.split_idx + 1):
            s = self.dataset[i]
            T_w2c = np.linalg.inv(s.cam.c2w)
            frame_errs = []
            for j in range(self.num_pts):
                p_c = T_w2c[:3,:3] @ pts_3d[j] + T_w2c[:3,3]
                uv_h = s.cam.k @ p_c
                u_re, v_re = uv_h[0]/uv_h[2], uv_h[1]/uv_h[2]
                dist = np.linalg.norm(np.array([u_re, v_re]) - tracks_raw[i, j])
                frame_errs.append(dist)
            errors_per_frame.append(np.mean(frame_errs))
            total_rmse += np.mean(frame_errs)
        
        avg_rmse = total_rmse / (self.split_idx + 1)
        dists = [np.linalg.norm(pts_3d[k+1] - pts_3d[k]) for k in range(self.num_pts - 1)]
        
        print(f"\n" + "="*40)
        print(f"解算精度报告: 平均误差 {avg_rmse:.3f} px | 状态: {'⭐优质' if avg_rmse < 3 else '⚠️待检查'}")
        print("="*40)
        
        return pts_3d, {"rmse": avg_rmse, "dists": dists}, errors_per_frame

# ==============================================================================
# [Module 3] 评估与渲染
# ==============================================================================

class StaticQualitativeEvaluator:
    def __init__(self, dataset, save_dir):
        self.dataset = dataset
        self.save_dir = save_dir

    def export_3d_alignment_check(self, pts_3d, mps_path):
        all_geoms = []
        pc_path = os.path.join(mps_path, "aria", "pc_scene_and_head_traj.ply")
        if os.path.exists(pc_path):
            pcd = o3d.io.read_point_cloud(pc_path)
            pcd.paint_uniform_color([0.3, 0.3, 0.3])
            all_geoms.append(pcd)

        cmap = plt.get_cmap('tab10')
        for i in range(len(pts_3d)):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere.translate(pts_3d[i])
            sphere.paint_uniform_color(cmap(i % 10)[:3])
            all_geoms.append(sphere.sample_points_uniformly(200))
            
        combined = o3d.geometry.PointCloud()
        for g in all_geoms: combined += g
        o3d.io.write_point_cloud(os.path.join(self.save_dir, "static_solver_pc.ply"), combined)

    def plot_error_analysis(self, errors):
        plt.figure(figsize=(10, 4))
        plt.plot(errors, color='purple', label='Reprojection Error')
        plt.axhline(y=np.mean(errors), color='r', linestyle='--')
        plt.title("Static Solver Reprojection Error (px)")
        plt.savefig(os.path.join(self.save_dir, "static_solver_eval.png"))
        plt.close()

class EnhancedVisualizer:
    def __init__(self, dataset, save_dir):
        self.dataset = dataset
        self.save_dir = save_dir
        self.cmap = plt.get_cmap('hsv')

    def render_static_rainbow(self, tracks_raw, points_3d, split_idx, fps=10):
        print(f"[Task 3/5] 正在生成可视化视频...")
        video_path = os.path.join(self.save_dir, "static_solver_vis.mp4")
        h_vis, w_vis = self.dataset[0].cam.rgb.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vis, h_vis))
        num_pts = tracks_raw.shape[1]

        for i in tqdm(range(split_idx + 1), desc="Rendering"):
            img = self.dataset[i].cam.rgb.copy()
            T_w2c = np.linalg.inv(self.dataset[i].cam.c2w)
            
            for j in range(num_pts):
                color = [int(c*255) for c in self.cmap(j/num_pts)[:3][::-1]]
                
                # 1. 轨迹 (直接使用 u, v)
                for t in range(max(1, i - 10), i + 1):
                    p1 = (int(tracks_raw[t-1, j, 0]), int(tracks_raw[t-1, j, 1]))
                    p2 = (int(tracks_raw[t, j, 0]), int(tracks_raw[t, j, 1]))
                    cv2.line(img, p1, p2, color, 2)
                
                # 2. 3D 投影点 (白心彩色边)
                pc = T_w2c[:3,:3] @ points_3d[j] + T_w2c[:3,3]
                uv_h = self.dataset[i].cam.k @ pc
                uv = (int(uv_h[0]/uv_h[2]), int(uv_h[1]/uv_h[2]))
                cv2.circle(img, uv, 6, (255, 255, 255), -1)
                cv2.circle(img, uv, 4, color, -1)
            
            out.write(img)
        out.release()

# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    config_path = os.path.join(save_dir, "handle_selection.json")
    
    vrs_files = [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')]
    vrs = os.path.join(args.mps_path, vrs_files[0])
    hand_csv = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    
    ds = AriaDataset(args.mps_path, vrs, hand_csv, save_dir)
    solver = AriaStaticSolver(ds, config_path)
    
    tracks_raw = solver.run_tracking()
    pts_3d_world, metrics, error_list = solver.solve_3d_with_metrics(tracks_raw)
    
    # --- 生成 JSON 逐帧数据 ---
    per_frame_data = []
    for i in range(solver.split_idx + 1):
        s = ds[i]
        T_w2c = np.linalg.inv(s.cam.c2w)
        
        current_frame_p3d_cam = []
        current_frame_p2d_vis = []
        
        for p_world in pts_3d_world:
            # 3D 坐标转换到相机系
            p_cam = T_w2c[:3, :3] @ p_world + T_w2c[:3, 3]
            current_frame_p3d_cam.append(p_cam.tolist())
            
            # 投影到像素系 (完全对齐，无需坐标翻转)
            uv_h = s.cam.k @ p_cam
            u, v = uv_h[0] / uv_h[2], uv_h[1] / uv_h[2]
            current_frame_p2d_vis.append([float(u), float(v)])
            
        per_frame_data.append({
            "frame": i,
            "timestamp_ns": int(s.ts),
            "p3d_cam": current_frame_p3d_cam,
            "p2d_vis": current_frame_p2d_vis
        })

    # --- 保存最终结果 ---
    output_json = {
        "metadata": {
            "reference_frame_idx": solver.ref_idx,
            "split_frame_idx": solver.split_idx,
            "num_points": solver.num_pts,
            "average_reprojection_rmse_px": metrics["rmse"]
        },
        "static_handle_points_world": pts_3d_world.tolist(),
        "adjacent_distances_m": metrics["dists"],
        "frames": per_frame_data
    }

    with open(os.path.join(save_dir, "static_solver_results.json"), 'w') as f:
        json.dump(output_json, f, indent=4)
    
    # --- 渲染与导出 ---
    evaluator = StaticQualitativeEvaluator(ds, save_dir)
    evaluator.export_3d_alignment_check(pts_3d_world, args.mps_path)
    evaluator.plot_error_analysis(error_list)
    
    vis = EnhancedVisualizer(ds, save_dir)
    vis.render_static_rainbow(tracks_raw, pts_3d_world, solver.split_idx)
    
    print(f"\n[Success] 静态解算完成！")
    print(f" - 结果 JSON: {os.path.join(save_dir, 'static_solver_results.json')}")
    print(f" - 调试视频: {os.path.join(save_dir, 'static_solver_vis.mp4')}")
    print(f" - 对齐点云: {os.path.join(save_dir, 'static_solver_pc.ply')}")

if __name__ == "__main__":
    main()