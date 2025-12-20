# -*- coding: utf-8 -*-
# @Time    : 2025/12/04
# @Author  : Assistant
# @FileName: static_solver.py

import os
import cv2
import sys
import json
import torch
import imageio
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

# ==============================================================================
# [Module 1] 几何数学类
# ==============================================================================

class MathUtils:
    @staticmethod
    def triangulate_n_views(poses, k_matrices, tracks_2d):
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
# [Module 2] 静态解算与定性评估类
# ==============================================================================

class AriaStaticSolver:
    def __init__(self, dataset: AriaDataset, config_path: str, device="cuda"):
        self.dataset = dataset
        self.device = device
        self.img_h = dataset[0].cam.rgb.shape[0]
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.ref_idx = self.config["reference_frame_idx"]
        self.split_idx = self.config["split_frame_idx"]
        self.init_pts_raw = self.config["handle_points_2d_raw"]

    def run_tracking(self):
        print(f"\n[Task 1/5] 执行 CoTracker 静态追踪...")
        frames = []
        target_res = 512
        scale = target_res / self.img_h
        for i in tqdm(range(self.split_idx + 1), desc="Loading Video"):
            img_vis = self.dataset[i].cam.rgb
            frames.append(cv2.resize(np.rot90(img_vis, k=1), (target_res, target_res)))
        
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(self.device).float()
        queries = [[float(self.ref_idx), p[0] * scale, p[1] * scale] for p in self.init_pts_raw]
        
        ckpt = os.path.join(os.path.dirname(__file__), "checkpoints/scaled_offline.pth")
        model = CoTrackerPredictor(checkpoint=ckpt).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_tracks, _ = model(video, queries=torch.tensor([queries]).to(self.device).float())
        
        tracks_raw = pred_tracks[0].cpu().numpy() / scale
        del video, model; torch.cuda.empty_cache()
        return tracks_raw

    def solve_3d_with_metrics(self, tracks_raw):
        """解算 3D 并计算重投影误差曲线"""
        print(f"[Task 2/5] 正在执行多视图三角化与误差分析...")
        pts_3d = []
        errors_per_frame = [] # 用于定性评估绘图

        for pt_idx in range(3):
            poses, ks, obs_2d = [], [], []
            for i in range(self.split_idx + 1):
                s = self.dataset[i]
                poses.append(s.cam.c2w); ks.append(s.cam.k); obs_2d.append(tracks_raw[i, pt_idx])
            p3d = MathUtils.triangulate_n_views(poses, ks, obs_2d)
            pts_3d.append(p3d)
            
        pts_3d = np.array(pts_3d)

        # --- 计算重投影误差指标 ---
        total_rmse = 0
        for i in range(self.split_idx + 1):
            s = self.dataset[i]
            T_w2c = np.linalg.inv(s.cam.c2w)
            frame_errs = []
            for j in range(3):
                p_c = T_w2c[:3,:3] @ pts_3d[j] + T_w2c[:3,3]
                uv_h = s.cam.k @ p_c
                u_re, v_re = uv_h[0]/uv_h[2], uv_h[1]/uv_h[2]
                dist = np.linalg.norm(np.array([u_re, v_re]) - tracks_raw[i, j])
                frame_errs.append(dist)
            errors_per_frame.append(np.mean(frame_errs))
            total_rmse += np.mean(frame_errs)
        
        avg_rmse = total_rmse / (self.split_idx + 1)
        L01 = np.linalg.norm(pts_3d[0] - pts_3d[1])
        L12 = np.linalg.norm(pts_3d[2] - pts_3d[1])
        
        print(f"\n" + "="*40)
        print(f"解算精度报告:")
        print(f" -> 平均重投影误差: {avg_rmse:.3f} 像素")
        print(f" -> 把手总长度: {(L01+L12)*100:.2f} cm")
        print(f" -> 状态评级: {'⭐优质' if avg_rmse < 3 else '⚠️待检查'}")
        print("="*40)
        
        return pts_3d, {"L01": L01, "L12": L12, "rmse": avg_rmse}, errors_per_frame

# ==============================================================================
# [Module 3] 静态定性评估器
# ==============================================================================

class StaticQualitativeEvaluator:
    def __init__(self, dataset, save_dir):
        self.dataset = dataset
        self.save_dir = save_dir

    def export_3d_alignment_check(self, pts_3d, mps_path):
        """将 3D 解算点与场景点云融合，验证物理贴合度"""
        print("[Task 4/5] 正在生成 3D 物理对齐评估文件...")
        all_geoms = []
        pc_path = os.path.join(mps_path, "aria", "pc_scene_and_head_traj.ply")
        
        if os.path.exists(pc_path):
            pcd = o3d.io.read_point_cloud(pc_path)
            pcd.paint_uniform_color([0.3, 0.3, 0.3]) # 暗色背景
            all_geoms.append(pcd)

        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # L(红) C(绿) R(蓝)
        for i in range(3):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate(pts_3d[i])
            sphere.paint_uniform_color(colors[i])
            all_geoms.append(sphere.sample_points_uniformly(200))
            
        combined = o3d.geometry.PointCloud()
        for g in all_geoms: combined += g
        
        out_path = os.path.join(self.save_dir, "static_solver_pc.ply")
        o3d.io.write_point_cloud(out_path, combined)
        print(f" -> 3D 检查文件已保存: {out_path} (请用 MeshLab 查看)")

    def plot_error_analysis(self, errors):
        """绘制重投影误差随时间的变化图"""
        print("[Task 5/5] 正在生成重投影误差分析图表...")
        plt.figure(figsize=(10, 4))
        plt.plot(errors, color='purple', linewidth=2, label='Reprojection Error')
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', label='Average RMSE')
        plt.title("Static Solution Stability (Pixels)")
        plt.xlabel("Frame Index")
        plt.ylabel("Error (px)")
        plt.legend(); plt.grid(True)
        
        plot_path = os.path.join(self.save_dir, "static_solver_eval.png")
        plt.savefig(plot_path)
        print(f" -> 误差曲线图已保存: {plot_path}")

# ==================== 渲染器 ====================
class EnhancedVisualizer:
    def __init__(self, dataset, save_dir):
        self.dataset = dataset; self.save_dir = save_dir
        self.cmap = plt.get_cmap('jet')

    def render_static_rainbow(self, tracks_raw, points_3d, split_idx, fps=10):
        print(f"[Task 3/5] 正在生成彩虹轨迹评估视频...")
        video_path = os.path.join(self.save_dir, "static_solver_vis.mp4")
        h_vis, w_vis = self.dataset[0].cam.rgb.shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vis, h_vis))
        gif_frames = []

        for i in tqdm(range(split_idx + 1), desc="Rendering"):
            img = self.dataset[i].cam.rgb.copy()
            # 1. Rainbow Trail
            for pt_idx in range(3):
                for t in range(max(1, i - 15), i + 1):
                    p1 = (int(h_vis - 1 - tracks_raw[t-1, pt_idx, 1]), int(tracks_raw[t-1, pt_idx, 0]))
                    p2 = (int(h_vis - 1 - tracks_raw[t, pt_idx, 1]), int(tracks_raw[t, pt_idx, 0]))
                    cv2.line(img, p1, p2, [int(c*255) for c in self.cmap(t/(split_idx+1))[:3][::-1]], 2)
            # 2. 3D Reprojection
            T_w2c = np.linalg.inv(self.dataset[i].cam.c2w)
            for j, p3 in enumerate(points_3d):
                pc = T_w2c[:3,:3] @ p3 + T_w2c[:3,3]
                uv_h = self.dataset[i].cam.k @ pc
                uv = (int(h_vis-1-uv_h[1]/uv_h[2]), int(uv_h[0]/uv_h[2]))
                cv2.circle(img, uv, 10, [(0,0,255),(0,255,0),(255,0,0)][j], -1)
            
            cv2.rectangle(img, (0, 0), (320, 60), (0,0,0), -1)
            cv2.putText(img, f"STATIC F:{i}", (20, 40), 1, 1.5, (0, 255, 255), 2)
            out.write(img)
            gif_frames.append(cv2.cvtColor(cv2.resize(img,(0,0),fx=0.5,fy=0.5), cv2.COLOR_BGR2RGB))
            
        out.release()
        imageio.mimsave(video_path.replace(".mp4", ".gif"), gif_frames, fps=fps, loop=0)

# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    config_path = os.path.join(save_dir, "handle_selection.json")
    vrs = os.path.join(args.mps_path, [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')][0])
    hand_csv = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    
    ds = AriaDataset(args.mps_path, vrs, hand_csv, save_dir)
    solver = AriaStaticSolver(ds, config_path)
    
    tracks_raw = solver.run_tracking()
    pts_3d, metrics, error_list = solver.solve_3d_with_metrics(tracks_raw)
    
    # 保存结果
    with open(os.path.join(save_dir, "static_solver_results.json"), 'w') as f:
        json.dump({"metadata": {"split_frame": solver.split_idx}, "handle_points_3d": pts_3d.tolist(), "rigid_constraints": metrics}, f, indent=4)

    # 可视化与定性评估
    EnhancedVisualizer(ds, save_dir).render_static_rainbow(tracks_raw, pts_3d, solver.split_idx)
    
    evaluator = StaticQualitativeEvaluator(ds, save_dir)
    evaluator.export_3d_alignment_check(pts_3d, args.mps_path)
    evaluator.plot_error_analysis(error_list)

if __name__ == "__main__":
    main()

# python static_solver.py --mps_path "../data/mps_open_cabinet_5_vrs/"