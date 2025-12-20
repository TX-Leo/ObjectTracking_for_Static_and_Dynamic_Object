# -*- coding: utf-8 -*-
# @Time    : 2025/12/04
# @Author  : Assistant
# @FileName: dynamic_solver.py
# @Description: 第三阶段：动态解算器。利用全量 3D 世界坐标对齐，结合 CoTracker 射线与手部 3D 锚点解算。

import os
import cv2
import sys
import json
import torch
import argparse
import imageio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d

# ==================== 路径处理 ====================
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
# [Module 1] 3D Lifting 核心几何工具
# ==============================================================================

class LiftingUtils:
    @staticmethod
    def get_world_ray(u_v, v_v, K, T_d2w, h):
        """
        从校正后的虚拟相机（设备中心）发出世界射线
        u_v, v_v: Portrait 像素坐标
        """
        # 1. Vis -> Raw (Landscape)
        u_raw, v_raw = v_v, h - 1 - u_v
        # 2. Raw Pixel -> Device Frame Direction
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        dir_device = np.array([(u_raw - cx)/fx, (v_raw - cy)/fy, 1.0])
        dir_device /= np.linalg.norm(dir_device)
        # 3. Device Frame -> World Frame (通过 T_device_to_world)
        ray_origin = T_d2w[:3, 3]
        ray_dir = T_d2w[:3, :3] @ dir_device
        return ray_origin.astype(np.float32), ray_dir.astype(np.float32)

    @staticmethod
    def transform_hand_to_world(p_hand_cam, T_d2w, T_c2d):
        """将相机系下的手部点转为世界系"""
        # P_world = T_dev_world @ T_cam_dev @ P_cam
        T_c2w = T_d2w @ T_c2d
        p_homo = np.append(p_hand_cam, 1.0)
        p_world = (T_c2w @ p_homo)[:3]
        return p_world.astype(np.float32)

    @staticmethod
    def intersect_ray_sphere(o, d, center, radius):
        """射线与球体相交求解"""
        oc = o - center
        a, b = np.dot(d, d), 2.0 * np.dot(oc, d)
        c = np.dot(oc, oc) - radius**2
        delta = b**2 - 4*a*c
        if delta < 0: return o + (-np.dot(oc, d)/a) * d
        t = (-b - np.sqrt(delta)) / (2.0 * a)
        return o + t * d

# ==============================================================================
# [Module 2] 动态处理器
# ==============================================================================

class AriaDynamicSolver:
    def __init__(self, dataset: AriaDataset, config_path: str, static_json_path: str, device="cuda"):
        self.dataset = dataset
        self.device = device
        self.img_h = dataset[0].cam.rgb.shape[0]
        
        with open(static_json_path, 'r') as f:
            data = json.load(f)
        self.static_pts_3d = np.array(data["handle_points_3d"])
        self.L01, self.L12 = data["rigid_constraints"]["L01"], data["rigid_constraints"]["L12"]
        self.split_idx = data["metadata"]["split_frame"]
        with open(config_path, 'r') as f:
            data = json.load(f)
        self.ref_idx = data["reference_frame_idx"] # 选点时的清晰帧

        # --- 关键：获取外参 T_camera_to_device ---
        rgb_calib = self.dataset.device_calib.get_camera_calib("camera-rgb")
        self.T_c2d = rgb_calib.get_transform_device_camera().inverse().to_matrix()

    def run_tracking(self):
        """切片追踪逻辑"""
        print(f"\n[Task 1/3] 执行动态切片追踪 (Frame {self.split_idx} -> End)...")
        frames = []
        target_res = 512
        scale = target_res / self.img_h
        for i in tqdm(range(self.ref_idx, len(self.dataset)), desc="Loading Video"):
            frames.append(cv2.resize(np.rot90(self.dataset[i].cam.rgb, k=1), (target_res, target_res)))
        
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(self.device).float()
        
        # 使用 ref_idx 的投影作为查询点
        s_ref = self.dataset[self.ref_idx]
        T_w2d = np.linalg.inv(s_ref.cam.c2w)
        queries = []
        for pt3 in self.static_pts_3d:
            pt_d = T_w2d[:3,:3] @ pt3 + T_w2d[:3,3]
            uv_h = s_ref.cam.k @ pt_d
            queries.append([0.0, (uv_h[0]/uv_h[2]) * scale, (uv_h[1]/uv_h[2]) * scale])
        
        model = CoTrackerPredictor(checkpoint=os.path.join(os.path.dirname(__file__), "checkpoints/scaled_offline.pth")).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                pred_tracks, _ = model(video, queries=torch.tensor([queries]).to(self.device).float())
        
        tracks_raw = pred_tracks[0].cpu().numpy() / scale
        del video, model; torch.cuda.empty_cache()
        return tracks_raw

    def solve_dynamic(self, tracks_raw):
        """利用全量 3D 世界位姿执行解算"""
        print("[Task 2/3] 正在执行 3D Lifting (Full World Alignment)...")
        results = []
        track_offset = self.split_idx - self.ref_idx # 70-40=30 
        for i in tqdm(range(len(tracks_raw)-track_offset), desc="Lifting"):
            abs_idx = i + self.split_idx
            print(f'abs_idx:{abs_idx}')
            s = self.dataset[abs_idx]
            K, T_d2w = s.cam.k, s.cam.c2w
            
            # 1. 寻找手部世界坐标锚点
            hand = next((h for h in s.hands if not h.is_right), None)
            if hand:
                h_p1_w = LiftingUtils.transform_hand_to_world(hand.hand_keypoints_3d[1], T_d2w, self.T_c2d) # index fingertip
                h_p0_w = LiftingUtils.transform_hand_to_world(hand.hand_keypoints_3d[0], T_d2w, self.T_c2d) # thumb fingertip
                world_anchor = (h_p1_w + h_p0_w) / 2.0
            else:
                world_anchor = None

            # 2. 生成射线 (基于虚拟相机的 2D 像素)
            uvs_v = [[self.img_h - 1 - tracks_raw[i+ track_offset, n, 1], tracks_raw[i+ track_offset, n, 0]] for n in range(3)]
            
            # P1: 中心点 (射线 + 世界锚点投影)
            o1, d1 = LiftingUtils.get_world_ray(uvs_v[1][0], uvs_v[1][1], K, T_d2w, self.img_h)
            if world_anchor is not None:
                # 寻找射线上离世界手部位置最近的点
                t1 = np.dot(world_anchor - o1, d1)
                p1_3d = o1 + t1 * d1
            else:
                # 丢手时保持上一帧深度或使用静态深度
                p1_3d = o1 + np.linalg.norm(results[-1]['p3d'][1] - o1 if results else 0.5) * d1

            # P0 & P2: 射线 + 刚体约束
            o0, d0 = LiftingUtils.get_world_ray(uvs_v[0][0], uvs_v[0][1], K, T_d2w, self.img_h)
            p0_3d = LiftingUtils.intersect_ray_sphere(o0, d0, p1_3d, self.L01)
            o2, d2 = LiftingUtils.get_world_ray(uvs_v[2][0], uvs_v[2][1], K, T_d2w, self.img_h)
            p2_3d = LiftingUtils.intersect_ray_sphere(o2, d2, p1_3d, self.L12)
            
            pts_3d = [p0_3d, p1_3d, p2_3d]
            
            # 重新投影用于验证
            T_w2d = np.linalg.inv(T_d2w)
            uvs_re = []
            for p in pts_3d:
                p_d = T_w2d[:3,:3] @ p + T_w2d[:3,3]
                uv_h = K @ p_d
                uvs_re.append([int(self.img_h - 1 - (uv_h[1]/uv_h[2])), int(uv_h[0]/uv_h[2])])

            results.append({
                "frame": abs_idx,
                "timestamp_ns": int(s.ts),
                "p3d": [p.tolist() for p in pts_3d],
                "p2d_vis": uvs_re
            })
        return results

# ==================== 渲染器 ====================
class DynamicVisualizer:
    @staticmethod
    def render(dataset, results, save_path, fps=10):
        print("[Task 3/3] 渲染最终评估视频...")
        video_p = os.path.join(save_path, "dynamic_solver_vis.mp4")
        h, w = dataset[0].cam.rgb.shape[:2]
        out = cv2.VideoWriter(video_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        gif_frames = []

        for res in tqdm(results, desc="Rendering"):
            img = dataset[res["frame"]].cam.rgb.copy()
            uvs = res["p2d_vis"]
            colors = [(0,0,255), (0,255,0), (255,0,0)]
            for j, uv in enumerate(uvs):
                cv2.circle(img, tuple(uv), 10, colors[j], -1)
                cv2.circle(img, tuple(uv), 12, (255,255,255), 2)
            cv2.line(img, tuple(uvs[0]), tuple(uvs[1]), (255,255,255), 2)
            cv2.line(img, tuple(uvs[1]), tuple(uvs[2]), (255,255,255), 2)
            
            cv2.rectangle(img, (0, 0), (w, 60), (0,0,0), -1)
            cv2.putText(img, f"WORLD TRACKING F:{res['frame']}", (20, 40), 1, 1.5, (0, 255, 0), 2)
            out.write(img)
            gif_frames.append(cv2.cvtColor(cv2.resize(img,(0,0),fx=0.5,fy=0.5), cv2.COLOR_BGR2RGB))
        
        out.release()
        imageio.mimsave(video_p.replace(".mp4", ".gif"), gif_frames, fps=fps, loop=0)

# ==================== 评估器 ====================
class QualitativeEvaluator:
    def __init__(self, mps_path, result_json_path):
        self.mps_path = mps_path
        if not os.path.exists(result_json_path):
            raise FileNotFoundError(f"找不到结果文件: {result_json_path}")
            
        with open(result_json_path, 'r') as f:
            self.results = json.load(f)
        
        self.save_dir = os.path.dirname(result_json_path)
        # 尝试加载场景点云
        self.pc_path = os.path.join(mps_path, "aria", "pc_scene_and_head_traj.ply")

        
    def export_evaluation_ply(self):
        """
        核心功能：将环境点云和把手轨迹融合成一个 PLY 文件。
        你可以下载此文件在本地查看，避开 OpenGL 窗口报错。
        """
        print("[Eval] 正在融合场景与轨迹点云...")
        all_geoms = []

        # 1. 加载环境背景
        self.pc_hpath = os.path.join(self.mps_path, "aria", "pc_scene_and_head_traj.ply")
        if os.path.exists(self.pc_hpath):
            pcd = o3d.io.read_point_cloud(self.pc_hpath)
            # 将背景调暗，方便看轨迹
            pcd.paint_uniform_color([0.2, 0.2, 0.2])
            all_geoms.append(pcd)
        else:
            print("[Warning] 找不到背景点云 pc_scene.ply，将只生成轨迹。")

        # 2. 生成轨迹小球
        cmap = plt.get_cmap('jet')
        valid_results = [r for r in self.results if r["p3d"] is not None]
        total = len(valid_results)
        
        trajectory_pcd = o3d.geometry.PointCloud()
        pts_list = []
        colors_list = []

        for i, res in enumerate(valid_results):
            # 取把手中心点 (Index 1)
            center_pt = np.array(res["p3d"][1])
            
            # 为了让轨迹更明显，我们在这里生成一个小球的点云采样
            color = cmap(i / max(1, total))[:3]
            
            # 创建一个小球并采样
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
            sphere.translate(center_pt)
            sphere_pcd = sphere.sample_points_uniformly(number_of_points=100)
            sphere_pcd.paint_uniform_color(color)
            
            all_geoms.append(sphere_pcd)

        # 3. 合并并保存
        combined = o3d.geometry.PointCloud()
        for g in all_geoms:
            if isinstance(g, o3d.geometry.PointCloud):
                combined += g
            else:
                # 如果是 Mesh 则转换
                combined += g.sample_points_uniformly(100)

        out_path = os.path.join(self.save_dir, "dynamic_solver_pc.ply")
        o3d.io.write_point_cloud(out_path, combined)
        print(f"\n[Success] 3D 评估场景已保存至: {out_path}")
        print("请下载此 PLY 文件并使用 MeshLab 或 CloudCompare 查看轨迹。")

    def plot_trajectory_analysis(self):
        """
        分析轨迹的平滑度和位移
        """
        print("[Eval] 正在生成轨迹分析图表...")
        valid_res = [r for r in self.results if r["p3d"] is not None]
        frames = [r["frame"] for r in valid_res]
        
        # 计算相对于第一帧的位移
        pts = np.array([r["p3d"][1] for r in valid_res])
        start_pt = pts[0]
        displacements = np.linalg.norm(pts - start_pt, axis=1)

        plt.figure(figsize=(12, 5))
        
        # 子图 1: 累计位移
        plt.subplot(1, 2, 1)
        plt.plot(frames, displacements * 100, color='blue', linewidth=2)
        plt.title("Handle Displacement from Start")
        plt.xlabel("Frame")
        plt.ylabel("Distance (cm)")
        plt.grid(True)

        # 子图 2: Z 轴深度变化
        plt.subplot(1, 2, 2)
        plt.plot(frames, pts[:, 2], color='red', linewidth=2)
        plt.title("Z-axis (Depth) Variation")
        plt.xlabel("Frame")
        plt.ylabel("Z Coordinate (m)")
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "dynamic_solver_eval.png")
        plt.savefig(plot_path)
        print(f"[Success] 分析图表已保存至: {plot_path}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    config_path = os.path.join(save_dir, "handle_selection.json")
    static_json = os.path.join(save_dir, "static_solver_results.json")
    vrs_f = os.path.join(args.mps_path, [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')][0])
    hand_c = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    
    ds = AriaDataset(args.mps_path, vrs_f, hand_c, save_dir)
    solver = AriaDynamicSolver(ds, config_path, static_json)
    
    tracks = solver.run_tracking()
    results = solver.solve_dynamic(tracks)
    
    out_path = os.path.join(save_dir, "dynamic_solver_results.json")
    with open(out_path, 'w') as f: json.dump(results, f, indent=4)
    
    DynamicVisualizer.render(ds, results, save_dir, fps=int(getattr(ds, 'fps', 10)))

    try:
        evaluator = QualitativeEvaluator(args.mps_path, out_path)
        # 执行 3D 导出
        evaluator.export_evaluation_ply()
        # 执行图表生成
        evaluator.plot_trajectory_analysis()
    except Exception as e:
        print(f"[Error] 评估失败: {e}")
    
    print(f"完成！数据已对齐世界坐标系。")

if __name__ == "__main__":
    main()

# python dynamic_solver.py --mps_path "../data/mps_open_cabinet_5_vrs/"

