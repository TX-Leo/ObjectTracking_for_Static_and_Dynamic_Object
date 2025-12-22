# -*- coding: utf-8 -*-
import os
import cv2
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# ==================== 路径处理 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from aria.AriaDataset import AriaDataset
from cotracker.predictor import CoTrackerPredictor

# ==============================================================================
# [Helper] One-Euro Filter
# ==============================================================================

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev, self.dx_prev = None, None

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        d_alpha = self._alpha(self.dcutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = d_alpha * dx + (1 - d_alpha) * self.dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(dx_hat)
        alpha = self._alpha(cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat

# ==============================================================================
# [Module] 动态处理器 V3.1 - Robust Interaction Solver
# ==============================================================================

class AriaDynamicSolver:
    def __init__(self, dataset: AriaDataset, config_path: str, static_json_path: str, device="cuda"):
        self.dataset = dataset
        self.device = device
        self.config_path = config_path
        self.img_h = dataset[0].cam.h # 640
        
        with open(static_json_path, 'r') as f:
            static_data = json.load(f)
        
        static_pts_world = np.array(static_data["static_handle_points_world"], dtype=np.float32)
        self.p_ref_origin = static_pts_world[0].copy()
        self.model_pts_3d = (static_pts_world - self.p_ref_origin).astype(np.float32)
        
        self.split_idx = static_data["metadata"]["split_frame_idx"]
        
        with open(self.config_path, 'r') as f:
            self.config_content = json.load(f)
            self.ref_idx = self.config_content["reference_frame_idx"]
            self.num_pts = len(self.config_content["handle_points_2d_raw"])

        self.filter_r = OneEuroFilter(freq=dataset.fps, mincutoff=0.5, beta=0.01)
        self.filter_t = OneEuroFilter(freq=dataset.fps, mincutoff=1.0, beta=0.01)

    def run_tracking(self):
        print(f"\n[Task 1/3] 执行 CoTracker 轨迹提取...")
        frames = []
        for i in tqdm(range(self.ref_idx, len(self.dataset)), desc="Loading Video"):
            # 必须使用 Dataset 预处理后的图像
            frames.append(cv2.cvtColor(self.dataset[i].cam.rgb, cv2.COLOR_BGR2RGB))
        
        video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(self.device).float()
        # handle_points_2d_vis 对应 640x640 屏幕点击坐标 (x, y)
        queries = [[0.0, float(p[0]), float(p[1])] for p in self.config_content["handle_points_2d_vis"]]
        
        model_path = os.path.join(os.path.dirname(__file__), "checkpoints/scaled_offline.pth")
        model = CoTrackerPredictor(checkpoint=model_path).to(self.device)
        with torch.no_grad():
            pred_tracks, vis_scores = model(video, queries=torch.tensor([queries]).to(self.device).float())
        
        return pred_tracks[0].cpu().numpy(), vis_scores[0].cpu().numpy()

    def solve_dynamic(self, tracks_vis, visibility):
        print(f"[Task 2/3] 执行解算 (手部点在相机系)...")
        results = []
        start_offset = self.split_idx - self.ref_idx
        last_rvec, last_tvec = None, None
        
        for i in tqdm(range(start_offset, len(tracks_vis)), desc="Solving"):
            abs_idx = i + self.ref_idx
            s = self.dataset[abs_idx]
            K, T_c2w = s.cam.k, s.cam.c2w
            
            # 1. 寻找手 (s.hands 中的 keypoints 已经在相机系)
            best_hand = max(s.hands, key=lambda h: h.confidence) if s.hands else None
            if best_hand is None or best_hand.confidence < 0.4:
                results.append({"frame": abs_idx, "status": "HAND_LOST", "metrics": {"rmse":0,"grasp_dist_cm":0,"hand_conf":0}})
                last_rvec, last_tvec = None, None
                continue

            # 2. 坐标转换 (Visual -> Camera Raw Landscape)
            img_pts_raw = []
            for pt in tracks_vis[i]:
                u_raw = pt[1]
                v_raw = self.img_h - 1 - pt[0]
                img_pts_raw.append([u_raw, v_raw])
            img_pts_raw = np.array(img_pts_raw, dtype=np.float32)

            # 3. PnP 解算
            if last_rvec is None:
                success, rvec, tvec, _ = cv2.solvePnPRansac(self.model_pts_3d, img_pts_raw, K, None)
            else:
                success, rvec, tvec = cv2.solvePnP(self.model_pts_3d, img_pts_raw, K, None, 
                                                  rvec=last_rvec, tvec=last_tvec, 
                                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

            if success:
                rvec, tvec = self.filter_r(rvec), self.filter_t(tvec)
                last_rvec, last_tvec = rvec, tvec

                # --- [关键修改：全部在相机系操作] ---
                R_obj2cam, _ = cv2.Rodrigues(rvec)
                
                # 物体点在相机系下的 3D 坐标
                pts_cam = [(R_obj2cam @ p_m + tvec.squeeze()) for p_m in self.model_pts_3d]
                pts_cam = np.array(pts_cam)
                obj_center_cam = np.mean(pts_cam, axis=0)

                # 手部点已经在相机系
                t_tip_cam = np.array(best_hand.hand_keypoints_3d[0]) # Thumb Tip
                i_tip_cam = np.array(best_hand.hand_keypoints_3d[1]) # Index Tip
                hand_center_cam = (t_tip_cam + i_tip_cam) / 2.0
                
                # 现在算出来的 grasp_dist 应该是厘米级的
                grasp_dist = np.linalg.norm(obj_center_cam - hand_center_cam)

                # 计算物体世界坐标 (用于保存 JSON)
                pts_world = [(T_c2w[:3,:3] @ p_c + T_c2w[:3,3]) for p_c in pts_cam]

                # 4. 定义统一的投影函数 (Camera 3D -> Visual 2D)
                def cam_3d_to_vis_xy(p_c):
                    if p_c[2] <= 0: return None
                    u_r = K[0,0] * (p_c[0]/p_c[2]) + K[0,2]
                    v_r = K[1,1] * (p_c[1]/p_c[2]) + K[1,2]
                    # 匹配顺时针 90 度：x = H-1-v, y = u
                    return (int(self.img_h - 1 - v_r), int(u_r))

                reproj_err = np.sqrt(np.mean(np.sum((cv2.projectPoints(self.model_pts_3d, rvec, tvec, K, None)[0].squeeze() - img_pts_raw)**2, axis=1)))

                results.append({
                    "frame": abs_idx,
                    "status": "LOCKED" if (grasp_dist < 0.15 and reproj_err < 10.0) else "DRIFTING",
                    "p3d_world": [p.tolist() for p in pts_world],
                    "p2d_vis": [cam_3d_to_vis_xy(p) for p in pts_cam],
                    "hand_p2d_vis": [cam_3d_to_vis_xy(t_tip_cam), cam_3d_to_vis_xy(i_tip_cam)],
                    "metrics": {
                        "rmse": float(reproj_err),
                        "grasp_dist_cm": float(grasp_dist * 100),
                        "hand_conf": float(best_hand.confidence)
                    }
                })
            else:
                results.append({"frame": abs_idx, "status": "LOST", "metrics": {"rmse":0,"grasp_dist_cm":0,"hand_conf":0}})
                last_rvec, last_tvec = None, None

        return results

# ==============================================================================
# [Module] 渲染器 (修复坐标索引)
# ==============================================================================

class DynamicVisualizer:
    @staticmethod
    def render(dataset, results, save_path):
        print("[Task 3/3] 渲染 HUD...")
        video_p = os.path.join(save_path, "dynamic_interaction_v4.mp4")
        h, w = dataset[0].cam.h, dataset[0].cam.w
        out = cv2.VideoWriter(video_p, cv2.VideoWriter_fourcc(*'mp4v'), int(dataset.fps), (w, h))
        
        res_map = {r["frame"]: r for r in results}
        FONT = cv2.FONT_HERSHEY_DUPLEX

        for f_idx in tqdm(range(results[0]["frame"], len(dataset)), desc="Rendering"):
            img = dataset[f_idx].cam.rgb.copy()
            res = res_map.get(f_idx, None)
            
            # HUD Background
            cv2.rectangle(img, (0,0), (w, 50), (20,20,20), -1)
            
            if res and res.get("status") in ["LOCKED", "DRIFTING"]:
                m = res["metrics"]
                st_col = (0,255,0) if res["status"] == "LOCKED" else (0,150,255)
                cv2.putText(img, f"MODE: {res['status']} | FRAME: {f_idx}", (20, 35), FONT, 0.7, st_col, 1)

                # 右侧监控面板
                overlay = img.copy()
                cv2.rectangle(overlay, (w-210, 60), (w-10, 190), (40,40,40), -1)
                img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                
                info = [
                    (f"RMSE: {m.get('rmse',0.0):.1f} px", (255,255,255)),
                    (f"GRASP: {m.get('grasp_dist_cm',0.0):.1f} cm", (0,255,255) if m.get('grasp_dist_cm',0.0)<12 else (0,0,255)),
                    (f"H-CONF: {m.get('hand_conf',0.0):.2f}", (0,255,0))
                ]
                for j, (txt, col) in enumerate(info):
                    cv2.putText(img, txt, (w-200, 90+j*35), FONT, 0.5, col, 1)

                # --- 投影绘制 (关键修正：uv 已经是 (x, y) 顺序) ---
                # 1. 绘制手部指尖 (紫色十字)
                for uv in res.get("hand_p2d_vis", []):
                    if uv: cv2.drawMarker(img, uv, (255,0,255), cv2.MARKER_TILTED_CROSS, 15, 2)

                # 2. 绘制物体骨架
                ov = res.get("p2d_vis", [])
                for j, uv in enumerate(ov):
                    if uv is None: continue
                    col = [int(c*255) for c in plt.get_cmap('rainbow')(j/len(ov))[:3][::-1]]
                    cv2.circle(img, uv, 6, (255,255,255), -1)
                    cv2.circle(img, uv, 4, col, -1)
                    if j > 0 and ov[j-1]:
                        cv2.line(img, ov[j-1], uv, (255,255,255), 1)
            else:
                msg = f"MODE: {res['status'] if res else 'SEARCHING'} | FRAME: {f_idx}"
                cv2.putText(img, msg, (20, 35), FONT, 0.7, (0,0,255), 1)

            out.write(img)
        out.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    config_path = os.path.join(save_dir, "handle_selection.json")
    static_json = os.path.join(save_dir, "static_solver_results.json")
    
    vrs_files = [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')]
    vrs_f = os.path.join(args.mps_path, vrs_files[0])
    hand_c = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    
    ds = AriaDataset(args.mps_path, vrs_f, hand_c, save_dir)
    solver = AriaDynamicSolver(ds, config_path, static_json)
    
    tracks_vis, vis_scores = solver.run_tracking()
    results = solver.solve_dynamic(tracks_vis, vis_scores)
    
    if results:
        with open(os.path.join(save_dir, "dynamic_solver_results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        DynamicVisualizer.render(ds, results, save_dir)

if __name__ == "__main__":
    main()

# 优化空间：
#     1.交互约束优化 (Interaction-Constrained Optimization)
#     2.因子图优化 (Factor Graph Optimization / GTSAM)
#     3. 手部作为“3D 锚点”的深度校正 (Hand-as-a-Proxy)