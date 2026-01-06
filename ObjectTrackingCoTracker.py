# -*- coding: utf-8 -*-
# @FileName: ObjectTrackingCoTracker.py

import os
import cv2
import sys
import json
import time
import torch
import numpy as np
import imageio
from tqdm import tqdm
import math
import gc
import argparse
from pathlib import Path
from typing import List, Dict
from scipy.signal import medfilt

from cotracker.predictor import CoTrackerPredictor

# [Config]
COTRACKER_RES = 640 
HEATMAP_SIGMA=  10

class ObjectTrackingCoTrackerPredictor:
    def __init__(self, checkpoint_path: str, res: int = 640, device: str = "cuda"):
        self.device = device
        self.res = res
        print(f"║ [Predictor] Loading CoTracker Weights: {os.path.basename(checkpoint_path)}")
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)

    def _infer_segment(self, frames_np: np.ndarray, queries: List):
        """执行单段推理并释放显存"""
        video_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)[None].to(self.device).float()
        queries_tensor = torch.tensor([queries]).to(self.device).float()
        
        with torch.no_grad():
            pred_tracks, pred_vis = self.model(video_tensor, queries=queries_tensor)
        
        tracks = pred_tracks[0].cpu().numpy()
        visibilities = pred_vis[0].cpu().numpy()
        
        del video_tensor, queries_tensor, pred_tracks, pred_vis
        torch.cuda.empty_cache()
        gc.collect()
        
        return tracks, visibilities

    def run_segmented_tracking(self, img_paths: List[str], ref_idx: int, split_idx: int, init_points: List):
        """分段追踪逻辑，适应长序列并记录性能指标"""
        sample_img = cv2.imread(img_paths[0])
        h_orig, w_orig = sample_img.shape[:2]
        scale_w, scale_h = self.res / w_orig, self.res / h_orig
        total_frames = len(img_paths)
        
        start_time = time.perf_counter()

        # --- Segment 1 ---
        print(f"║ [Inference] Segment 1: 0 -> {split_idx} (Ref: {ref_idx})")
        seg1_paths = img_paths[:split_idx + 1]
        frames_1 = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (self.res, self.res)) for p in seg1_paths])
        queries_1 = [[float(ref_idx), p[0] * scale_w, p[1] * scale_h] for p in init_points]
        tracks_1, vis_1 = self._infer_segment(frames_1, queries_1)
        del frames_1

        # --- Segment 2 ---
        print(f"║ [Inference] Segment 2: {split_idx} -> {total_frames-1}")
        seg2_paths = img_paths[split_idx:]
        frames_2 = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (self.res, self.res)) for p in seg2_paths])
        last_pts_scaled = tracks_1[split_idx] 
        queries_2 = [[0.0, pt[0], pt[1]] for pt in last_pts_scaled]
        tracks_2, vis_2 = self._infer_segment(frames_2, queries_2)
        del frames_2

        # --- Merge ---
        full_tracks = np.concatenate([tracks_1, tracks_2[1:]], axis=0)
        full_vis = np.concatenate([vis_1, vis_2[1:]], axis=0)
        full_tracks[:, :, 0] /= scale_w
        full_tracks[:, :, 1] /= scale_h

        # Boundary logic
        for t in range(total_frames):
            for n in range(full_tracks.shape[1]):
                u, v = full_tracks[t, n]
                if not full_vis[t, n] or u < 0 or u >= w_orig or v < 0 or v >= h_orig:
                    full_tracks[t, n] = [0.0, 0.0]
                    full_vis[t, n] = 0

        duration = time.perf_counter() - start_time
        meta = {
            "throughput_fps": round(total_frames / duration, 2),
            "inference_duration_sec": round(duration, 3),
            "original_resolution": [w_orig, h_orig],
            "inference_resolution": [self.res, self.res]
        }
        return full_tracks, full_vis, meta

class ObjectTrackingCoTrackerEvaluator:
    @staticmethod
    def _compute_jitter_metrics(tracks, visibilities):
        """核心逻辑：计算一组序列的平均抖动率"""
        T, N, _ = tracks.shape
        if T < 8 or N < 2: return 0.0, 1.0 # 样本太少不具统计意义
        
        jitter_ratios = []
        for i in range(N):
            for j in range(i + 1, N):
                mask = (visibilities[:, i] > 0) & (visibilities[:, j] > 0)
                if mask.sum() < 8: continue
                
                dists = np.linalg.norm(tracks[mask, i] - tracks[mask, j], axis=1)
                smooth_dists = medfilt(dists, kernel_size=7) if len(dists) > 7 else dists
                jitter = dists - smooth_dists
                jitter_ratio = np.std(jitter) / (np.mean(dists) + 1e-6)
                jitter_ratios.append(jitter_ratio)
        
        avg_jitter = np.mean(jitter_ratios) if jitter_ratios else 1.0
        score = max(0, 100 * (1 - avg_jitter * 20))
        return round(score, 2), round(avg_jitter, 5)

    @staticmethod
    def evaluate_all_phases(tracks, visibilities, ref_idx, split_idx):
        """同时评估全局和三个关键阶段"""
        # 全局
        g_score, g_jitter = ObjectTrackingCoTrackerEvaluator._compute_jitter_metrics(tracks, visibilities)
        
        # 阶段 A: Pre-Anchor (0 -> ref)
        s1_score, s1_jitter = ObjectTrackingCoTrackerEvaluator._compute_jitter_metrics(
            tracks[:ref_idx+1], visibilities[:ref_idx+1])
        
        # 阶段 B: Triangulation Interval (ref -> split)
        s2_score, s2_jitter = ObjectTrackingCoTrackerEvaluator._compute_jitter_metrics(
            tracks[ref_idx:split_idx+1], visibilities[ref_idx:split_idx+1])
        
        # 阶段 C: Manipulation (split -> end)
        s3_score, s3_jitter = ObjectTrackingCoTrackerEvaluator._compute_jitter_metrics(
            tracks[split_idx:], visibilities[split_idx:])

        return {
            "global": {"score": g_score, "jitter": g_jitter},
            "stages": [
                {"name": "PRE_ANCHOR", "range": [0, ref_idx], "score": s1_score, "jitter": s1_jitter},
                {"name": "TRIANGULATION", "range": [ref_idx, split_idx], "score": s2_score, "jitter": s2_jitter},
                {"name": "MANIPULATION", "range": [split_idx, len(tracks)-1], "score": s3_score, "jitter": s3_jitter}
            ]
        }
    
class ObjectTrackingHeatmapGenerator:
    """[新增] 热图生成类：计算高斯热图并保存到对应帧文件夹"""
    def __init__(self, sigma=12):
        self.sigma = sigma

    def _generate_gaussian_heatmap(self, points, vis, h, w):
        heatmap = np.zeros((h, w), dtype=np.float32)
        kernel_size = int(self.sigma * 4 + 1)
        grid = np.arange(kernel_size) - (kernel_size // 2)
        gaussian_1d = np.exp(-grid**2 / (2 * self.sigma**2))
        kernel_2d = np.outer(gaussian_1d, gaussian_1d)

        for i in range(len(points)):
            if vis[i] == 0: continue
            ix, iy = int(round(points[i, 0])), int(round(points[i, 1]))
            x1, x2 = ix - kernel_size // 2, ix + kernel_size // 2 + 1
            y1, y2 = iy - kernel_size // 2, iy + kernel_size // 2 + 1
            kx1, kx2 = max(0, -x1), min(kernel_size, w - x1)
            ky1, ky2 = max(0, -y1), min(kernel_size, h - y1)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], kernel_2d[ky1:ky2, kx1:kx2])
        return (heatmap * 255).astype(np.uint8)

    def run(self, tracks, visibilities, img_paths, save_root, fps):
        print(f"║ [Heatmap] Generating Gaussian Heatmaps (Sigma: {self.sigma})...")
        h, w = cv2.imread(img_paths[0]).shape[:2]
        video_path = os.path.join(save_root, "ot_heatmap_video_vis.mp4")
        out_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), False)
        gif_frames = []

        for t in tqdm(range(len(img_paths)), desc="Heatmap Processing"):
            heatmap = self._generate_gaussian_heatmap(tracks[t], visibilities[t], h, w)
            # 保存到每一帧文件夹下
            frame_dir = os.path.dirname(img_paths[t])
            cv2.imwrite(os.path.join(frame_dir, "heatmap.png"), heatmap)
            out_video.write(heatmap)
            gif_frames.append(heatmap)
        
        out_video.release()
        imageio.mimsave(os.path.join(save_root, "ot_heatmap_video_vis.gif"), gif_frames, fps=fps, loop=0)

class ObjectTrackingCoTrackerVisualizer:
    def __init__(self, mps_path, fps):
        self.mps_path = mps_path
        self.fps = fps
        self.colors = [(52, 152, 219), (46, 204, 113), (231, 76, 60), (241, 196, 15), (155, 89, 182)]
        
    def _draw_hud(self, img, t, vis_at_t, split_idx):
        h, w = img.shape[:2]
        panel_w, panel_h = 240, 140
        x, y = w - panel_w - 20, 20
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.rectangle(img, (x, y), (x + panel_w, y + panel_h), (200, 200, 200), 1, cv2.LINE_AA)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, "OBJECT TRACKING ENGINE", (x + 10, y + 25), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (x + 10, y + 35), (x + panel_w - 10, y + 35), (100, 100, 100), 1)
        cv2.putText(img, f"FRAME: {t:05d}", (x + 15, y + 55), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        for i in range(len(vis_at_t)):
            if i >= 5: break # 只画前5个槽位
            cx, cy = x + 20 + i * 25, y + 75
            color = (46, 204, 113) if vis_at_t[i] > 0 else (80, 80, 80)
            cv2.circle(img, (cx, cy), 5, color, -1 if vis_at_t[i] > 0 else 1, cv2.LINE_AA)
        status = "IN MOTION" if t >= split_idx else "PRE-GRASP STATIC"
        s_color = (52, 152, 219) if t >= split_idx else (46, 204, 113)
        alpha = (math.sin(t * 0.3) + 1.2) / 2.2
        cv2.putText(img, f">> {status}", (x + 15, y + 115), font, 0.5, [int(c*alpha) for c in s_color], 1, cv2.LINE_AA)

    def _draw_tracks(self, img, t, tracks, vis, n_pts):
        for n in range(n_pts):
            if vis[t, n] == 0: continue
            color = self.colors[n % len(self.colors)]
            trail_len = 20
            for i in range(max(1, t - trail_len), t + 1):
                if vis[i-1, n] and vis[i, n]:
                    p1 = (int(tracks[i-1, n, 0]), int(tracks[i-1, n, 1]))
                    p2 = (int(tracks[i, n, 0]), int(tracks[i, n, 1]))
                    alpha = (i - (t - trail_len)) / trail_len
                    cv2.line(img, p1, p2, color, int(1 + 2*alpha), cv2.LINE_AA)
            ix, iy = int(tracks[t, n, 0]), int(tracks[t, n, 1])
            cv2.line(img, (ix-8, iy), (ix+8, iy), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (ix, iy-8), (ix, iy+8), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(img, (ix, iy), 4, color, -1, cv2.LINE_AA)

    def render(self, img_paths, tracks, vis, split_idx):
        aria_video_vis = os.path.join(self.mps_path, "aria", "aria_video_vis.mp4")
        cap_aria = cv2.VideoCapture(aria_video_vis)
        h_orig, w_orig = cv2.imread(img_paths[0]).shape[:2]
        w_aria = int(cap_aria.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_aria = int(cap_aria.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_raw = cv2.VideoWriter(os.path.join(self.mps_path, "aria", "ot_cotracker_video_raw.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w_orig, h_orig))
        out_aria = cv2.VideoWriter(os.path.join(self.mps_path, "aria", "ot_cotracker_video_aria.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w_aria, h_aria))
        
        gif_raw, gif_aria = [], []
        n_pts = tracks.shape[1]
        for t in tqdm(range(len(img_paths)), desc="Rendering"):
            frame_raw = cv2.imread(img_paths[t])
            ret, frame_aria = cap_aria.read()
            if not ret: break
            self._draw_tracks(frame_raw, t, tracks, vis, n_pts)
            self._draw_hud(frame_raw, t, vis[t], split_idx)
            tracks_scaled = tracks.copy()
            tracks_scaled[:,:,0] *= (w_aria/w_orig); tracks_scaled[:,:,1] *= (h_aria/h_orig)
            self._draw_tracks(frame_aria, t, tracks_scaled, vis, n_pts)
            self._draw_hud(frame_aria, t, vis[t], split_idx)
            out_raw.write(frame_raw)
            out_aria.write(frame_aria)
            gif_raw.append(cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB))
            gif_aria.append(cv2.cvtColor(frame_aria, cv2.COLOR_BGR2RGB))
        cap_aria.release(); out_raw.release(); out_aria.release()
        imageio.mimsave(os.path.join(self.mps_path, "aria", "ot_cotracker_video_raw.gif"), gif_raw, fps=self.fps, loop=0)
        imageio.mimsave(os.path.join(self.mps_path, "aria", "ot_cotracker_video_aria.gif"), gif_aria, fps=self.fps, loop=0)

class ObjectTrackingCoTracker:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        self.res = COTRACKER_RES
        self.aria_dir = os.path.join(mps_path, "aria")
        self._load_configs()
        ckpt = os.path.join(Path(__file__).resolve().parent, "checkpoints/scaled_offline.pth")
        self.predictor = ObjectTrackingCoTrackerPredictor(ckpt, res=self.res)

    def _load_configs(self):
        config_path = os.path.join(self.aria_dir, "ot_keypoints_selector.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        cam_config_path = os.path.join(self.aria_dir, "aria_cam_config.json")
        self.fps = 10
        if os.path.exists(cam_config_path):
            with open(cam_config_path, 'r') as f:
                self.fps = json.load(f).get("fps", 10)
        data_root = os.path.join(self.aria_dir, "all_data")
        frame_dirs = sorted([d for d in os.listdir(data_root) if d.isdigit()])
        # 兼容性处理：优先读取 png，否则读取 jpg
        self.img_paths = []
        for d in frame_dirs:
            p = os.path.join(data_root, d, "rgb.png")
            self.img_paths.append(p)

    def _compute_point_stats(self, visibilities: np.ndarray) -> List[Dict]:
        """计算每个追踪点的详细统计信息"""
        T, N = visibilities.shape
        stats = []
        for n in range(N):
            visible_count = int(np.sum(visibilities[:, n] > 0))
            stats.append({
                "point_id": n,
                "visible_frames": visible_count,
                "total_frames": T,
                "visibility_ratio": round(visible_count / T, 4)
            })
        return stats

    def print_professional_report(self, meta, eval_results, point_stats):
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'ARIA COTRACKER COMPREHENSIVE REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        
        # 1. 任务指标
        print(f"║ [TASK METRICS] {'':<43}║")
        print(f"║  - Total Frames    : {len(self.img_paths):<37}║")
        print(f"║  - Throughput      : {meta['throughput_fps']:<10} FPS {'':<22}║")
        print(f"║ {'':<58} ║")

        # 2. 全局稳定性
        g = eval_results["global"]
        status = "STABLE" if g["score"] > 70 else "UNSTABLE"
        print(f"║ [GLOBAL STABILITY] {'':<39}║")
        print(f"║  - Reliability     : {g['score']:<10} ({status}) {'':<17}║")
        print(f"║  - Jitter Ratio    : {g['jitter']:.2%} {'':<31}║")
        print(f"║ {'':<58} ║")

        # 3. 分阶段稳定性 (核心新增)
        print(f"║ [PHASE-WISE QUALITY] {'':<37}║")
        for s in eval_results["stages"]:
            s_info = f"{s['name'][:5]}[{s['range'][0]}-{s['range'][1]}]"
            score_str = f"Score: {s['score']:<6} | Jitter: {s['jitter']:.2%}"
            print(f"║  - {s_info:<15}: {score_str:<32} ║")
        print(f"║ {'':<58} ║")

        # 4. 逐点可见性
        print(f"║ [POINT VISIBILITY] {'':<39}║")
        for s in point_stats:
            p_str = f"Pt {s['point_id']}: {s['visible_frames']}/{s['total_frames']} ({s['visibility_ratio']*100:.1f}%)"
            print(f"║  - {p_str:<47} ║")
        print("╚" + "═" * 60 + "╝\n")

    def run(self):
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'OBJECT TRACKING COTRACKER PIPELINE START':^60}║")
        print("╠" + "═" * 60 + "╣")
        
        # 1. 执行追踪
        tracks, vis, meta = self.predictor.run_segmented_tracking(
            self.img_paths, 
            self.config["reference_frame_idx"],
            self.config["split_frame_idx"],
            self.config["keypoints_2d"]
        )

        # 2. 统计与评估
        eval_results = ObjectTrackingCoTrackerEvaluator.evaluate_all_phases(tracks, vis, self.config["reference_frame_idx"], self.config["split_frame_idx"])
        point_stats = self._compute_point_stats(vis)

        # 3. 汇总并保存
        res_save = {
            "config": {
                "cotracker_res": self.res,
                "reference_idx": self.config["reference_frame_idx"],
                "split_idx": self.config["split_frame_idx"]
            },
            "metadata": meta,
            "stability_analysis": eval_results, # 包含全局和阶段数据
            "point_statistics": point_stats,
            "tracks": tracks.tolist(),
            "visibilities": vis.tolist()
        }
        
        save_path = os.path.join(self.aria_dir, "ot_cotracker_results.json")
        with open(save_path, 'w') as f:
            json.dump(res_save, f, indent=4)

        # 4. 打印报告
        self.print_professional_report(meta, eval_results, point_stats)

        # 5. 生成热图 [新并入功能]
        heatmap_gen = ObjectTrackingHeatmapGenerator(sigma=HEATMAP_SIGMA)
        heatmap_gen.run(tracks, vis, self.img_paths, self.aria_dir, self.fps)

        # 6. 可视化
        viz = ObjectTrackingCoTrackerVisualizer(self.mps_path, self.fps)
        viz.render(self.img_paths, tracks, vis, self.config["split_frame_idx"])

        print(f"║ [Status] Results saved to: {save_path} ║")
        print("╚" + "═" * 60 + "╝\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()
    
    ot_cotracker = ObjectTrackingCoTracker(args.mps_path)
    ot_cotracker.run()

# conda activate aria
# cd src
# python -m object_tracking.ObjectTrackingCoTracker --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_5_vrs/" 