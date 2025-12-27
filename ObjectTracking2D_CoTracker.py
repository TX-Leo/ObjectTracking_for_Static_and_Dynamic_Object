# -*- coding: utf-8 -*-
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
import matplotlib.pyplot as plt
from datetime import datetime
import gc
import argparse

# ==================== 路径修正 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from aria.AriaDataset import AriaDataset
from cotracker.predictor import CoTrackerPredictor

# ==============================================================================
# [Class 1] CoTracker 核心推理类 (支持分段推理)
# ==============================================================================
class AriaCoTracker:
    def __init__(self, checkpoint_path, res=640, device="cuda"):
        self.device = device
        self.res = res
        print(f"[{datetime.now()}] 正在加载 CoTracker 模型: {checkpoint_path}")
        self.model = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)

    def _infer_segment(self, frames, queries):
        """内部函数：执行单段推理并释放内存"""
        video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].to(self.device).float()
        queries_tensor = torch.tensor([queries]).to(self.device).float()
        
        with torch.no_grad():
            pred_tracks, pred_vis = self.model(video_tensor, queries=queries_tensor)
        
        # 拷贝到 CPU
        tracks = pred_tracks[0].cpu().numpy()
        visibilities = pred_vis[0].cpu().numpy()
        
        # 显式清理 GPU 内存
        del video_tensor, queries_tensor, pred_tracks, pred_vis
        torch.cuda.empty_cache()
        gc.collect()
        
        return tracks, visibilities

    def process_video_segmented(self, dataset, ref_idx, split_idx, init_points_raw):
        """分两段执行全序列追踪以节省显存"""
        img_h, img_w = dataset[0].cam.rgb.shape[:2]
        scale_w, scale_h = self.res / img_w, self.res / img_h
        total_frames = len(dataset)
        
        start_time_all = time.perf_counter()

        # --- 第一段: 0 -> split_idx ---
        print(f"[{datetime.now()}] 推理第一段: 0 -> {split_idx} (Ref: {ref_idx})")
        frames_1 = [cv2.resize(dataset[i].cam.rgb, (self.res, self.res)) for i in range(split_idx + 1)]
        queries_1 = [[float(ref_idx), p[0] * scale_w, p[1] * scale_h] for p in init_points_raw]
        tracks_1, vis_1 = self._infer_segment(frames_1, queries_1)
        del frames_1 # 释放内存

        # --- 衔接点处理 ---
        # 获取第一段在 split_idx 处的坐标作为第二段的起点
        last_pts_scaled = tracks_1[split_idx] # 这里已经是 scaled 坐标了

        # --- 第二段: split_idx -> end ---
        print(f"[{datetime.now()}] 推理第二段: {split_idx} -> {total_frames-1}")
        frames_2 = [cv2.resize(dataset[i].cam.rgb, (self.res, self.res)) for i in range(split_idx, total_frames)]
        # 在第二段中，起始帧就是查询帧，索引为 0
        queries_2 = [[0.0, pt[0], pt[1]] for pt in last_pts_scaled]
        tracks_2, vis_2 = self._infer_segment(frames_2, queries_2)
        del frames_2 # 释放内存

        end_time_all = time.perf_counter()

        # --- 合并结果 ---
        # tracks_1 形状 [split_idx + 1, N, 2]
        # tracks_2 形状 [total - split_idx, N, 2]
        # 合并时去掉 tracks_2 的第一帧（它是重复的）
        full_tracks = np.concatenate([tracks_1, tracks_2[1:]], axis=0)
        full_vis = np.concatenate([vis_1, vis_2[1:]], axis=0)

        # 还原到原始图像分辨率
        full_tracks[:, :, 0] /= scale_w
        full_tracks[:, :, 1] /= scale_h

        # 边界后处理
        for t in range(total_frames):
            for n in range(full_tracks.shape[1]):
                u, v = full_tracks[t, n]
                if not full_vis[t, n] or u < 0 or u >= img_w or v < 0 or v >= img_h:
                    full_tracks[t, n] = [0.0, 0.0]
                    full_vis[t, n] = 0

        perf_meta = {
            "total_time": end_time_all - start_time_all,
            "latency": (end_time_all - start_time_all) / total_frames,
            "fps": total_frames / (end_time_all - start_time_all)
        }
        print(f"分段推理完成！Running FPS: {perf_meta['fps']:.1f}")
        return full_tracks, full_vis, perf_meta

# ==============================================================================
# [Class 2] 刚性约束评估类 (保持不变)
# ==============================================================================
class RigidityEvaluator:
    @staticmethod
    def evaluate(tracks, visibilities):
        T, N, _ = tracks.shape
        if N < 2: return {"score": 0, "msg": "Insufficient points"}
        pair_distances = []
        for i in range(N):
            for j in range(i + 1, N):
                mask = (visibilities[:, i] > 0) & (visibilities[:, j] > 0)
                if mask.sum() < 10: continue # 至少10帧有效数据
                d = np.linalg.norm(tracks[mask, i] - tracks[mask, j], axis=1)
                pair_distances.append({
                    "pair": [i, j],
                    "mean_dist_px": float(np.mean(d)),
                    "cv": float(np.std(d) / (np.mean(d) + 1e-6))
                })
        avg_cv = np.mean([p["cv"] for p in pair_distances]) if pair_distances else 1.0
        score = max(0, 100 * (1 - avg_cv * 10)) # 惩罚系数调大
        return {"rigidity_score": score, "avg_cv": avg_cv, "details": pair_distances}

# ==============================================================================
# [Class 3] 酷炫可视化类 (优化渲染)
# ==============================================================================
import cv2
import numpy as np
import math
import os
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

class CoTrackerVisualizer:
    # ==================== 视觉参数宏定义 (封装在类内) ====================
    FONT = cv2.FONT_HERSHEY_DUPLEX
    AA = cv2.LINE_AA
    
    C_GOLD = (0, 180, 255)    # 科技金 (BGR)
    C_CYAN = (255, 255, 0)    # 青色
    C_GREEN = (0, 255, 120)   # 荧光绿
    C_RED = (80, 80, 255)     # 警示红
    C_BG = (25, 20, 15)       # 面板背景色
    
    HUD_ALPHA = 0.4           # 透明度 (更透明)
    TRAIL_LEN = 25            # 尾迹长度
    TRAIL_MAX_THICK = 3       # 尾迹粗细
    PT_RADIUS = 5             # 十字准星圆环半径
    MAX_SLOTS = 5             # 右上角固定 5 个槽位

    def __init__(self, raw_frames, aria_video_path, save_dir, fps=20):
        """
        :param raw_frames: List[np.ndarray] 来自 dataset.rgb 的原始帧
        :param aria_video_path: str 路径指向 'mps_path/aria/aria_video_vis.mp4'
        :param save_dir: 保存目录
        :param fps: 输出帧率
        """
        self.raw_frames = raw_frames
        self.aria_video_path = aria_video_path
        self.save_dir = save_dir
        self.fps = fps
        self.cmap = plt.get_cmap('hsv')

    # ==================== 内部功能函数 ====================
    def _draw_hud(self, img, t, num_pts, visibilities_at_t, split_idx):
        """绘制右上角 HUD 面板"""
        h_img, w_img = img.shape[:2]
        p_w, p_h = 220, 150
        tx, ty = w_img - p_w - 20, 40
        
        # 背景
        overlay = img.copy()
        cv2.rectangle(overlay, (tx, ty), (tx + p_w, ty + p_h), self.C_BG, -1)
        cv2.addWeighted(overlay, self.HUD_ALPHA, img, 1 - self.HUD_ALPHA, 0, img)
        
        # 装饰角
        l = 10
        cv2.line(img, (tx, ty), (tx+l, ty), self.C_GOLD, 1, self.AA)
        cv2.line(img, (tx, ty), (tx, ty+l), self.C_GOLD, 1, self.AA)
        cv2.line(img, (tx+p_w, ty+p_h), (tx+p_w-l, ty+p_h), self.C_GOLD, 1, self.AA)
        cv2.line(img, (tx+p_w, ty+p_h), (tx+p_w, ty+p_h-l), self.C_GOLD, 1, self.AA)

        # 文本信息
        cv2.putText(img, "OBJECT TRACKING", (tx+12, ty+25), self.FONT, 0.5, (255,255,255), 1, self.AA)
        cv2.line(img, (tx+10, ty+35), (tx+p_w-10, ty+35), (100,100,100), 1, self.AA)
        cv2.putText(img, f"FRAME: {t:03d}", (tx+12, ty+55), self.FONT, 0.45, (200,200,200), 1, self.AA)

        # 固定 5 位横排点阵槽
        for n in range(self.MAX_SLOTS):
            x, y = tx + 12 + n * 18, ty + 75
            if n < num_pts:
                color = self.C_GREEN if visibilities_at_t[n] > 0 else self.C_RED
                cv2.rectangle(img, (x, y), (x+10, y+10), color, 1, self.AA)
                if visibilities_at_t[n] > 0:
                    cv2.rectangle(img, (x+2, y+2), (x+8, y+8), color, -1)
                else: # 丢失画叉
                    cv2.line(img, (x+2, y+2), (x+8, y+8), color, 1, self.AA)
                    cv2.line(img, (x+8, y+2), (x+2, y+8), color, 1, self.AA)
            else:
                cv2.rectangle(img, (x, y), (x+10, y+10), (60,60,60), 1, self.AA)

        # 状态文字 (呼吸感)
        is_moving = t > split_idx
        s_color = self.C_CYAN if is_moving else self.C_GREEN
        s_text = "MOTION" if is_moving else "STATIC"
        breath = (math.sin(t * 0.2) + 1) / 2
        alpha_color = tuple([int(c * (0.4 + 0.6 * breath)) for c in s_color])
        cv2.putText(img, f"[ >> {s_text} << ]", (tx + 35, ty + p_h - 20), self.FONT, 0.5, alpha_color, 1, self.AA)

    def _draw_tracking_features(self, img, t, tracks, visibilities, num_pts):
        """在给定图像上绘制所有追踪点和尾迹"""
        for n in range(num_pts):
            if visibilities[t, n] == 0: continue
            
            # 颜色计算
            raw_color = self.cmap(n/num_pts)[:3][::-1]
            color = tuple([int(c*255) for c in raw_color])
            
            # 1. 绘制高级尾迹
            for i in range(max(1, t - self.TRAIL_LEN), t + 1):
                if visibilities[i-1, n] and visibilities[i, n]:
                    pt1 = (int(tracks[i-1, n, 0]), int(tracks[i-1, n, 1]))
                    pt2 = (int(tracks[i, n, 0]), int(tracks[i, n, 1]))
                    progress = (i - (t - self.TRAIL_LEN)) / self.TRAIL_LEN
                    thick = int(1 + (self.TRAIL_MAX_THICK - 1) * progress)
                    fade_color = tuple([int(c * (0.3 + 0.7 * progress)) for c in color])
                    cv2.line(img, pt1, pt2, fade_color, thick, self.AA)
            
            # 2. 绘制锋利十字准星 (无白点)
            ix, iy = int(tracks[t, n, 0]), int(tracks[t, n, 1])
            # cv2.circle(img, (ix, iy), self.PT_RADIUS, color, 1, self.AA)
            l = self.PT_RADIUS + 4
            cv2.line(img, (ix-l, iy), (ix+l, iy), color, 1, self.AA)
            cv2.line(img, (ix, iy-l), (ix, iy+l), color, 1, self.AA)

    # ==================== 核心渲染函数 ====================
    def render(self, tracks, visibilities, split_idx):
        """
        核心逻辑：
        1. 循环遍历所有帧
        2. 每帧提取 raw_frame 和 aria_video_frame
        3. 对两张图进行相同的绘图操作
        4. 分别保存到两个视频和两个GIF
        """
        # --- A. 准备视频输入 ---
        cap_aria = cv2.VideoCapture(self.aria_video_path)
        w_aria = int(cap_aria.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_aria = int(cap_aria.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # --- B. 准备视频输出 ---
        h_raw, w_raw = self.raw_frames[0].shape[:2]
        names = ["ot_cotracker_video_vis_clean", "ot_cotracker_vis_aria"]
        writers = [
            cv2.VideoWriter(os.path.join(self.save_dir, f"{names[0]}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w_raw, h_raw)),
            cv2.VideoWriter(os.path.join(self.save_dir, f"{names[1]}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w_aria, h_aria))
        ]
        
        gif_collectors = [[], []]
        num_pts = tracks.shape[1]
        num_frames = min(len(self.raw_frames), tracks.shape[0])

        print(f"[{os.path.basename(self.aria_video_path)}] 开始双背景渲染...")

        for t in tqdm(range(num_frames)):
            # 1. 获取背景图
            # 背景1: Raw Dataset Frame
            img_raw = self.raw_frames[t]
            
            # 背景2: Aria Video Frame
            ret, img_aria = cap_aria.read()
            if not ret: break

            # 2. 处理坐标缩放 (防止 Aria 视频和 Raw 图像分辨率不一致)
            tracks_aria = tracks.copy()
            if (w_aria != w_raw) or (h_aria != h_raw):
                tracks_aria[:, :, 0] *= (w_aria / w_raw)
                tracks_aria[:, :, 1] *= (h_aria / h_raw)

            # 3. 在 Raw 背景上绘制
            self._draw_tracking_features(img_raw, t, tracks, visibilities, num_pts)
            self._draw_hud(img_raw, t, num_pts, visibilities[t], split_idx)
            
            # 4. 在 Aria 背景上绘制
            self._draw_tracking_features(img_aria, t, tracks_aria, visibilities, num_pts)
            self._draw_hud(img_aria, t, num_pts, visibilities[t], split_idx)

            # 5. 保存结果
            writers[0].write(img_raw)
            writers[1].write(img_aria)
            gif_collectors[0].append(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB))
            gif_collectors[1].append(cv2.cvtColor(img_aria, cv2.COLOR_BGR2RGB))

        # --- C. 释放资源与保存 GIF ---
        cap_aria.release()
        for w in writers: w.release()

        for i, name in enumerate(names):
            gif_path = os.path.join(self.save_dir, f"{name}.gif")
            print(f"正在保存 GIF: {gif_path}")
            imageio.mimsave(gif_path, gif_collectors[i], fps=self.fps, loop=0)

        print("双背景可视化任务完成！")


# ==============================================================================
# [Class 4] 管理类
# ==============================================================================
class CoTrackerManager:
    def __init__(self, mps_path, res=640):
        self.mps_path = mps_path
        self.save_dir = os.path.join(mps_path, "cotracker")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 加载配置
        config_path = os.path.join(self.save_dir, "handle_selection.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}，请先运行选点脚本。")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # 查找 VRS 文件
        vrs_files = [f for f in os.listdir(mps_path) if f.endswith('.vrs')]
        if not vrs_files:
            raise FileNotFoundError("MPS 路径下未找到 .vrs 文件")
        vrs_path = os.path.join(mps_path, vrs_files[0])

        # 查找 Hand Tracking CSV (如果有)
        hand_csv = os.path.join(mps_path, "hand_tracking/hand_tracking_results.csv")
    
        # 初始化数据集
        self.ds = AriaDataset(mps_path, vrs_path, hand_csv, self.save_dir)
        
        # 初始化追踪器
        checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints/scaled_offline.pth")
        self.tracker = AriaCoTracker(checkpoint, res)
    
    
    def run(self):
        # 1. 准备数据：将 Dataset 转换为 RGB 列表 (新版 Visualizer 的要求)
        print(f"[{datetime.now()}] 正在预加载视频帧...")
        frames_list = [self.ds[i].cam.rgb for i in range(len(self.ds))]

        aria_video_path = os.path.join(self.mps_path, "aria", "aria_video_vis.mp4")

        # 2. 执行分段追踪
        # 从 config 中获取 split_frame_idx
        split_idx = self.config.get("split_idx") or self.config.get("split_frame_idx")
        
        tracks, vis, meta = self.tracker.process_video_segmented(
            self.ds, 
            self.config["reference_frame_idx"], 
            split_idx,
            self.config["handle_points_2d"]
        )

        # 3. 刚性评估
        eval_res = RigidityEvaluator.evaluate(tracks, vis)

        # 4. 保存 JSON 结果
        res_json = {
            "metadata": meta,
            "rigidity_evaluation": eval_res,
            "tracks_2d": tracks.tolist(),
            "visibilities": vis.tolist()
        }
        with open(os.path.join(self.save_dir, "ot_cotracker_results.json"), 'w') as f:
            json.dump(res_json, f, indent=4)

        # 5. 调用新版炫酷渲染器
        print(f"[{datetime.now()}] 开始高质量渲染 (视频+GIF)...")
        vis_tool = CoTrackerVisualizer(
            raw_frames=frames_list, 
            aria_video_path=aria_video_path,
            save_dir=self.save_dir, 
            fps=self.ds.fps
        )
        
        vis_tool.render(
            tracks=tracks, 
            visibilities=vis, 
            split_idx=split_idx
        )

        print(f"[{datetime.now()}] 任务全部完成！输出目录: {self.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    parser.add_argument("--res", type=int, default=640)
    args = parser.parse_args()
    
    CoTrackerManager(args.mps_path, args.res).run()