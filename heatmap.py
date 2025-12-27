# -*- coding: utf-8 -*-
import os
import cv2
import json
import numpy as np
import imageio
from tqdm import tqdm
import argparse

class HeatmapProcessor:
    def __init__(self, mps_path, sigma=15, res_h=None, res_w=None):
        """
        :param mps_path: 数据根目录
        :param sigma: 高斯核的标准差，决定了点的大小
        :param res_h: 热图高度（若不指定，则从JSON或第一帧读取）
        :param res_w: 热图宽度
        """
        self.mps_path = mps_path
        self.sigma = sigma
        self.heatmap_root = os.path.join(mps_path, "heatmap")
        os.makedirs(self.heatmap_root, exist_ok=True)
        
        # 加载 CoTracker 结果
        json_path = os.path.join(mps_path, "cotracker", "ot_cotracker_results.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到结果文件: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.tracks = np.array(data["tracks_2d"])  # [T, N, 2]
            self.visibilities = np.array(data["visibilities"])  # [T, N]
        
        self.num_frames = self.tracks.shape[0]
        self.num_points = self.tracks.shape[1]
        
        # 确定分辨率
        if res_h is None or res_w is None:
            # 默认假设是原始尺寸，尝试从视频中获取或设定默认
            self.h, self.w = 540, 640 # 这是一个默认值，实际会根据第一帧坐标动态调整或手动指定
            # 自动探测分辨率：寻找坐标的最大值
            max_x = np.max(self.tracks[:,:,0])
            max_y = np.max(self.tracks[:,:,1])
            self.w = int(max(self.w, max_x + 50))
            self.h = int(max(self.h, max_y + 50))
        else:
            self.h, self.w = res_h, res_w

    def _generate_gaussian_heatmap(self, points, vis):
        """
        生成单帧的多点融合高斯热图
        """
        # 创建全黑图
        heatmap = np.zeros((self.h, self.w), dtype=np.float32)
        
        for i in range(self.num_points):
            if vis[i] == 0:
                continue
                
            x, y = points[i]
            
            # 1. 快速方法：绘制一个白点，然后高斯模糊
            # 2. 精确方法：计算每个像素的指数分布
            # 这里采用性能较好且效果美观的方法：计算局部 Patch
            
            kernel_size = int(self.sigma * 4 + 1)
            # 生成 1D 高斯
            grid = np.arange(kernel_size) - (kernel_size // 2)
            gaussian_1d = np.exp(-grid**2 / (2 * self.sigma**2))
            # 生成 2D 高斯核
            kernel_2d = np.outer(gaussian_1d, gaussian_1d)
            
            # 计算写入边界
            ix, iy = int(round(x)), int(round(y))
            x1, x2 = ix - kernel_size // 2, ix + kernel_size // 2 + 1
            y1, y2 = iy - kernel_size // 2, iy + kernel_size // 2 + 1
            
            # 处理越界
            kx1, kx2 = max(0, -x1), min(kernel_size, self.w - x1)
            ky1, ky2 = max(0, -y1), min(kernel_size, self.h - y1)
            
            x1, x2 = max(0, x1), min(self.w, x2)
            y1, y2 = max(0, y1), min(self.h, y2)
            
            if x2 > x1 and y2 > y1:
                # 叠加到热图上，使用 np.maximum 保持多点重合时不溢出
                heatmap[y1:y2, x1:x2] = np.maximum(heatmap[y1:y2, x1:x2], kernel_2d[ky1:ky2, kx1:kx2])
        
        # 归一化到 0-255
        heatmap = (heatmap * 255).astype(np.uint8)
        return heatmap

    def run(self, fps=20):
        video_path = os.path.join(self.heatmap_root, "heatmap_video_vis.mp4")
        gif_path = os.path.join(self.heatmap_root, "heatmap_video_vis.gif")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, fps, (self.w, self.h), False) # False 表示灰度
        
        frames_for_gif = []
        
        print(f"开始生成热图，共 {self.num_frames} 帧...")
        
        for t in tqdm(range(self.num_frames)):
            # 1. 生成热图
            heatmap = self._generate_gaussian_heatmap(self.tracks[t], self.visibilities[t])
            
            # 2. 按照路径保存： mps_path/heatmap/XXXXX/heatmap.png
            frame_dir = os.path.join(self.heatmap_root, f"{t:05d}")
            os.makedirs(frame_dir, exist_ok=True)
            cv2.imwrite(os.path.join(frame_dir, "heatmap.png"), heatmap)
            
            # 3. 收集用于视频和GIF
            out_video.write(heatmap)
            
            # GIF 抽帧或者压缩（防止内存溢出）
            if t % 1 == 0: # 可以改为 t % 2 如果视频太长
                frames_for_gif.append(heatmap)
                
        out_video.release()
        
        print(f"正在保存 GIF: {gif_path}...")
        imageio.mimsave(gif_path, frames_for_gif, fps=fps, loop=0)
        
        print(f"任务完成！所有热图保存在: {self.heatmap_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the mps data")
    parser.add_argument("--sigma", type=int, default=8, help="Gaussian sigma (size of the spot)")
    parser.add_argument("--fps", type=int, default=10, help="Output FPS")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()
    
    processor = HeatmapProcessor(
        mps_path=args.mps_path, 
        sigma=args.sigma,
        res_h=args.height,
        res_w=args.width
    )
    processor.run(fps=args.fps)

    # python heatmap.py --mps_path "../data/mps_open_cabinet_5_vrs/"