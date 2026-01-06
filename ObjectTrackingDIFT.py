# -*- coding: utf-8 -*-
import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 特征提取核心参数
TIMESTEP = 20         # 0-200, 越大语义越强(防乱序), 越小纹理越强(对齐准)
IMG_SIZE = 1024         # 输入模型的分辨率
TARGET_FEATURE_SIZE = 256  # 融合特征的目标尺寸，建议设为 256 或 512 提升精度

# 2. 层选择 (SD v1.5 UNet up_blocks 索引为 0,1,2,3)
# index 1 (64x64级别): 语义强，确定点在大致位置
# index 2 (128x128级别): 结构强，确定点的精确边缘
LAYER_1_IDX = 1        # 通常 64x64 级别
LAYER_2_IDX = 2        # 通常 128x128 级别
LAYER_1_WEIGHT = 1.0   # 语义层权重
LAYER_2_WEIGHT = 1.0   # 结构层权重

# 3. 增强/隐藏参数
# ENSEMBLE_SIZE: 集成次数。多次运行取平均特征。1为不开启，建议2-4可极大增加稳定性，但是速度变慢。
ENSEMBLE_SIZE = 2      
# PROMPT: 引导词。如果为空，则使用无偏见提取。若有特定物体（如"handle"），可尝试加入。
PROMPT = "a door handle on a wooden cabinet"            
# CONTEXT_WINDOW: 领域范围。3表示考虑 3x3 邻域特征，缓解重复纹理歧义。
CONTEXT_WINDOW = 5     
# LOCAL_SEARCH_RADIUS: 局部搜索范围(0-1)。0为全图搜索。0.2表示只在源点周围20%区域内找。
LOCAL_SEARCH_RADIUS = 0.2 
# SORT_BY_X: 强制结果按X轴排序。横向排列的把手必须开启。
SORT_BY_X = True 

@dataclass
class ObjectTrackingDIFTConfig:
    reference_frame_idx: int = 0
    split_frame_idx: int = 0
    keypoints_2d: List[Tuple[int, int]] = field(default_factory=list)
    DIFT_config: dict = field(default_factory=dict)
    mps_path: str = ""

class ObjectTrackingDIFT:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        self.config = ObjectTrackingDIFTConfig(mps_path=mps_path)
        self._load_phase_info()
        
        print(f"[*] Initializing Package: {MODEL} | Device: {DEVICE}")
        self.unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16).to(DEVICE)
        self.vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float16).to(DEVICE)
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float16).to(DEVICE)
    
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _load_phase_info(self):
        """从 aria_phases_results.json 提取关键帧索引"""
        phase_path = os.path.join(self.mps_path, "aria", "aria_phases_results.json")
        if not os.path.exists(phase_path):
            print(f"║ [Warning] Phase results not found at {phase_path}")
            return

        with open(phase_path, 'r') as f:
            data = json.load(f)
            indices = data.get("key_indices", {})
            self.config.reference_frame_idx = indices.get("nav2grasp_idx", 0)
            self.config.split_frame_idx = indices.get("grasp2manip_idx", 0)

    def _load_image_from_file(self, idx: int):
        """直接从 all_data 文件夹读取图片"""
        # 搜索路径：aria/all_data/00XXX/rgb.png
        base_path = os.path.join(self.mps_path, "aria", "all_data", f"{idx:05d}")
        img_path = os.path.join(base_path, "rgb.png")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"║ [Error] Could not read image at {img_path}")
            # 返回一个黑色占位图
            return np.zeros((640, 640, 3), dtype=np.uint8)
        
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    @torch.no_grad()
    def _forward_pass(self, img_tensor):
        """单次前向传播，通过 Hook 捕获指定层的特征"""
        dtype = torch.float16
        
        # 1. VAE 采样获取 Latents
        latents = self.vae.encode(img_tensor).latent_dist.sample() * 0.18215
        
        # 2. 将 PROMPT 转换为文本嵌入
        # 如果 PROMPT 为空，则编码空字符串 "" 以保持模型一致性（比全零张量更好）
        text_inputs = self.tokenizer(
            PROMPT if PROMPT else "", 
            padding="max_length", 
            max_length=self.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        # 获取 CLIP 的输出 [1, 77, 768]
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(DEVICE))[0]


         # 3. 准备时间步
        t_tensor = torch.tensor([TIMESTEP], device=DEVICE)
        
        collected = []
        def hook_fn(module, input, output):
            collected.append(output[0] if isinstance(output, tuple) else output)

        # 4. 注册指定层的 Hook
        h1 = self.unet.up_blocks[LAYER_1_IDX].resnets[1].register_forward_hook(hook_fn)
        h2 = self.unet.up_blocks[LAYER_2_IDX].resnets[1].register_forward_hook(hook_fn)
        
        self.unet(latents, t_tensor, encoder_hidden_states=prompt_embeds)
        
        h1.remove(); h2.remove()
        
        # 5. 提取、归一化并对齐分辨率
        # 结果存为 float32 防止后续计算溢出
        f1 = F.normalize(collected[0].to(torch.float32), dim=1)
        f2 = F.normalize(collected[1].to(torch.float32), dim=1)
        
        f1_up = F.interpolate(f1, size=(TARGET_FEATURE_SIZE, TARGET_FEATURE_SIZE), mode='bilinear')
        f2_up = F.interpolate(f2, size=(TARGET_FEATURE_SIZE, TARGET_FEATURE_SIZE), mode='bilinear')
        
        # 返回加权融合特征
        return (f1_up * LAYER_1_WEIGHT), (f2_up * LAYER_2_WEIGHT)

    def extract_multi_scale_features(self, img_pil):
        """带有 ENSEMBLE 机制的多尺度特征提取"""
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE, dtype=torch.float16)
        
        sum_f1 = None
        sum_f2 = None
        
        # --- ENSEMBLE LOOP START ---
        for i in range(ENSEMBLE_SIZE):
            if ENSEMBLE_SIZE > 1:
                print(f"    > Pass {i+1}/{ENSEMBLE_SIZE}...")
            
            f1, f2 = self._forward_pass(img_tensor)
            
            if sum_f1 is None:
                sum_f1, sum_f2 = f1, f2
            else:
                sum_f1 += f1
                sum_f2 += f2
        # --- ENSEMBLE LOOP END ---

        # 取均值并最终归一化
        final_f1 = F.normalize(sum_f1 / ENSEMBLE_SIZE, dim=1)
        final_f2 = F.normalize(sum_f2 / ENSEMBLE_SIZE, dim=1)
        
        # 最终特征：[1, C1+C2, H, W]
        return torch.cat([final_f1, final_f2], dim=1)

    def run(self):
        DIFT_reference = os.path.join(str(Path(self.mps_path).parent), 'DIFT_reference')
        src_img_path = os.path.join(DIFT_reference, 'rgb.png')
        src_json_path = os.path.join(DIFT_reference, 'ot_keypoints_selector.json')
        if not os.path.exists(src_img_path):
            raise FileNotFoundError(f"❌ 找不到参考图片: {src_img_path}。请先运行手动标注脚本。")
        if not os.path.exists(src_json_path):
            raise FileNotFoundError(f"❌ 找不到参考JSON: {src_json_path}。请先运行手动标注脚本。")
        
        tgt_img_path = os.path.join(self.mps_path, "aria", "all_data", f"{self.config.reference_frame_idx:05d}", "rgb.png")
        save_json_path = os.path.join(self.mps_path, 'aria', 'ot_keypoints_selector.json')
        save_vis_path = os.path.join(self.mps_path, 'aria', 'ot_keypoints_selector_DIFT.png')

        src_pil = Image.open(src_img_path).convert('RGB')
        tgt_pil = Image.open(tgt_img_path).convert('RGB')
        orig_w, orig_h = src_pil.size
        
        with open(src_json_path, 'r') as f:
            src_data = json.load(f)
        src_points = src_data["keypoints_2d"]
        
        print(f"[*] Processing {len(src_points)} points. Ensemble: {ENSEMBLE_SIZE}, Window: {CONTEXT_WINDOW}")
        
        # 提取两图特征
        print("[*] Feature Extraction: Source...")
        src_feat = self.extract_multi_scale_features(src_pil) 
        print("[*] Feature Extraction: Target...")
        tgt_feat = self.extract_multi_scale_features(tgt_pil)
        
        B, C, H_f, W_f = src_feat.shape
        print(f"[*] Final Feature map size: {H_f}x{W_f}")

        pred_points = []
        for i, (x, y) in enumerate(src_points):
            # A. 特征图坐标映射
            xf = int(x * W_f / orig_w)
            yf = int(y * H_f / orig_h)
            
            # B. 上下文采样 (Context Window)
            r = CONTEXT_WINDOW // 2
            y_min, y_max = max(0, yf-r), min(H_f, yf+r+1)
            x_min, x_max = max(0, xf-r), min(W_f, xf+r+1)
            src_vec = src_feat[0, :, y_min:y_max, x_min:x_max].mean(dim=(1, 2)).view(1, C, 1, 1)
            
            # C. 全图计算相似度并上采样回 640 空间
            cos_sim = (src_vec * tgt_feat).sum(dim=1) 
            cos_sim_up = F.interpolate(cos_sim.unsqueeze(0), size=(orig_h, orig_w), mode='bilinear').squeeze()
            
            # D. 局部搜索策略 (防止点乱飞)
            if LOCAL_SEARCH_RADIUS > 0:
                mask = torch.zeros_like(cos_sim_up)
                h_r, w_r = int(orig_h * LOCAL_SEARCH_RADIUS), int(orig_w * LOCAL_SEARCH_RADIUS)
                mask[max(0, y-h_r):min(orig_h, y+h_r), max(0, x-w_r):min(orig_w, x+w_r)] = 1.0
                cos_sim_up *= mask

            # E. 找到最大似然位置
            idx = torch.argmax(cos_sim_up.view(-1))
            ty, tx = divmod(idx.item(), orig_w)
            pred_points.append([int(tx), int(ty)])
            print(f"    - Point {i} mapped: ({x}, {y}) -> ({tx}, {ty})")

        # F. 强制排序处理
        if SORT_BY_X:
            print("[!] Sorting results by X-axis to maintain order.")
            pred_points.sort(key=lambda p: p[0])

        # 保存 JSON
        self.config.keypoints_2d = pred_points

        json_data = {
            'method': "DIFT",
            "reference_frame_idx": self.config.reference_frame_idx,
            "split_frame_idx": self.config.split_frame_idx,
            "num_points": len(self.config.keypoints_2d),
            "keypoints_2d": self.config.keypoints_2d,
            "DIFT_config":{ "MODEL": MODEL,
                            "DEVICE": DEVICE,
                            "TIMESTEP": TIMESTEP,
                            "IMG_SIZE": IMG_SIZE,
                            "TARGET_FEATURE_SIZE": TARGET_FEATURE_SIZE,
                            "LAYER_1_IDX": LAYER_1_IDX,
                            "LAYER_2_IDX": LAYER_2_IDX,
                            "LAYER_1_WEIGHT": LAYER_1_WEIGHT,
                            "LAYER_2_WEIGHT": LAYER_2_WEIGHT,
                            "ENSEMBLE_SIZE": ENSEMBLE_SIZE, 
                            "PROMPT": PROMPT,
                            "CONTEXT_WINDOW": CONTEXT_WINDOW,
                            "LOCAL_SEARCH_RADIUS": LOCAL_SEARCH_RADIUS,
                            "SORT_BY_X": SORT_BY_X
                            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(save_json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        
        self.visualize(src_img_path, tgt_img_path, src_points, pred_points, save_vis_path)

    def visualize(self, src_path, tgt_path, src_pts, tgt_pts, save_path):
        img1 = cv2.imread(src_path)
        img2 = cv2.imread(tgt_path)
        
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(src_pts, tgt_pts)):
            color = colors[i % len(colors)]
            cv2.circle(img1, (int(x1), int(y1)), 8, color, -1)
            cv2.circle(img2, (int(x2), int(y2)), 8, color, -1)
            cv2.putText(img1, str(i), (int(x1)+10, int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(img2, str(i), (int(x2)+10, int(y2)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        combined = np.hstack((img1, img2))
        cv2.imwrite(save_path, combined)
        print(f"[+] Visualization saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()

    ot_dift = ObjectTrackingDIFT(args.mps_path)
    ot_dift.run()

# conda activate aria
# cd src
# python -m object_tracking.ObjectTrackingDIFT --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_5_vrs/" 