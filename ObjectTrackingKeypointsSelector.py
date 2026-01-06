# -*- coding: utf-8 -*-
# @FileName: ObjectTrackingKeypointsSelector.py

import os
import json
import cv2
import time
import argparse
import numpy as np
import gradio as gr
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ObjectTrackingKeypointsSelectorConfig:
    reference_frame_idx: int = 0
    split_frame_idx: int = 0
    keypoints_2d: List[Tuple[int, int]] = field(default_factory=list)
    mps_path: str = ""

class ObjectTrackingKeypointsSelector:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        self.config = ObjectTrackingKeypointsSelectorConfig(mps_path=mps_path)
        
        # 1. 自动加载 Phase 结果
        self._load_phase_info()
        
        # 2. UI 状态
        self.points = []
        self.current_img_rgb = None
        self.confirmed = False
        
        # 颜色循环 (专业调色盘: 蓝, 绿, 红, 黄, 紫, 深蓝)
        self.colors = [
            (52, 152, 219), (46, 204, 113), (231, 76, 60), 
            (241, 196, 15), (155, 89, 182), (52, 73, 94)
        ]

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

    def _draw_points(self, img):
        if img is None: return None
        img_draw = img.copy()
        for i, p in enumerate(self.points):
            color = self.colors[i % len(self.colors)]
            # 绘制带白色外圈的发光点
            cv2.circle(img_draw, p, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img_draw, p, 7, (255, 255, 255), 1, cv2.LINE_AA)
            # 绘制编号
            cv2.putText(img_draw, str(i), (p[0]+15, p[1]-15), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv2.LINE_AA)
        return img_draw

    def _get_ref_frame(self):
        idx = self.config.reference_frame_idx
        self.current_img_rgb = self._load_image_from_file(idx)
        return self._draw_points(self.current_img_rgb)

    def _on_click(self, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        self.points.append((x, y))
        return self._draw_points(self.current_img_rgb), f"Point Added: [{x}, {y}]"

    def _undo(self):
        if self.points: self.points.pop()
        return self._draw_points(self.current_img_rgb), "Undo successful."

    def _clear(self):
        self.points = []
        return self.current_img_rgb, "All points cleared."

    def _confirm(self):
        if len(self.points) < 3:
            return "Error: Annotation requires at least 3 points."
        self.config.keypoints_2d = self.points
        self.confirmed = True
        return "SUCCESS: Closing UI and saving results..."

    def launch_ui(self):
        theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")
        
        with gr.Blocks(theme=theme, title="Keypoints Selector") as demo:
            gr.Markdown(f"""
            # 🛠️ Keypoints Selector
            **Target Frame:** `aria/all_data/{self.config.reference_frame_idx:05d}/rgb.png`
            """)
            
            with gr.Row():
                with gr.Column(scale=4):
                    img_display = gr.Image(
                        value=self._get_ref_frame(),
                        label="Click on the object keypoints", 
                        interactive=False,
                        height=700
                    )
                    with gr.Row():
                        undo_btn = gr.Button("⏪ Undo", variant="secondary")
                        clear_btn = gr.Button("🗑️ Clear", variant="stop")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Metadata")
                    status_box = gr.Textbox(label="Log", value="Waiting for click...")
                    
                    with gr.Group():
                        gr.Number(value=self.config.reference_frame_idx, label="Reference (nav2grasp)", interactive=False)
                        gr.Number(value=self.config.split_frame_idx, label="Split (grasp2manip)", interactive=False)
                    
                    save_btn = gr.Button("💾 Confirm & Save", variant="primary", size="lg")
            
            img_display.select(self._on_click, None, [img_display, status_box])
            undo_btn.click(self._undo, None, [img_display, status_box])
            clear_btn.click(self._clear, None, [img_display, status_box])
            save_btn.click(self._confirm, None, status_box)
            
        demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
        print(f"║ [UI] Started. Please open http://localhost:7860")
        
        while not self.confirmed:
            time.sleep(0.2)
        demo.close()
        return self.config
    
    def visualize(self):
        """将选中的关键点绘制在图上并保存为图片文件"""
        # 获取原始图像
        img_rgb = self._load_image_from_file(self.config.reference_frame_idx)
        # 调用现有的绘图逻辑（注意：此时 self.points 已经包含了最终结果）
        annotated_img_rgb = self._draw_points(img_rgb)
        
        # 将 RGB 转回 BGR 格式以便 cv2 保存
        annotated_img_bgr = cv2.cvtColor(annotated_img_rgb, cv2.COLOR_RGB2BGR)
        
        # 确定保存路径
        save_img_path = os.path.join(self.mps_path, "aria", "ot_keypoints_selector_MANUAL.png")
        
        # 执行保存
        cv2.imwrite(save_img_path, annotated_img_bgr)
        return save_img_path
       
    def export_to_reference(self):
        """将结果文件和原始图片复制到上级目录的 DIFT_reference 文件夹"""
        # 1. 确定并创建目标文件夹
        parent_dir = Path(self.mps_path).parent
        target_dir = parent_dir / "DIFT_reference"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 定义源文件路径
        src_json = os.path.join(self.mps_path, "aria", "ot_keypoints_selector.json")
        src_viz_img = os.path.join(self.mps_path, "aria", "ot_keypoints_selector_MANUAL.png")
        src_raw_img = os.path.join(self.mps_path, "aria", "all_data", f"{self.config.reference_frame_idx:05d}", "rgb.png")
        
        # 3. 执行复制 (使用 shutil.copy2 以保留元数据)
        try:
            if os.path.exists(src_json):
                shutil.copy2(src_json, target_dir / "ot_keypoints_selector.json")
            
            if os.path.exists(src_viz_img):
                shutil.copy2(src_viz_img, target_dir / "ot_keypoints_selector_MANUAL.png")
                
            if os.path.exists(src_raw_img):
                # 原始图片复制过去也叫 rgb.png
                shutil.copy2(src_raw_img, target_dir / "rgb.png")
                
            print(f"║ [Export] Files successfully copied to: {target_dir}")
        except Exception as e:
            print(f"║ [Export] Error during copying: {e}")

    def save_results(self):
        save_path = os.path.join(self.mps_path, "aria", "ot_keypoints_selector.json")
        
        data = {
            'method': "MANUAL",
            "reference_frame_idx": self.config.reference_frame_idx,
            "split_frame_idx": self.config.split_frame_idx,
            "num_points": len(self.config.keypoints_2d),
            "keypoints_2d": self.config.keypoints_2d,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

        img_save_path = self.visualize()
        self.export_to_reference()

        # 终端专业报告
        print("\n" + "╔" + "═" * 60 + "╗")
        print(f"║{'KEYPOINTS SELECTION REPORT':^60}║")
        print("╠" + "═" * 60 + "╣")
        print(f"║  - JSON Saved To  : {save_path:<36} ║")
        print(f"║  - Image Saved To : {os.path.basename(img_save_path):<36} ║")
        print(f"║  - Reference Index: {self.config.reference_frame_idx:<36} ║")
        print(f"║  - Total Points   : {len(self.config.keypoints_2d):<36} ║")
        print("╚" + "═" * 60 + "╝\n")
    
    def run(self):
        self.launch_ui()
        self.save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True, help="Path to the MPS directory")
    args = parser.parse_args()

    ot_kpts_selector = ObjectTrackingKeypointsSelector(args.mps_path)
    ot_kpts_selector.run()

# conda activate aria
# cd src
# python -m object_tracking.ObjectTrackingKeypointsSelector --mps_path "./data/open_cabinet_0/mps_open_cabinet_0_5_vrs/" 