# -*- coding: utf-8 -*-
# @FileName: handle_selection.py
# @Description: 第一阶段：手动选择参考帧（用于选点）和分界帧（运动起始点）。

import os
import cv2
import sys
import json
import time
import argparse
import numpy as np
import gradio as gr

# ==================== 路径修正 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from aria.AriaDataset import AriaDataset

# ==============================================================================
# [Module] 几何辅助工具
# ==============================================================================

class GeometryHelper:
    @staticmethod
    def rotate_point_vis_to_raw(u_vis, v_vis, vis_w, vis_h):
        """Portrait (显示) -> Landscape (原始)"""
        u_raw = v_vis
        v_raw = vis_w - 1 - u_vis
        return float(u_raw), float(v_raw)

# ==============================================================================
# [Module] 交互式选择器 (支持双帧模式)
# ==============================================================================

class AriaHandleSelector:
    def __init__(self, dataset: AriaDataset):
        self.dataset = dataset
        self.total_frames = len(dataset)
        self.state = {
            "ref_frame_idx": 0,    # 用于标注的清晰帧
            "split_frame_idx": 0,  # 运动开始的临界帧
            "handle_points_vis": [], 
            "confirmed": False
        }
        self.current_img_vis = None

    def _get_frame_data(self, idx):
        self.current_img_vis = cv2.cvtColor(self.dataset[idx].cam.rgb, cv2.COLOR_BGR2RGB)
        # 换帧时不自动清空点，方便用户在不同帧查看
        img_draw = self.current_img_vis.copy()
        for i, p in enumerate(self.state["handle_points_vis"]):
            cv2.circle(img_draw, p, 8, (255, 0, 0), -1)
        return img_draw, f"当前预览帧: {idx}"

    def _on_click(self, evt: gr.SelectData):
        if len(self.state["handle_points_vis"]) >= 3:
            return self.current_img_vis, "已标注3个点，如需更改请刷新页面或切换帧"
        
        x, y = evt.index[0], evt.index[1]
        self.state["handle_points_vis"].append((x, y))
        
        img_draw = self.current_img_vis.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, p in enumerate(self.state["handle_points_vis"]):
            cv2.circle(img_draw, p, 8, colors[i], -1)
        return img_draw, f"已在当前帧标注 {len(self.state['handle_points_vis'])}/3 点"

    def _set_as_ref(self, idx):
        self.state["ref_frame_idx"] = idx
        return f"已设定参考帧（选点帧）为: {idx}"

    def _set_as_split(self, idx):
        self.state["split_frame_idx"] = idx
        return f"已设定分界帧（运动起始）为: {idx}"

    def _confirm(self):
        if len(self.state["handle_points_vis"]) < 3:
            return "错误: 请先标注3个关键点。"
        self.state["confirmed"] = True
        return "配置已保存！请关闭网页并在控制台查看结果。"

    def launch_ui(self):
        with gr.Blocks(title="Aria Handle Selector") as demo:
            gr.Markdown("## 第一阶段：关键点标注与阶段分割 (双帧模式)")
            gr.Markdown("""
            **操作流程：**
            1. 滑动进度条找一帧**把手看得很清楚、没被手遮挡**的静态帧，点击图上 3 个关键点 (L, C, R)，然后点击 **'Set as Ref'**。
            2. 滑动进度条找到**把手即将开始运动**的那一帧，点击 **'Set as Split'**。
            3. 点击 **Confirm** 保存。
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    img_display = gr.Image(label="标注视图", interactive=False)
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(0, self.total_frames-1, value=0, step=1, label="拖动寻找帧")
                    with gr.Row():
                        btn_ref = gr.Button("1. Set as Reference (Points Frame)")
                        btn_split = gr.Button("2. Set as Split (Movement Start)")
                    status_box = gr.Textbox(label="状态", value="请开始标注...")
                    confirm_btn = gr.Button("3. Confirm & Save", variant="primary")
            
            frame_slider.change(self._get_frame_data, frame_slider, [img_display, status_box])
            img_display.select(self._on_click, None, [img_display, status_box])
            btn_ref.click(self._set_as_ref, frame_slider, status_box)
            btn_split.click(self._set_as_split, frame_slider, status_box)
            confirm_btn.click(self._confirm, None, status_box)
            demo.load(lambda: self._get_frame_data(0)[0], None, img_display)
            
        demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
        while not self.state["confirmed"]: time.sleep(0.5)
        demo.close()
        return self.state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps_path", type=str, required=True)
    args = parser.parse_args()

    save_dir = os.path.join(args.mps_path, "cotracker")
    os.makedirs(save_dir, exist_ok=True)
    
    vrs_file = os.path.join(args.mps_path, [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')][0])
    hand_csv = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    dataset = AriaDataset(args.mps_path, vrs_file, hand_csv, save_dir)

    selector = AriaHandleSelector(dataset)
    res = selector.launch_ui()

    # 处理坐标
    h_vis, w_vis = dataset[0].cam.rgb.shape[:2]
    points_raw = [GeometryHelper.rotate_point_vis_to_raw(p[0], p[1], w_vis, h_vis) for p in res["handle_points_vis"]]

    config_data = {
        "reference_frame_idx": res["ref_frame_idx"],
        "split_frame_idx": res["split_frame_idx"],
        "handle_points_2d_vis": res["handle_points_vis"],
        "handle_points_2d_raw": points_raw
    }

    out_path = os.path.join(save_dir, "handle_selection.json")
    with open(out_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    print(f"\n[Success] 标注完成！")
    print(f" - 参考帧: {res['ref_frame_idx']} (点位已固定)")
    print(f" - 分界帧: {res['split_frame_idx']} (动态起点)")
    print(f" - 结果保存至: {out_path}")

if __name__ == "__main__":
    main()

# python handle_selection.py --mps_path "../data/mps_open_cabinet_5_vrs/"

