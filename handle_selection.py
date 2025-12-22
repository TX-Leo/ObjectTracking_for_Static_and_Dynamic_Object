# -*- coding: utf-8 -*-
# @FileName: handle_selection.py
# @Description: 第一阶段：手动选择参考帧和分界帧（支持不定数量点位标注）。

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

class GeometryHelper:
    @staticmethod
    def rotate_point_vis_to_raw(u_vis, v_vis, vis_w, vis_h):
        """Portrait (显示) -> Landscape (原始)"""
        u_raw = v_vis
        v_raw = vis_w - 1 - u_vis
        return float(u_raw), float(v_raw)

# ==============================================================================
# [Module] 交互式选择器 (支持多点模式)
# ==============================================================================

class AriaHandleSelector:
    def __init__(self, dataset: AriaDataset):
        self.dataset = dataset
        self.total_frames = len(dataset)
        self.state = {
            "ref_frame_idx": 0,
            "split_frame_idx": 0,
            "handle_points_vis": [], 
            "confirmed": False
        }
        self.current_img_vis = None
        # 预定义一组颜色循环
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 0, 255), (0, 255, 128)
        ]

    def _draw_points(self, img):
        img_draw = img.copy()
        for i, p in enumerate(self.state["handle_points_vis"]):
            color = self.colors[i % len(self.colors)]
            cv2.circle(img_draw, p, 8, color, -1)
            cv2.putText(img_draw, str(i), (p[0]+10, p[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img_draw

    def _get_frame_data(self, idx):
        self.current_img_vis = cv2.cvtColor(self.dataset[idx].cam.rgb, cv2.COLOR_BGR2RGB)
        img_draw = self._draw_points(self.current_img_vis)
        return img_draw, f"当前预览帧: {idx} | 已标 {len(self.state['handle_points_vis'])} 点"

    def _on_click(self, evt: gr.SelectData):
        x, y = evt.index[0], evt.index[1]
        self.state["handle_points_vis"].append((x, y))
        img_draw = self._draw_points(self.current_img_vis)
        return img_draw, f"已标注第 {len(self.state['handle_points_vis'])} 个点: ({x}, {y})"

    def _undo(self):
        if self.state["handle_points_vis"]:
            self.state["handle_points_vis"].pop()
        img_draw = self._draw_points(self.current_img_vis)
        return img_draw, f"已撤销，当前剩余 {len(self.state['handle_points_vis'])} 点"

    def _clear(self):
        self.state["handle_points_vis"] = []
        return self.current_img_vis, "已清空所有点"

    def _set_as_ref(self, idx):
        self.state["ref_frame_idx"] = idx
        return f"已设定参考帧: {idx}"

    def _set_as_split(self, idx):
        self.state["split_frame_idx"] = idx
        return f"已设定分界帧: {idx}"

    def _confirm(self):
        if len(self.state["handle_points_vis"]) < 3:
            return "错误: 至少需要标注 3 个点才能进行 PnP 解算！"
        self.state["confirmed"] = True
        return "配置已保存！请回到控制台查看输出并关闭网页。"

    def launch_ui(self):
        with gr.Blocks(title="Aria Multi-Point Selector") as demo:
            gr.Markdown("## 第一阶段：关键点标注 (多点模式)")
            gr.Markdown("""
            **操作指南：**
            1. **选参考帧**：滑动条找一帧静态帧，点击图中标注点（建议 **6个以上** 以提高PnP稳定性）。
            2. **设为参考**：点击 'Set as Ref'。
            3. **选分界点**：滑动条找到物体开始运动的一帧，点击 'Set as Split'。
            4. **确认**：点击 'Confirm' 退出。
            """)
            
            with gr.Row():
                with gr.Column(scale=3):
                    img_display = gr.Image(label="标注视图 (点击取点)", interactive=False)
                    with gr.Row():
                        undo_btn = gr.Button("Undo (撤销上一点)")
                        clear_btn = gr.Button("Clear All (清空点位)", variant="stop")
                
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(0, self.total_frames-1, value=0, step=1, label="帧序列进度")
                    btn_ref = gr.Button("1. Set as Reference (Points Frame)", variant="secondary")
                    btn_split = gr.Button("2. Set as Split (Movement Start)", variant="secondary")
                    status_box = gr.Textbox(label="状态信息", value="等待操作...")
                    confirm_btn = gr.Button("3. Confirm & Save", variant="primary")
            
            # 事件绑定
            frame_slider.change(self._get_frame_data, frame_slider, [img_display, status_box])
            img_display.select(self._on_click, None, [img_display, status_box])
            undo_btn.click(self._undo, None, [img_display, status_box])
            clear_btn.click(self._clear, None, [img_display, status_box])
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
    
    vrs_files = [f for f in os.listdir(args.mps_path) if f.endswith('.vrs')]
    if not vrs_files:
        print("Error: No .vrs file found in mps_path")
        return
    
    vrs_file = os.path.join(args.mps_path, vrs_files[0])
    hand_csv = os.path.join(args.mps_path, "hand_tracking/hand_tracking_results.csv")
    dataset = AriaDataset(args.mps_path, vrs_file, hand_csv, save_dir)

    selector = AriaHandleSelector(dataset)
    res = selector.launch_ui()

    # 处理坐标转换
    h_vis, w_vis = dataset[0].cam.rgb.shape[:2]
    points_raw = [GeometryHelper.rotate_point_vis_to_raw(p[0], p[1], w_vis, h_vis) for p in res["handle_points_vis"]]

    config_data = {
        "reference_frame_idx": res["ref_frame_idx"],
        "split_frame_idx": res["split_frame_idx"],
        "num_points": len(res["handle_points_vis"]),
        "handle_points_2d_vis": res["handle_points_vis"],
        "handle_points_2d_raw": points_raw
    }

    out_path = os.path.join(save_dir, "handle_selection.json")
    with open(out_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    print(f"\n[Success] 标注完成！")
    print(f" - 参考帧: {res['ref_frame_idx']}")
    print(f" - 分界帧: {res['split_frame_idx']}")
    print(f" - 总点数: {len(res['handle_points_vis'])}")
    print(f" - 结果保存至: {out_path}")

if __name__ == "__main__":
    main()
# python handle_selection_v2.py --mps_path "../data/mps_open_cabinet_5_vrs/"