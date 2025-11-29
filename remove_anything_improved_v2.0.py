import torch
import sys
import argparse
import numpy as np
import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import cv2

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class InteractiveBoxSelector:
    """交互式框选工具"""

    def __init__(self, img_path):
        self.img = cv2.imread(img_path)
        self.img_display = self.img.copy()
        self.img_path = img_path
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.box_coords = None

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新终点并重绘
                self.end_point = (x, y)
                self.img_display = self.img.copy()
                cv2.rectangle(self.img_display, self.start_point,
                              self.end_point, (0, 255, 0), 2)
                cv2.imshow('Select Watermark Area', self.img_display)

        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            self.drawing = False
            self.end_point = (x, y)
            self.img_display = self.img.copy()
            cv2.rectangle(self.img_display, self.start_point,
                          self.end_point, (0, 255, 0), 2)
            cv2.imshow('Select Watermark Area', self.img_display)

            # 保存坐标
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            # 确保 x1 < x2, y1 < y2
            self.box_coords = [min(x1, x2), min(y1, y2),
                               max(x1, x2), max(y1, y2)]

    def select_box(self):
        """显示图片并让用户选择区域"""
        # 调整显示大小（如果图片太大）
        height, width = self.img.shape[:2]
        max_height = 900
        if height > max_height:
            scale = max_height / height
            display_width = int(width * scale)
            display_height = int(height * scale)
            self.img = cv2.resize(self.img, (display_width, display_height))
            self.img_display = self.img.copy()
            self.scale_factor = scale
        else:
            self.scale_factor = 1.0

        cv2.namedWindow('Select Watermark Area')
        cv2.setMouseCallback('Select Watermark Area', self.mouse_callback)

        print("=" * 60)
        print("交互式选择水印区域:")
        print("1. 在图片上按住鼠标左键并拖动，框选水印区域")
        print("2. 松开鼠标完成选择")
        print("3. 按 'r' 键重新选择")
        print("4. 按 'Enter' 或 'Space' 确认选择")
        print("5. 按 'q' 或 'ESC' 退出")
        print("=" * 60)

        cv2.imshow('Select Watermark Area', self.img_display)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                # 重置
                self.img_display = self.img.copy()
                self.start_point = None
                self.end_point = None
                self.box_coords = None
                cv2.imshow('Select Watermark Area', self.img_display)
                print("已重置，请重新选择区域")

            elif key in [13, 32]:  # Enter or Space
                if self.box_coords is not None:
                    cv2.destroyAllWindows()
                    # 如果有缩放，需要转换回原始坐标
                    if self.scale_factor != 1.0:
                        self.box_coords = [
                            int(coord / self.scale_factor)
                            for coord in self.box_coords
                        ]
                    print(f"已选择区域: {self.box_coords}")
                    return self.box_coords
                else:
                    print("请先选择一个区域！")

            elif key in [ord('q'), 27]:  # q or ESC
                cv2.destroyAllWindows()
                print("已取消选择")
                return None

        cv2.destroyAllWindows()
        return self.box_coords


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in", "box", "interactive"],
        help="The way to select coords: click, key_in, box, or interactive",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=False,
        help="The coordinate of point prompts, [x1 y1 x2 y2 ...].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=False,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--box_coords", type=float, nargs=4, required=False,
        help="Box coordinates [x1, y1, x2, y2] (top-left and bottom-right corners).",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    # 使用交互式选择模式（推荐！）
    python remove_anything_improved_v2.0.py \
        --input_img ./materials/3.jpg \
        --coords_type interactive \
        --dilate_kernel_size 25 \
        --output_dir ./outputs \
        --sam_model_type "vit_b" \
        --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama

    # 使用框选模式（已知坐标）
    python remove_anything_improved_v2.0.py \
        --input_img ./materials/3.jpg \
        --coords_type box \
        --box_coords 700 1180 850 1240 \
        --dilate_kernel_size 20 \
        --output_dir ./outputs \
        --sam_model_type "vit_b" \
        --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)

    # 处理不同的坐标输入方式
    if args.coords_type == "interactive":
        # 交互式选择模式
        print("\n启动交互式选择模式...")
        selector = InteractiveBoxSelector(args.input_img)
        box_coords = selector.select_box()

        if box_coords is None:
            print("未选择区域，程序退出")
            sys.exit(0)

        # 将框转换为点提示
        x1, y1, x2, y2 = box_coords
        point_coords = [
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x1, y2],  # 左下
            [x2, y2],  # 右下
            [(x1 + x2) / 2, (y1 + y2) / 2]  # 中心点
        ]
        point_labels = [1, 1, 1, 1, 1]
        latest_coords = point_coords[0]

        print(f"\n开始处理水印区域: [{x1}, {y1}, {x2}, {y2}]")

    elif args.coords_type == "box":
        # 框选模式
        if args.box_coords is None:
            raise ValueError("--box_coords is required when coords_type is 'box'")

        x1, y1, x2, y2 = args.box_coords
        point_coords = [
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x1, y2],  # 左下
            [x2, y2],  # 右下
            [(x1 + x2) / 2, (y1 + y2) / 2]  # 中心点
        ]
        point_labels = [1, 1, 1, 1, 1]
        latest_coords = point_coords[0]

    elif args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
        point_coords = [latest_coords]
        point_labels = args.point_labels if args.point_labels else [1]

    elif args.coords_type == "key_in":
        if args.point_coords is None:
            raise ValueError("--point_coords is required when coords_type is 'key_in'")

        coords_array = np.array(args.point_coords).reshape(-1, 2)
        point_coords = coords_array.tolist()

        if args.point_labels is None or len(args.point_labels) != len(point_coords):
            point_labels = [1] * len(point_coords)
        else:
            point_labels = args.point_labels

        latest_coords = point_coords[0]

    print("\n正在使用 SAM 模型分割水印区域...")
    # 使用 SAM 预测 mask
    masks, _, _ = predict_masks_with_sam(
        img,
        point_coords,
        point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    if args.dilate_kernel_size is not None:
        print(f"正在膨胀 mask (kernel_size={args.dilate_kernel_size})...")
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / f"{img_stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n正在保存可视化结果到: {out_dir}")
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width / dpi / 0.77, height / dpi / 0.77))
        plt.imshow(img)
        plt.axis('off')

        # 可视化所有点
        show_points(plt.gca(), point_coords, point_labels,
                    size=(width * 0.04) ** 2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()

    print("\n正在使用 LaMa 模型修复图片...")
    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)

    print(f"\n✅ 完成！结果已保存到: {out_dir}")
    print(f"   - 去水印图片: inpainted_with_mask_0.png")
    print(f"   - 分割掩码: mask_0.png")
    print(f"   - 可视化结果: with_mask_0.png")