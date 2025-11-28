import torch
import sys
import argparse
import numpy as np
import datetime
from pathlib import Path
from matplotlib import pyplot as plt

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in", "box"],
        help="The way to select coords: click, key_in, or box",
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
    # 使用框选模式（推荐）
    python remove_anything_improved.py \
        --input_img ./materials/3.jpg \
        --coords_type box \
        --box_coords 700 1180 850 1240 \
        --dilate_kernel_size 20 \
        --output_dir ./outputs \
        --sam_model_type "vit_b" \
        --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama

    # 使用多点模式
    python remove_anything_improved.py \
        --input_img ./materials/3.jpg \
        --coords_type key_in \
        --point_coords 730 1210 800 1210 730 1250 \
        --point_labels 1 1 1 \
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
    if args.coords_type == "box":
        # 框选模式
        if args.box_coords is None:
            raise ValueError("--box_coords is required when coords_type is 'box'")

        # 将框转换为点提示（框的四个角点）
        x1, y1, x2, y2 = args.box_coords
        point_coords = [
            [x1, y1],  # 左上
            [x2, y1],  # 右上
            [x1, y2],  # 左下
            [x2, y2],  # 右下
            [(x1 + x2) / 2, (y1 + y2) / 2]  # 中心点
        ]
        point_labels = [1, 1, 1, 1, 1]  # 所有点都是前景
        latest_coords = point_coords[0]  # 用于可视化

    elif args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
        point_coords = [latest_coords]
        point_labels = args.point_labels if args.point_labels else [1]

    elif args.coords_type == "key_in":
        if args.point_coords is None:
            raise ValueError("--point_coords is required when coords_type is 'key_in'")

        # 支持多点输入
        coords_array = np.array(args.point_coords).reshape(-1, 2)
        point_coords = coords_array.tolist()

        # 如果没有提供足够的标签，默认都是前景点
        if args.point_labels is None or len(args.point_labels) != len(point_coords):
            point_labels = [1] * len(point_coords)
        else:
            point_labels = args.point_labels

        latest_coords = point_coords[0]  # 用于可视化

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
        masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / f"{img_stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # inpaint the masked image
    for idx, mask in enumerate(masks):
        mask_p = out_dir / f"mask_{idx}.png"
        img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_inpainted_p)

    print(f"Results saved to: {out_dir}")