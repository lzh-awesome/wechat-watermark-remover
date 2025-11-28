import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatermarkRemover:
    """智能水印去除器"""

    def __init__(
            self,
            sam_model_type: str = "vit_b",
            sam_ckpt: str = "./pretrained_models/sam_vit_b_01ec64.pth",
            lama_config: str = "./lama/configs/prediction/default.yaml",
            lama_ckpt: str = "./pretrained_models/big-lama",
            device: str = None
    ):
        self.sam_model_type = sam_model_type
        self.sam_ckpt = sam_ckpt
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"初始化水印去除器，使用设备: {self.device}")

    def detect_watermark_regions(
            self,
            img: np.ndarray,
            method: str = "edge_density"
    ) -> List[Tuple[int, int]]:
        """
        自动检测水印区域

        Args:
            img: 输入图像数组
            method: 检测方法 (edge_density, color_variance, corner)

        Returns:
            水印可能位置的坐标列表 [(x, y), ...]
        """
        height, width = img.shape[:2]
        coords = []

        if method == "corner":
            # 检测四个角落区域（水印常见位置）
            margin = 100
            corner_positions = [
                (margin, margin),  # 左上
                (width - margin, margin),  # 右上
                (margin, height - margin),  # 左下
                (width - margin, height - margin),  # 右下
            ]
            coords.extend(corner_positions)

        elif method == "edge_density":
            # 基于边缘密度检测
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # 将图像分成网格
            grid_size = 8
            h_step = height // grid_size
            w_step = width // grid_size

            max_density = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    y1, y2 = i * h_step, (i + 1) * h_step
                    x1, x2 = j * w_step, (j + 1) * w_step
                    region = edges[y1:y2, x1:x2]
                    density = np.sum(region) / (h_step * w_step)

                    if density > max_density:
                        max_density = density

                    # 如果边缘密度异常高，可能是水印
                    if density > 20:  # 阈值可调
                        coords.append((x1 + w_step // 2, y1 + h_step // 2))

        elif method == "color_variance":
            # 基于颜色方差检测（水印通常颜色单一）
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # 使用自适应阈值
            thresh = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )

            # 查找轮廓
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 筛选可能的水印轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < (width * height * 0.1):  # 面积在合理范围
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        coords.append((cx, cy))

        logger.info(f"检测到 {len(coords)} 个潜在水印位置")
        return coords if coords else [(width // 2, height // 2)]

    def remove_watermark(
            self,
            img_path: str,
            output_dir: str,
            coords: Optional[List[Tuple[int, int]]] = None,
            auto_detect: bool = True,
            detect_method: str = "corner",
            dilate_kernel_size: int = 15
    ) -> dict:
        """
        去除水印

        Args:
            img_path: 输入图片路径
            output_dir: 输出目录
            coords: 手动指定的水印坐标列表
            auto_detect: 是否自动检测水印
            detect_method: 自动检测方法
            dilate_kernel_size: 膨胀核大小

        Returns:
            处理结果字典
        """
        try:
            # 加载图像
            img = load_img_to_array(img_path)
            logger.info(f"加载图像: {img_path}, 尺寸: {img.shape}")

            # 获取水印坐标
            if auto_detect and coords is None:
                coords = self.detect_watermark_regions(img, method=detect_method)
            elif coords is None:
                raise ValueError("必须提供坐标或启用自动检测")

            # 准备标签（全部设为前景点）
            point_labels = [1] * len(coords)

            # 使用 SAM 生成遮罩
            logger.info(f"使用 SAM 生成遮罩，坐标数: {len(coords)}")
            masks, _, _ = predict_masks_with_sam(
                img,
                coords,
                point_labels,
                model_type=self.sam_model_type,
                ckpt_p=self.sam_ckpt,
                device=self.device,
            )
            masks = masks.astype(np.uint8) * 255

            # 膨胀遮罩
            if dilate_kernel_size > 0:
                masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

            # 创建输出目录
            img_stem = Path(img_path).stem
            out_dir = Path(output_dir) / img_stem
            out_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "input_path": img_path,
                "output_dir": str(out_dir),
                "masks": [],
                "inpainted": []
            }

            # 对每个遮罩进行修复
            for idx, mask in enumerate(masks):
                # 保存遮罩
                mask_p = out_dir / f"mask_{idx}.png"
                save_array_to_img(mask, mask_p)
                results["masks"].append(str(mask_p))

                # 使用 LAMA 修复
                logger.info(f"修复遮罩 {idx + 1}/{len(masks)}")
                img_inpainted = inpaint_img_with_lama(
                    img, mask, self.lama_config, self.lama_ckpt,
                    device=self.device
                )

                # 保存修复后的图像
                img_inpainted_p = out_dir / f"removed_{idx}.png"
                save_array_to_img(img_inpainted, img_inpainted_p)
                results["inpainted"].append(str(img_inpainted_p))

                # 更新img为修复后的图像，用于下一次迭代
                img = img_inpainted

            # 保存最终结果
            final_output = out_dir / "final_result.png"
            save_array_to_img(img, final_output)
            results["final_output"] = str(final_output)

            logger.info(f"处理完成，结果保存至: {final_output}")
            return results

        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise

    def batch_remove(
            self,
            img_dir: str,
            output_dir: str,
            auto_detect: bool = True,
            detect_method: str = "corner",
            extensions: List[str] = None
    ) -> List[dict]:
        """
        批量去除水印

        Args:
            img_dir: 输入图片目录
            output_dir: 输出目录
            auto_detect: 是否自动检测
            detect_method: 检测方法
            extensions: 支持的文件扩展名

        Returns:
            处理结果列表
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']

        img_dir = Path(img_dir)
        results = []

        # 查找所有图片
        img_files = []
        for ext in extensions:
            img_files.extend(img_dir.glob(f"*{ext}"))
            img_files.extend(img_dir.glob(f"*{ext.upper()}"))

        logger.info(f"找到 {len(img_files)} 张图片")

        # 处理每张图片
        for img_path in img_files:
            try:
                result = self.remove_watermark(
                    str(img_path),
                    output_dir,
                    auto_detect=auto_detect,
                    detect_method=detect_method
                )
                results.append(result)
            except Exception as e:
                logger.error(f"处理 {img_path} 失败: {str(e)}")
                results.append({
                    "input_path": str(img_path),
                    "error": str(e)
                })

        return results


if __name__ == "__main__":
    # 测试代码
    remover = WatermarkRemover()

    # 单张图片处理
    result = remover.remove_watermark(
        img_path="./example/remove-anything/dog.jpg",
        output_dir="./results_auto",
        auto_detect=True,
        detect_method="corner"
    )
    print("处理结果:", result)