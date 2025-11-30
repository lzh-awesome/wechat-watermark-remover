"""
微信水印去除 API 服务
支持自动检测和手动指定坐标两种模式
"""
import os
import uuid
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask

# 初始化 FastAPI
app = FastAPI(
    title="Watermark Removal API",
    description="微信公众号图片水印去除服务",
    version="1.0.0"
)

# 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局配置
UPLOAD_DIR = Path("./api_uploads")
OUTPUT_DIR = Path("./api_outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型配置（根据你的实际路径修改）
SAM_MODEL_TYPE = "vit_b"
SAM_CHECKPOINT = "./pretrained_models/sam_vit_b_01ec64.pth"
LAMA_CONFIG = "./lama/configs/prediction/default.yaml"
LAMA_CHECKPOINT = "./pretrained_models/big-lama"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"服务启动中...")
print(f"设备: {DEVICE}")
print(f"SAM 模型: {SAM_MODEL_TYPE}")


class WatermarkDetector:
    """自动水印检测器"""

    @staticmethod
    def detect_wechat_watermark(
            img: np.ndarray,
            threshold: int = 200,
            min_area: int = 500,
            max_area_ratio: float = 0.3
    ) -> List[List[int]]:
        """
        自动检测微信水印区域

        Args:
            img: 输入图片
            threshold: 亮度阈值（微信水印通常是浅色）
            min_area: 最小面积
            max_area_ratio: 最大面积占比

        Returns:
            检测到的水印区域列表 [[x1, y1, x2, y2], ...]
        """
        # 转灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        max_area = height * width * max_area_ratio

        # 检测高亮区域
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作连接文字区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        watermark_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                # 扩展边界，确保完整包含水印
                padding = 10
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                watermark_regions.append([x1, y1, x2, y2])

        return watermark_regions

    @staticmethod
    def detect_by_position(
            img: np.ndarray,
            position: str = "bottom_right",
            width_ratio: float = 0.3,
            height_ratio: float = 0.15
    ) -> List[int]:
        """
        根据位置预设检测水印

        Args:
            img: 输入图片
            position: 位置 (bottom_right, bottom_center, bottom_left, top_right, etc.)
            width_ratio: 宽度占比
            height_ratio: 高度占比

        Returns:
            水印区域 [x1, y1, x2, y2]
        """
        height, width = img.shape[:2]
        w = int(width * width_ratio)
        h = int(height * height_ratio)

        position_map = {
            "bottom_right": [width - w, height - h, width, height],
            "bottom_center": [width // 2 - w // 2, height - h, width // 2 + w // 2, height],
            "bottom_left": [0, height - h, w, height],
            "top_right": [width - w, 0, width, h],
            "top_center": [width // 2 - w // 2, 0, width // 2 + w // 2, h],
            "top_left": [0, 0, w, h],
            "center": [width // 2 - w // 2, height // 2 - h // 2,
                       width // 2 + w // 2, height // 2 + h // 2],
        }

        return position_map.get(position, position_map["bottom_right"])


def remove_watermark(
        img: np.ndarray,
        box_coords: List[int],
        dilate_kernel_size: int = 25
) -> np.ndarray:
    """
    去除水印核心函数

    Args:
        img: 输入图片
        box_coords: 水印区域 [x1, y1, x2, y2]
        dilate_kernel_size: 膨胀核大小

    Returns:
        去水印后的图片
    """
    x1, y1, x2, y2 = box_coords

    # 构建点提示（框的四个角 + 中心点）
    point_coords = [
        [x1, y1],  # 左上
        [x2, y1],  # 右上
        [x1, y2],  # 左下
        [x2, y2],  # 右下
        [(x1 + x2) / 2, (y1 + y2) / 2]  # 中心点
    ]
    point_labels = [1, 1, 1, 1, 1]

    # 使用 SAM 分割
    masks, _, _ = predict_masks_with_sam(
        img,
        point_coords,
        point_labels,
        model_type=SAM_MODEL_TYPE,
        ckpt_p=SAM_CHECKPOINT,
        device=DEVICE,
    )
    masks = masks.astype(np.uint8) * 255

    # 膨胀 mask
    if dilate_kernel_size > 0:
        masks = [dilate_mask(mask, dilate_kernel_size) for mask in masks]

    # 使用 LaMa 修复
    mask = masks[0]
    img_inpainted = inpaint_img_with_lama(
        img, mask, LAMA_CONFIG, LAMA_CHECKPOINT, device=DEVICE
    )

    return img_inpainted


# ==================== API 端点 ====================

@app.get("/")
async def root():
    """健康检查"""
    return {
        "status": "running",
        "service": "Watermark Removal API",
        "version": "1.0.0",
        "device": DEVICE
    }


@app.post("/api/remove-watermark/auto")
async def remove_watermark_auto(
        file: UploadFile = File(...),
        dilate_kernel_size: int = Form(25),
        threshold: int = Form(200),
        detection_mode: str = Form("brightness")  # brightness 或 position
):
    """
    自动检测并去除水印

    Args:
        file: 上传的图片文件
        dilate_kernel_size: 膨胀核大小 (15-35)
        threshold: 亮度检测阈值 (150-220)
        detection_mode: 检测模式 (brightness/position)

    Returns:
        处理后的图片和检测信息
    """
    try:
        # 保存上传文件
        file_id = str(uuid.uuid4())
        upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 加载图片
        img = load_img_to_array(str(upload_path))

        # 自动检测水印
        detector = WatermarkDetector()

        if detection_mode == "brightness":
            watermark_regions = detector.detect_wechat_watermark(
                img, threshold=threshold
            )

            if not watermark_regions:
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "message": "未检测到水印，请尝试调整阈值或使用手动模式",
                        "threshold": threshold
                    }
                )

            # 选择最大的区域（通常是主水印）
            box_coords = max(watermark_regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))

        else:  # position 模式
            box_coords = detector.detect_by_position(img, position="bottom_right")

        # 去除水印
        img_result = remove_watermark(img, box_coords, dilate_kernel_size)

        # 保存结果
        output_path = OUTPUT_DIR / f"{file_id}_result.jpg"
        save_array_to_img(img_result, str(output_path))

        # 清理上传文件
        upload_path.unlink()

        return {
            "success": True,
            "message": "水印去除成功",
            "file_id": file_id,
            "watermark_region": box_coords,
            "download_url": f"/api/download/{file_id}_result.jpg"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove-watermark/manual")
async def remove_watermark_manual(
        file: UploadFile = File(...),
        x1: int = Form(...),
        y1: int = Form(...),
        x2: int = Form(...),
        y2: int = Form(...),
        dilate_kernel_size: int = Form(25)
):
    """
    手动指定坐标去除水印

    Args:
        file: 上传的图片文件
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
        dilate_kernel_size: 膨胀核大小

    Returns:
        处理后的图片
    """
    try:
        # 保存上传文件
        file_id = str(uuid.uuid4())
        upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 加载图片
        img = load_img_to_array(str(upload_path))

        # 验证坐标
        height, width = img.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise HTTPException(
                status_code=400,
                detail=f"坐标无效。图片尺寸: {width}x{height}"
            )

        # 去除水印
        box_coords = [x1, y1, x2, y2]
        img_result = remove_watermark(img, box_coords, dilate_kernel_size)

        # 保存结果
        output_path = OUTPUT_DIR / f"{file_id}_result.jpg"
        save_array_to_img(img_result, str(output_path))

        # 清理上传文件
        upload_path.unlink()

        return {
            "success": True,
            "message": "水印去除成功",
            "file_id": file_id,
            "watermark_region": box_coords,
            "download_url": f"/api/download/{file_id}_result.jpg"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remove-watermark/batch")
async def remove_watermark_batch(
        files: List[UploadFile] = File(...),
        position: str = Form("bottom_right"),
        dilate_kernel_size: int = Form(25)
):
    """
    批量去除水印（按固定位置）

    Args:
        files: 多个图片文件
        position: 水印位置 (bottom_right, bottom_center, etc.)
        dilate_kernel_size: 膨胀核大小

    Returns:
        批处理结果
    """
    results = []
    detector = WatermarkDetector()

    for file in files:
        try:
            # 保存上传文件
            file_id = str(uuid.uuid4())
            upload_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

            with open(upload_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 加载图片
            img = load_img_to_array(str(upload_path))

            # 检测水印区域
            box_coords = detector.detect_by_position(img, position=position)

            # 去除水印
            img_result = remove_watermark(img, box_coords, dilate_kernel_size)

            # 保存结果
            output_path = OUTPUT_DIR / f"{file_id}_result.jpg"
            save_array_to_img(img_result, str(output_path))

            # 清理上传文件
            upload_path.unlink()

            results.append({
                "filename": file.filename,
                "success": True,
                "file_id": file_id,
                "download_url": f"/api/download/{file_id}_result.jpg"
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })

    return {
        "total": len(files),
        "success_count": sum(1 for r in results if r["success"]),
        "results": results
    }


@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """下载处理后的图片"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename
    )


@app.delete("/api/cleanup/{file_id}")
async def cleanup_file(file_id: str):
    """清理临时文件"""
    try:
        output_path = OUTPUT_DIR / f"{file_id}_result.jpg"
        if output_path.exists():
            output_path.unlink()
        return {"success": True, "message": "文件已清理"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("微信水印去除 API 服务")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"SAM 模型: {SAM_MODEL_TYPE}")
    print(f"API 文档: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )