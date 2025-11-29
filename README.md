# 微信公众号图片水印去除工具

[English](README_EN.md) | 简体中文

一个基于 SAM (Segment Anything Model) 和 LaMa 的智能水印去除工具，特别优化用于去除微信公众号图片水印。

## ✨ 特性

- 🖱️ **交互式选择**：直接在图片上框选水印区域，所见即所得
- 🎯 **高精度分割**：基于 Meta 的 SAM 模型，精准识别水印区域
- 🎨 **智能修复**：使用 LaMa 模型进行内容感知的图像修复
- 📦 **多种输入模式**：支持交互式、框选、多点、单点四种选择方式
- 🚀 **简单易用**：一行命令即可完成水印去除

## 🎬 效果展示

| 原图 | 去水印后 |
|------|---------|
| ![原图](docs/before.jpg) | ![去水印后](docs/after.jpg) |

## 📋 环境要求

- Python >= 3.10
- CUDA 支持（推荐，CPU 模式会很慢）
- 8GB+ RAM
- 足够的磁盘空间存储模型文件（约 400MB）

## 🔧 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/wechat-watermark-remover.git
cd wechat-watermark-remover
```

### 2. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n watermark python=3.10
conda activate watermark

# 安装 PyTorch（根据你的 CUDA 版本选择）
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 下载预训练模型

创建 `pretrained_models` 文件夹并下载以下模型：

**SAM 模型** (选择一个)：
- [sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (375MB) - 推荐
- [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) (2.5GB) - 更高精度
- [sam_vit_l_0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) (1.2GB)

**LaMa 模型**：
- [big-lama](https://huggingface.co/smartywu/big-lama/tree/main) - 下载所有文件到 `pretrained_models/big-lama/`

下载后的目录结构：
```
pretrained_models/
├── sam_vit_b_01ec64.pth
└── big-lama/
    ├── config.yaml
    └── models/
        └── best.ckpt
```

## 🚀 使用方法

### 方式 1：交互式选择（推荐）⭐

最简单直观的方式，直接在图片上框选水印区域：

```bash
python remove_watermark.py \
    --input_img ./materials/your_image.jpg \
    --coords_type interactive \
    --dilate_kernel_size 25 \
    --output_dir ./outputs \
    --sam_model_type "vit_b" \
    --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

**操作步骤**：
1. 运行命令后会弹出图片窗口
2. 按住鼠标左键拖动，框选水印区域
3. 松开鼠标完成选择（会显示绿色框）
4. 按 `Enter` 或 `Space` 确认并开始处理
5. 按 `r` 键可以重新选择
6. 按 `q` 或 `ESC` 取消退出

### 方式 2：框选模式

如果你已知水印的坐标位置：

```bash
python remove_watermark.py \
    --input_img ./materials/your_image.jpg \
    --coords_type box \
    --box_coords 700 1180 850 1240 \
    --dilate_kernel_size 25 \
    --output_dir ./outputs \
    --sam_model_type "vit_b" \
    --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

其中 `--box_coords` 参数格式为：`x1 y1 x2 y2`（左上角和右下角坐标）

### 方式 3：多点模式

在水印的多个位置点击，提高精度：

```bash
python remove_watermark.py \
    --input_img ./materials/your_image.jpg \
    --coords_type key_in \
    --point_coords 730 1210 800 1210 730 1250 \
    --point_labels 1 1 1 \
    --dilate_kernel_size 25 \
    --output_dir ./outputs \
    --sam_model_type "vit_b" \
    --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

### 方式 4：单点模式

在水印中心点击一次：

```bash
python remove_watermark.py \
    --input_img ./materials/your_image.jpg \
    --coords_type key_in \
    --point_coords 730 1210 \
    --point_labels 1 \
    --dilate_kernel_size 25 \
    --output_dir ./outputs \
    --sam_model_type "vit_b" \
    --sam_ckpt ./pretrained_models/sam_vit_b_01ec64.pth \
    --lama_config ./lama/configs/prediction/default.yaml \
    --lama_ckpt ./pretrained_models/big-lama
```

## 📊 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_img` | 输入图片路径 | 必需 |
| `--coords_type` | 选择模式：`interactive`/`box`/`key_in`/`click` | 必需 |
| `--box_coords` | 框选坐标 [x1, y1, x2, y2] | 可选 |
| `--point_coords` | 点坐标 [x1, y1, x2, y2, ...] | 可选 |
| `--point_labels` | 点标签，1=前景，0=背景 | 可选 |
| `--dilate_kernel_size` | 膨胀核大小，建议 20-30 | None |
| `--output_dir` | 输出目录 | 必需 |
| `--sam_model_type` | SAM 模型类型：`vit_b`/`vit_l`/`vit_h` | `vit_h` |
| `--sam_ckpt` | SAM 模型路径 | 必需 |
| `--lama_config` | LaMa 配置文件路径 | `./lama/configs/prediction/default.yaml` |
| `--lama_ckpt` | LaMa 模型路径 | 必需 |

## 💡 使用技巧

1. **选择模式推荐**：
   - 新手或不确定坐标：使用 `interactive` 模式
   - 批量处理相同位置水印：使用 `box` 模式
   - 复杂形状水印：使用 `key_in` 多点模式

2. **膨胀核大小调整**：
   - 小水印：15-20
   - 中等水印：20-25
   - 大水印：25-35
   - 如果边缘有残留，增大该值

3. **模型选择**：
   - `vit_b`：速度快，效果好，推荐日常使用
   - `vit_l`：平衡选择
   - `vit_h`：最高精度，但速度较慢

4. **提高效果**：
   - 框选时稍微框大一点，确保水印完全包含
   - 对于半透明水印，适当增大 `dilate_kernel_size`
   - 如果一次效果不理想，可以对输出图片再处理一次

## 📁 项目结构

```
wechat-watermark-remover/
├── remove_watermark.py          # 主程序
├── sam_segment.py                # SAM 分割模块
├── lama_inpaint.py              # LaMa 修复模块
├── utils/                        # 工具函数
├── lama/                         # LaMa 相关代码
├── segment_anything/            # SAM 相关代码
├── materials/                    # 示例图片
├── outputs/                      # 输出结果
├── pretrained_models/           # 预训练模型
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

## 🔍 常见问题

**Q: 模型文件太大，下载很慢怎么办？**  
A: 可以使用国内镜像源或从百度网盘等下载，链接见 [模型下载](docs/model_download.md)

**Q: CUDA out of memory 错误？**  
A: 尝试使用更小的模型 `vit_b` 或减小输入图片尺寸

**Q: 去除后边缘有痕迹？**  
A: 增大 `--dilate_kernel_size` 参数，比如从 25 增加到 30-35

**Q: 能否批量处理多张图片？**  
A: 目前需要逐张处理，或者编写简单的脚本循环调用

**Q: 支持哪些图片格式？**  
A: 支持常见格式：JPG, PNG, BMP, WEBP 等

## 🙏 致谢

本项目基于以下优秀的开源项目：

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [LaMa](https://github.com/advimman/lama) - Resolution-robust Large Mask Inpainting
- [Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或通过以下方式联系：

- GitHub Issues: [项目 Issues 页面](https://github.com/你的用户名/wechat-watermark-remover/issues)

## ⭐ Star History

如果这个项目对你有帮助，欢迎点个 Star ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=你的用户名/wechat-watermark-remover&type=Date)](https://star-history.com/#你的用户名/wechat-watermark-remover&Date)
