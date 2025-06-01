# VBench 图像处理套件使用说明

## 概述

本套件实现了基于VBench论文的图像裁剪流水线，支持自动图像类型分类和多种宽高比的智能裁剪。

## 核心功能

### 1. 图像类型自动分类
- **scenery**: 风景、自然景观
- **architecture**: 建筑、城市景观
- **single_person**: 单人肖像
- **multiple_person**: 多人场景
- **animals**: 动物
- **food**: 食物
- **plant**: 植物、花卉
- **abstract**: 抽象艺术

### 2. 三步裁剪流程

#### 第一步：16:9 裁剪
- 将原始图像裁剪为16:9宽高比
- 保持图像主要内容居中

#### 第二步：1:1 裁剪
- 基于图像类型进行智能裁剪：
  - **风景类**（scenery, architecture, plant, abstract）：居中裁剪
  - **人物类**（single_person, multiple_person）：偏向上方裁剪
  - **其他类**（animals, food）：居中裁剪

#### 第三步：多比例裁剪
支持的宽高比：
- **1:1** - 正方形
- **8:5** - 1.6:1
- **7:4** - 1.75:1  
- **16:9** - 1.78:1

## 使用方法

### 1. 处理图像生成信息

```bash
python scripts/image_process_suite.py \
    --mode process \
    --image_dir /path/to/images \
    --metadata_file /path/to/metadata.json \
    --output_dir /path/to/output
```

**参数说明：**
- `--mode process`: 处理模式，生成图像裁剪信息
- `--image_dir`: 原始图像目录
- `--metadata_file`: 元数据文件（可选，包含图像描述和URL）
- `--output_dir`: 输出目录，将生成 `image_info.json`

### 2. 根据信息裁剪图像

```bash
python scripts/image_process_suite.py \
    --mode crop \
    --image_dir /path/to/images \
    --info_json /path/to/image_info.json \
    --output_dir /path/to/cropped_images \
    --target_ratio 16-9
```

**参数说明：**
- `--mode crop`: 裁剪模式，根据信息裁剪图像
- `--image_dir`: 原始图像目录
- `--info_json`: 图像信息JSON文件
- `--output_dir`: 裁剪后图像输出目录
- `--target_ratio`: 目标比例（1-1, 8-5, 7-4, 16-9）

## 元数据文件格式

```json
{
  "image1.jpg": {
    "caption": "图像描述文本",
    "url": "图像来源URL"
  },
  "image2.jpg": {
    "caption": "另一张图像的描述",
    "url": "另一张图像的URL"
  }
}
```

## 输出格式

生成的 `image_info.json` 包含每张图像的完整处理信息：

```json
[
  {
    "file_name": "example.jpg",
    "url": "www.example.com/photo",
    "type": "scenery",
    "origin_width": 4000,
    "origin_height": 3000,
    "first_crop": {
      "width": 4000,
      "height": 3000,
      "first_bbox": [0, 375, 4000, 2250]
    },
    "second_crop": {
      "width": 4000,
      "height": 2250,
      "second_bbox": [875, 0, 2250, 2250]
    },
    "diff_ratio_crop": {
      "1-1": [875, 375, 2250, 2250],
      "8-5": [50, 375, 3600, 2250],
      "7-4": [6, 375, 3938, 2248],
      "16-9": [0, 375, 4000, 2250]
    },
    "caption": "图像描述"
  }
]
```

## 处理流程图

```
原始图像 → 读取图像路径，获取宽高比例 H×W
    ↓
判断 H>W?
    ↓
├─ N (风景处理分支)
│   ↓
│   风景处理分支
│   ↓
│   第一次裁剪（相对于原图），比例是16:9
│   ↓
│   在16:9裁剪图像的基础上做第二次裁剪，裁剪比例是1:1
│   ↓
│   相对于原图，裁剪到不同比例的尺寸计算：
│   1:1,8:5,7:4,16:9，得到不同的 (X,Y,W,H)
│
└─ Y (人像处理分支)
    ↓
    人像处理分支
    ↓
    第一次裁剪（相对于原图），比例是1:1
    ↓
    在1:1裁剪图像的基础上做第二次裁剪，裁剪比例是16:9
    ↓
    相对于原图，裁剪到不同比例的尺寸计算：
    1:1,8:5,7:4,16:9，得到不同的 (X,Y,W,H)
    ↓
把以上修改写入到Json并保存
```

## 示例用法

### 完整处理流程示例

```bash
# 1. 准备图像和元数据
mkdir -p /tmp/test_images /tmp/output

# 2. 处理图像生成信息
python scripts/image_process_suite.py \
    --mode process \
    --image_dir /tmp/test_images \
    --metadata_file scripts/example_metadata.json \
    --output_dir /tmp/output

# 3. 裁剪为16:9比例
python scripts/image_process_suite.py \
    --mode crop \
    --image_dir /tmp/test_images \
    --info_json /tmp/output/image_info.json \
    --output_dir /tmp/output/16-9 \
    --target_ratio 16-9

# 4. 裁剪为1:1比例
python scripts/image_process_suite.py \
    --mode crop \
    --image_dir /tmp/test_images \
    --info_json /tmp/output/image_info.json \
    --output_dir /tmp/output/1-1 \
    --target_ratio 1-1
```

## 技术特点

1. **智能类型识别**：基于图像描述自动分类图像类型
2. **差异化处理**：不同类型图像采用不同的裁剪策略
3. **多比例支持**：一次处理生成多种宽高比的裁剪信息
4. **批量处理**：支持大规模图像批量处理
5. **可重现性**：使用固定随机种子确保结果一致

## 依赖要求

```bash
pip install Pillow
```

## 注意事项

1. 确保图像文件格式为 JPG、JPEG、PNG 或 BMP
2. 元数据文件为可选，如果不提供将使用文件名作为描述
3. 裁剪过程会保持图像质量，不进行压缩
4. 输出文件名格式：`原文件名_比例.扩展名`（如：`image_16-9.jpg`） 