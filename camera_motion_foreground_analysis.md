# 基于深度和前景分析的相机运镜检测

## 概述

本文档介绍了对 `camera_motion.py` 模块的改进，主要功能包括：

1. **前景背景分离**：通过离线处理好的mask文件区分前景和背景
2. **前景特征点提取**：基于前景区域提取CoTracker特征点
3. **深度分析**：使用Depth-Anything-V2计算相对深度变化
4. **主体大小分析**：分析主体在视频中的大小变化
5. **综合运镜分析**：结合多种信息判断dolly in/out运镜

## 主要功能模块

### 1. 前景Mask加载

```python
def load_foreground_masks(mask_dir, video_path):
    """
    加载离线处理好的前景mask文件
    Args:
        mask_dir: mask文件目录
        video_path: 视频文件路径
    Returns:
        masks: 前景mask数组，形状为 [T, H, W]
    """
```

**功能说明**：
- 根据视频文件名自动查找对应的mask文件
- mask文件格式：`{video_name}_masks.npy`
- 返回numpy数组，每帧包含二值化的前景mask

### 2. 前景中心点计算

```python
def calculate_foreground_center(mask):
    """
    计算前景区域的中心点
    Args:
        mask: 前景mask，形状为 [H, W]
    Returns:
        center: (x, y) 中心点坐标
    """
```

**功能说明**：
- 计算前景区域所有像素的质心
- 用于后续分析特征点到前景中心的距离变化

### 3. 前景特征点提取

```python
def extract_foreground_features(tracks, masks, visibility):
    """
    从前景区域提取特征点
    Args:
        tracks: cotracker特征点轨迹 [T, N, 2]
        masks: 前景mask [T, H, W]
        visibility: 特征点可见性 [T, N, 1]
    Returns:
        foreground_tracks: 前景区域的特征点轨迹
        foreground_visibility: 前景区域特征点的可见性
    """
```

**功能说明**：
- 过滤出位于前景区域内的特征点
- 保留可见性高的特征点（visibility > 0.5）
- 如果前景特征点不足，退回使用所有可见特征点

### 4. 深度模型加载

```python
def load_depth_model(device, encoder='vitl'):
    """
    加载Depth-Anything-V2模型
    Args:
        device: 计算设备
        encoder: 模型编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
    Returns:
        depth_model: 深度计算模型
    """
```

**功能说明**：
- 自动加载Depth-Anything-V2预训练模型
- 支持多种模型尺寸以平衡精度和速度
- 自动检测模型权重文件是否存在

### 5. 深度图计算

```python
def compute_depth_map(depth_model, image, input_size=518):
    """
    计算单帧深度图
    Args:
        depth_model: 深度模型
        image: 输入图像 (H, W, C)
        input_size: 模型输入尺寸
    Returns:
        depth_map: 深度图 (H, W)
    """
```

**功能说明**：
- 使用Depth-Anything-V2计算图像的相对深度
- 返回稠密深度图，值域归一化到[0, 1]
- 支持不同分辨率的输入图像

### 6. 主体大小变化分析

```python
def analyze_subject_size_changes(video_frames, masks=None):
    """
    分析主体大小变化
    Args:
        video_frames: 视频帧列表
        masks: 前景mask数组 (可选)
    Returns:
        size_ratios: 主体大小变化比例列表
    """
```

**功能说明**：
- 使用前景mask计算主体面积变化
- 如果没有mask，使用简单的阈值分割
- 返回每帧主体大小的时间序列

### 7. 深度变化分析

```python
def analyze_depth_changes(depth_maps, masks=None):
    """
    分析深度变化
    Args:
        depth_maps: 深度图列表
        masks: 前景mask数组 (可选)
    Returns:
        depth_stats: 深度统计信息字典
    """
```

**功能说明**：
- 计算前景区域的深度统计信息
- 包括均值、中位数、标准差等统计量
- 提取前景深度变化趋势

### 8. 综合Zoom运镜分析

```python
def analyze_zoom_by_depth_and_size(depth_stats, size_ratios, video_shape):
    """
    基于深度变化和主体大小变化综合判断zoom运镜
    Args:
        depth_stats: 深度统计信息列表
        size_ratios: 主体大小变化比例列表
        video_shape: 视频尺寸 (H, W)
    Returns:
        motion_type: "dolly_in", "dolly_out", "static", 或 None
    """
```

**核心算法**：

1. **深度变化分析**：计算首尾帧深度均值的变化
2. **大小变化分析**：计算主体大小的变化比例
3. **综合判断逻辑**：
   - `dolly_in`：主体变大（size_ratio > 1.15）且深度变小（depth_change < -0.1）
   - `dolly_out`：主体变小（size_ratio < 0.85）且深度变大（depth_change > 0.1）
   - `static`：变化不明显

**阈值设置**：
- `depth_threshold`：深度变化阈值（10%）
- `size_threshold`：大小变化阈值（15%）

## 类结构修改

### CameraPredict类增强

#### 新增方法

```python
def analyze_foreground_zoom(self, tracks, visibility, masks, video_shape):
    """基于前景区域分析zoom运镜"""
```

#### 修改方法

```python
def predict(self, video, fps, end_frame, masks=None):
    """添加masks参数支持前景分析"""
```

```python
def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos"):
    """返回visibility信息"""
```

## 使用方法

### 1. 基本调用

```python
# 初始化相机预测器（包含深度分析）
camera = CameraPredict(device, submodules_dict, enable_depth=True)

# 计算相机运镜（包含深度和前景分析）
all_results, video_results = compute_camera_motion(
    json_dir,
    device,
    submodules_dict,
    mask_dir="/path/to/masks",  # 可选：前景mask目录
    enable_depth=True  # 可选：启用深度分析
)
```

### 2. Mask文件准备

前景mask文件需要预先准备，文件结构如下：

```
mask_dir/
├── video1_masks.npy
├── video2_masks.npy
└── ...
```

每个mask文件包含：
- 形状：`[T, H, W]`（时间，高度，宽度）
- 数据类型：numpy.ndarray
- 值域：0（背景）或1（前景）

### 3. 深度模型准备

需要下载Depth-Anything-V2的预训练权重文件：

```bash
# 下载模型权重（以vitl为例）
# 将文件放置在：VBench-2.0/vbench2/third_party/Depth-Anything-V2-main/Depth-Anything-V2-main/checkpoints/
```

### 4. 参数配置

在调用 `compute_camera_motion` 时可以通过 `kwargs` 传递：

```python
compute_camera_motion(
    json_dir,
    device,
    submodules_dict,
    mask_dir="/path/to/masks",  # 可选：前景mask目录
    enable_depth=True           # 可选：启用深度分析，默认True
)
```

### 5. 高级配置

```python
# 禁用深度分析，仅使用前景特征点分析
compute_camera_motion(
    json_dir,
    device,
    submodules_dict,
    mask_dir="/path/to/masks",
    enable_depth=False
)

# 仅使用深度分析，不使用mask
compute_camera_motion(
    json_dir,
    device,
    submodules_dict,
    mask_dir=None,
    enable_depth=True
)
```

## 算法原理

### 1. 前景分离原理

利用预处理的分割mask将视频帧分为前景（主体对象）和背景，专注分析前景区域的运动特征。

### 2. 特征点分析原理

CoTracker在整个视频帧上均匀采样特征点，通过前景mask过滤，只保留前景区域的特征点进行分析。

### 3. 深度分析原理

使用Depth-Anything-V2计算图像的相对深度：

- **深度图计算**：将RGB图像转换为稠密深度图
- **前景深度提取**：从深度图中提取前景区域的深度信息
- **深度变化分析**：比较首尾帧前景深度的统计变化

**深度变化规律**：
- `dolly_in`：相机靠近主体，前景深度变小（更近）
- `dolly_out`：相机远离主体，前景深度变大（更远）

### 4. 主体大小分析原理

通过计算主体在图像中的像素面积来分析大小变化：

- **面积计算**：使用mask计算前景区域像素数量
- **变化趋势**：分析面积随时间的变化比例
- **阈值判断**：超过15%变化认为有显著变化

### 5. 综合Zoom检测原理

结合深度变化和主体大小变化进行多模态判断：

**Dolly In（推进）**：
- 主体在图像中变大（size_ratio > 1.15）
- 前景深度变小（depth_change < -0.1）
- 相机向主体靠近

**Dolly Out（拉远）**：
- 主体在图像中变小（size_ratio < 0.85）
- 前景深度变大（depth_change > 0.1）
- 相机远离主体

**多模态融合**：
- 同时满足深度和大小变化条件才判定为zoom
- 提供更准确的运镜类型识别

### 4. 距离变化分析

使用线性回归分析特征点到中心距离的时序变化：
- 正斜率：dolly out
- 负斜率：dolly in
- 接近零斜率：static

## 性能优化

1. **并行处理**：支持多进程并行处理多个视频
2. **内存优化**：按需加载mask文件和深度模型，减少内存占用
3. **容错机制**：深度模型或mask文件缺失时自动退回到原始算法
4. **阈值自适应**：根据视频分辨率自动调整判断阈值
5. **模型缓存**：深度模型单例模式，避免重复加载
6. **GPU加速**：深度计算使用GPU加速，提高处理速度

## 输出格式

```python
{
    'video_path': '/path/to/video.mp4',
    'video_results': 1.0,  # 1.0表示检测正确，0.0表示检测错误
    'motion_types': ['dolly_in', 'pan_left', ...]  # 检测到的运镜类型
}
```

## 注意事项

1. **Mask质量**：前景分割质量直接影响检测准确性
2. **时序一致性**：确保mask序列与视频帧一一对应
3. **分辨率匹配**：mask分辨率应与视频分辨率一致
4. **深度模型权重**：需要预先下载Depth-Anything-V2的模型权重文件
5. **GPU内存**：深度模型需要较大的GPU内存，建议使用16GB以上显存
6. **阈值调优**：根据具体应用场景调整depth_threshold、size_threshold等参数
7. **计算时间**：深度分析会增加处理时间，建议根据需求选择是否启用

## 扩展功能

该框架可以进一步扩展支持：

1. **多目标跟踪**：支持多个前景对象的独立分析
2. **3D运镜检测**：结合深度信息分析3D相机运动
3. **实时处理**：优化算法支持实时视频流处理
4. **自适应阈值**：基于视频内容自动调整判断阈值

## 依赖项

```python
import cv2
import numpy as np
import torch
import decord
from PIL import Image
from tqdm import tqdm
import sys
import os.path as osp
```

**深度分析相关依赖**：
- Depth-Anything-V2：需要放置在 `vbench2/third_party/Depth-Anything-V2-main/Depth-Anything-V2-main/`
- 模型权重文件：`checkpoints/depth_anything_v2_*.pth`

**安装说明**：
1. 确保已安装所有必要的Python包
2. 下载Depth-Anything-V2预训练模型权重
3. 将权重文件放置在正确的checkpoints目录中
4. 正确配置CoTracker模型路径

**可选依赖**：
- 如果不使用深度分析，可以设置 `enable_depth=False`
- 如果没有mask文件，系统会自动使用阈值分割作为备选方案
