# 基于前景分析的相机运镜检测

## 概述

本文档介绍了对 `camera_motion.py` 模块的改进，主要功能包括：

1. **前景背景分离**：通过离线处理好的mask文件区分前景和背景
2. **前景特征点提取**：基于前景区域提取CoTracker特征点
3. **运镜分析**：基于特征点的光流变化和距离特点判断dolly in/out运镜

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

### 4. Zoom运镜分析

```python
def analyze_zoom_motion(foreground_tracks, masks, video_shape):
    """
    基于前景特征点到中心点的平均距离变化分析zoom运镜
    Args:
        foreground_tracks: 前景特征点轨迹
        masks: 前景mask数组
        video_shape: 视频尺寸 (H, W)
    Returns:
        motion_type: "dolly_in", "dolly_out", "static", 或 None
    """
```

**核心算法**：

1. **距离计算**：计算每帧前景特征点到前景中心的平均距离
2. **趋势分析**：使用线性回归分析距离变化趋势
3. **运镜判断**：
   - `dolly_in`：距离递减趋势（slope < -0.5）
   - `dolly_out`：距离递增趋势（slope > 0.5）
   - `static`：距离变化不明显

**阈值设置**：
- `distance_threshold`：相对距离变化阈值（视频尺寸的2%）
- `slope_threshold`：趋势斜率阈值（0.5）

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
# 初始化相机预测器
camera = CameraPredict(device, submodules_dict)

# 计算相机运镜（包含前景分析）
all_results, video_results = compute_camera_motion(
    json_dir, 
    device, 
    submodules_dict, 
    mask_dir="/path/to/masks"  # 前景mask目录
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

### 3. 参数配置

在调用 `compute_camera_motion` 时可以通过 `kwargs` 传递：

```python
compute_camera_motion(
    json_dir, 
    device, 
    submodules_dict,
    mask_dir="/path/to/masks"  # 可选：前景mask目录
)
```

## 算法原理

### 1. 前景分离原理

利用预处理的分割mask将视频帧分为前景（主体对象）和背景，专注分析前景区域的运动特征。

### 2. 特征点分析原理

CoTracker在整个视频帧上均匀采样特征点，通过前景mask过滤，只保留前景区域的特征点进行分析。

### 3. Zoom检测原理

**Dolly In（推进）**：
- 相机向前景主体靠近
- 前景特征点相对于前景中心向外扩散
- 特征点到中心距离增加

**Dolly Out（拉远）**：
- 相机远离前景主体
- 前景特征点相对于前景中心向内收缩
- 特征点到中心距离减少

### 4. 距离变化分析

使用线性回归分析特征点到中心距离的时序变化：
- 正斜率：dolly out
- 负斜率：dolly in
- 接近零斜率：static

## 性能优化

1. **并行处理**：支持多进程并行处理多个视频
2. **内存优化**：按需加载mask文件，减少内存占用
3. **容错机制**：mask文件缺失时自动退回到原始算法
4. **阈值自适应**：根据视频分辨率自动调整判断阈值

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
4. **阈值调优**：根据具体应用场景调整distance_threshold和slope_threshold

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
```

确保已安装所有必要的Python包和正确配置CoTracker模型。
