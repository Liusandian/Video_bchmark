# 运镜幅度分析功能文档

## 概述

本文档介绍对 `camera_motion.py` 模块新增的运镜幅度分析功能。该功能基于首尾帧特征点移动像素距离，将运镜幅度分为5个档位，提供更细粒度的运镜强度评估。

## 新增功能

### 1. 运镜幅度计算

#### 函数：`calculate_motion_magnitude(tracks, visibility, video_shape)`

**功能说明**：
- 计算CoTracker特征点在首尾帧之间的平均位移距离
- 基于视频最小分辨率（scale = min(height, width)）进行归一化
- 只考虑在首尾帧都可见的有效特征点

**参数**：
- `tracks`: cotracker特征点轨迹 [T, N, 2]
- `visibility`: 特征点可见性 [T, N, 1]
- `video_shape`: 视频尺寸 (H, W)

**返回值**：
```python
{
    "magnitude_level": "moderate",      # 运镜档位
    "magnitude_ratio": 1.25,            # 相对于scale的比例
    "pixel_displacement": 960.5,        # 平均像素位移距离
    "valid_points": 85                  # 有效特征点数量
}
```

### 2. 运镜档位分类

#### 函数：`classify_motion_magnitude(magnitude_ratio)`

**档位定义**：

| 档位 | 英文标识 | 倍数范围 | 描述 |
|------|----------|----------|------|
| 极其明显 | `extremely_obvious` | ≥ 4.0x | 运镜幅度极大，非常显著的相机移动 |
| 非常明显 | `very_obvious` | 3.0x - 4.0x | 运镜幅度很大，明显的相机移动 |
| 明显 | `obvious` | 2.0x - 3.0x | 运镜幅度较大，可清晰感知 |
| 一般 | `moderate` | 1.0x - 2.0x | 运镜幅度中等，适度的相机移动 |
| 不明显 | `subtle` | 0.5x - 1.0x | 运镜幅度较小，轻微的相机移动 |
| 几乎没有 | `none` | < 0.5x | 运镜幅度极小，几乎静止 |

**计算公式**：
```
magnitude_ratio = average_pixel_displacement / min(video_height, video_width)
```

## 修改的代码结构

### 1. CameraPredict.predict() 方法

**修改前**：
```python
def predict(self, video, fps, end_frame, masks=None):
    # ... 特征点追踪和运镜类型检测
    return results  # 只返回运镜类型列表
```

**修改后**：
```python
def predict(self, video, fps, end_frame, masks=None):
    # ... 特征点追踪和运镜类型检测
    # 计算运镜幅度
    magnitude_info = calculate_motion_magnitude(pred_track, pred_visibility, (self.height, self.width))
    return results, magnitude_info  # 返回运镜类型和幅度信息
```

### 2. camera_motion() 函数

**扩展输出结构**：
```python
{
    'video_path': '/path/to/video.mp4',
    'video_results': 1.0,                    # 检测准确性分数
    'motion_types': ['pan_left', 'tilt_up'], # 检测到的运镜类型
    'magnitude_info': {                      # 新增：运镜幅度信息
        'magnitude_level': 'moderate',
        'magnitude_ratio': 1.25,
        'pixel_displacement': 960.5,
        'valid_points': 85
    },
    'expected_label': 'pan_left'             # 期望的标签
}
```

## 使用示例

### 基本用法

```python
from vbench2.camera_motion import compute_camera_motion

# 计算运镜结果，包含幅度分析
avg_score, video_results = compute_camera_motion(
    json_dir="path/to/prompts", 
    device="cuda:0", 
    submodules_dict={"repo": "facebookresearch/co-tracker", "model": "cotracker2_online"},
    mask_dir="path/to/masks"  # 可选
)

# 查看单个视频的结果
for result in video_results:
    print(f"视频: {result['video_path']}")
    print(f"运镜类型: {result['motion_types']}")
    print(f"运镜幅度: {result['magnitude_info']['magnitude_level']}")
    print(f"幅度比例: {result['magnitude_info']['magnitude_ratio']:.3f}")
    print(f"像素位移: {result['magnitude_info']['pixel_displacement']:.1f}")
    print("-" * 50)
```

### 档位统计分析

```python
# 统计各档位分布
magnitude_stats = {}
for result in video_results:
    level = result['magnitude_info']['magnitude_level']
    magnitude_stats[level] = magnitude_stats.get(level, 0) + 1

print("运镜幅度档位分布:")
for level, count in magnitude_stats.items():
    print(f"{level}: {count} 个视频")
```

### 幅度阈值分析

```python
# 分析不同运镜类型的幅度特征
motion_magnitude_analysis = {}
for result in video_results:
    for motion_type in result['motion_types']:
        if motion_type not in motion_magnitude_analysis:
            motion_magnitude_analysis[motion_type] = []
        motion_magnitude_analysis[motion_type].append(
            result['magnitude_info']['magnitude_ratio']
        )

# 计算各运镜类型的平均幅度
for motion_type, ratios in motion_magnitude_analysis.items():
    avg_ratio = sum(ratios) / len(ratios)
    print(f"{motion_type}: 平均幅度比例 {avg_ratio:.3f}")
```

## 技术细节

### 1. 特征点过滤策略

- **可见性过滤**：只考虑在首尾帧visibility > 0.5的特征点
- **边界检查**：确保特征点坐标在视频帧范围内
- **容错机制**：当有效特征点数量为0时，返回默认值

### 2. 位移计算方法

```python
# 欧几里得距离计算
dx = last_frame[i, 0] - first_frame[i, 0]
dy = last_frame[i, 1] - first_frame[i, 1]
displacement = np.sqrt(dx**2 + dy**2)
```

### 3. 归一化策略

使用视频最小分辨率进行归一化，确保不同分辨率视频的结果具有可比性：
```python
scale = min(video_height, video_width)
magnitude_ratio = average_displacement / scale
```

## 应用场景

### 1. 视频质量评估
- 评估生成视频的运镜平滑度
- 检测异常的相机抖动
- 量化运镜强度是否符合预期

### 2. 数据集分析
- 统计数据集中不同运镜幅度的分布
- 筛选特定幅度范围的视频样本
- 分析运镜类型与幅度的相关性

### 3. 模型比较
- 比较不同生成模型的运镜控制能力
- 评估模型在不同运镜幅度下的表现
- 进行细粒度的运镜质量评估

## 注意事项

### 1. 参数调优

档位阈值可根据具体应用场景调整：
```python
def classify_motion_magnitude(magnitude_ratio):
    # 可以根据需要调整这些阈值
    if magnitude_ratio >= 4.0:
        return "extremely_obvious"
    # ... 其他档位
```

### 2. 性能考虑

- 运镜幅度计算的时间复杂度为O(N)，其中N为特征点数量
- 内存占用主要来自特征点轨迹存储
- 建议在GPU上运行以获得最佳性能

### 3. 准确性因素

- **特征点质量**：CoTracker特征点的追踪准确性直接影响结果
- **视频质量**：低质量视频可能导致特征点追踪不稳定
- **运镜类型**：某些复杂运镜（如螺旋运动）可能需要特殊处理

## 未来扩展

### 1. 动态阈值
根据视频内容自适应调整档位阈值

### 2. 时序分析
分析整个视频序列的运镜幅度变化趋势

### 3. 多尺度分析
结合全局和局部特征点进行多尺度运镜分析

### 4. 运镜平滑度
评估运镜过程的平滑程度和一致性
