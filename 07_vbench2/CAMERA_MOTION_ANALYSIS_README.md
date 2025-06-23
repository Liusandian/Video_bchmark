# 相机运动分析 - 稳定特征点方法

## 概述

本项目实现了基于稳定特征点的相机运动分析方法，能够准确检测视频中的相机运动类型，特别是zoom in/out运动。该方法通过CoTracker提取10×10网格特征点，分析在80%以上帧中都可见的稳定特征点，基于这些点的运动模式来判断相机运动类型。

## 核心功能

### 1. 稳定特征点检测
- **目标**: 找到在至少80%的视频帧中都可见且稳定跟踪的特征点
- **方法**: 分析CoTracker输出的可见性矩阵，计算每个特征点的可见性比例
- **输出**: 稳定特征点的ID列表

### 2. Zoom运动检测
- **原理**: 分析特征点到图像中心的距离变化
- **Zoom In**: 边缘特征点向图像中心收缩，距离减小
- **Zoom Out**: 边缘特征点向外扩散，距离增大
- **特点**: 优先分析边缘点，因为边缘点对zoom运动更敏感

### 3. Pan/Tilt运动检测
- **原理**: 分析特征点的整体运动方向
- **Pan**: 水平方向的相对运动
- **Tilt**: 垂直方向的相对运动

## 使用方法

### 基本用法

```python
from vbench2.camera_motion import CameraPredict, test_stable_point_analysis

# 方法1: 使用测试函数
test_stable_point_analysis(
    video_path="your_video.mp4",
    output_dir="./results",
    device="cuda",
    stability_threshold=0.8
)

# 方法2: 使用类方法
camera = CameraPredict("cuda", submodules_dict)
pred_tracks, pred_visibility = camera.infer(video, fps=30)
results = camera.predict_with_stable_points(pred_tracks, pred_visibility)
```

### 命令行使用

```bash
# 基本使用
python test_camera_motion_analysis.py --video_path /path/to/video.mp4

# 指定参数
python test_camera_motion_analysis.py \
    --video_path /path/to/video.mp4 \
    --output_dir ./my_results \
    --device cuda \
    --stability_threshold 0.7
```

## 参数说明

- `stability_threshold`: 稳定性阈值 (0.0-1.0)
  - 0.8: 80%的帧中可见 (推荐)
  - 0.6: 60%的帧中可见 (更宽松)
  - 0.9: 90%的帧中可见 (更严格)

- `device`: 计算设备
  - "cuda": GPU计算 (推荐)
  - "cpu": CPU计算

## 输出说明

### 控制台输出示例

```
总特征点数: 100, 稳定特征点数: 67 (67.0%)
稳定特征点可见性统计: min=0.82, max=1.00, mean=0.94

Zoom分析结果: zoom_out
Zoom详细信息:
  total_stable_points: 67
  edge_points: 45
  center_points: 22
  edge_zoom_out_ratio: 0.7333
  edge_zoom_in_ratio: 0.0667
  mean_distance_change: 15.32

Pan/Tilt分析结果: ['static']
Pan/Tilt详细信息:
  mean_motion_x: 1.23
  mean_motion_y: -0.87
  motion_magnitude: 1.51

最终运动分类结果: ['zoom_out']
```

### 运动类型

- `zoom_in`: 镜头拉近，特征点向中心收缩
- `zoom_out`: 镜头拉远，特征点向外扩散
- `pan_left`: 左摇镜头
- `pan_right`: 右摇镜头
- `tilt_up`: 上仰镜头
- `tilt_down`: 下俯镜头
- `oblique`: 复合运动 (如zoom + tilt)
- `static`: 静止或微小运动
- `complex`: 复杂运动模式

## 算法优势

1. **高准确性**: 基于稳定特征点，过滤掉不稳定的跟踪结果
2. **强鲁棒性**: 能处理遮挡、光照变化等复杂情况
3. **细粒度分析**: 区分边缘点和中心点，提高zoom检测精度
4. **量化分析**: 提供详细的运动统计信息
5. **可配置性**: 可调节稳定性阈值适应不同场景

## 技术原理

### CoTracker特征点跟踪
- 使用10×10网格，共100个特征点
- 每个特征点有固定ID，保持跨帧一致性
- 输出轨迹坐标 `(T, N, 2)` 和可见性 `(T, N, 1)`

### 稳定性分析
```python
# 计算可见性比例
visibility_ratio = np.mean(pred_visibility > 0.5, axis=0)
# 筛选稳定点
stable_point_ids = np.where(visibility_ratio >= threshold)[0]
```

### Zoom检测算法
```python
# 计算到中心的距离
distances = np.sqrt((points_x - center_x)**2 + (points_y - center_y)**2)
# 分析距离变化
distance_changes = final_distances - initial_distances
# 判断运动类型
if edge_zoom_out_ratio > 0.6: return 'zoom_out'
elif edge_zoom_in_ratio > 0.6: return 'zoom_in'
```

## 注意事项

1. **GPU要求**: 推荐使用CUDA加速，CPU运行较慢
2. **内存使用**: 长视频可能需要较大内存
3. **模型下载**: 首次运行会自动下载CoTracker模型
4. **视频格式**: 支持常见格式 (mp4, avi, mov等)

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 解决: 使用CPU或减少视频分辨率
   - `--device cpu`

2. **没有检测到稳定特征点**
   - 解决: 降低稳定性阈值
   - `--stability_threshold 0.6`

3. **模型下载失败**
   - 解决: 检查网络连接，或手动下载模型

4. **结果不准确**
   - 解决: 调整阈值参数，检查视频质量 