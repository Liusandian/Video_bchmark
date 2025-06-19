# CoTracker特征点可视化

这个模块提供了将CoTracker模型返回的特征点可视化并使用OpenCV保存的功能。

## 功能特点

- ✅ 支持两种可视化模式：
  - **网格模式 (grid)**: 显示网格形式的特征点和连线
  - **轨迹模式 (tracks)**: 显示独立的特征点轨迹
- ✅ 使用OpenCV进行高效的视频处理和保存
- ✅ 支持特征点轨迹的连续显示
- ✅ 自动过滤不可见的特征点
- ✅ 在视频上叠加信息文本（帧数、可见点数等）

## 快速开始

### 1. 使用示例脚本

```bash
# 基本使用
python example_cotracker_visualization.py --video_path /path/to/your/video.mp4

# 指定输出目录和可视化类型
python example_cotracker_visualization.py \
    --video_path /path/to/your/video.mp4 \
    --output_dir ./my_visualizations \
    --visualization_type grid \
    --device cuda
```

### 2. 在代码中直接调用

```python
from vbench2.camera_motion import visualize_camera_motion

# 可视化单个视频
visualize_camera_motion(
    video_path="input_video.mp4",
    output_dir="./visualizations",
    device="cuda",
    visualization_type="grid"
)
```

### 3. 集成到相机运动评估中

```python
from vbench2.camera_motion import compute_camera_motion

# 在评估过程中同时保存可视化结果
results, video_results = compute_camera_motion(
    json_dir="./prompts",
    device="cuda",
    submodules_dict={
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    },
    save_visualizations=True  # 启用可视化保存
)
```

## 可视化模式说明

### 网格模式 (grid)
- 显示10x10网格形式的特征点
- 用绿色圆点表示特征点
- 用蓝色线条连接相邻的网格点
- 适合查看整体的运动模式

### 轨迹模式 (tracks)
- 显示每个特征点的轨迹
- 不同特征点使用不同颜色
- 显示特征点的历史轨迹路径
- 适合查看详细的运动轨迹

## 输出文件

可视化结果会保存为 `cotracker_visualization.mp4` 文件，包含：
- 原始视频帧
- 叠加的特征点
- 特征点轨迹（轨迹模式）或网格连线（网格模式）
- 帧信息文本
- 可见特征点数量

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `video_path` | str | - | 输入视频文件路径 |
| `output_dir` | str | "./visualizations" | 输出目录 |
| `device` | str | "cuda" | 计算设备 ("cuda" 或 "cpu") |
| `visualization_type` | str | "grid" | 可视化类型 ("grid" 或 "tracks") |
| `fps` | int | 30 | 输出视频帧率 |

## 系统要求

- Python 3.7+
- PyTorch
- OpenCV (cv2)
- decord
- numpy
- tqdm

## 性能优化建议

1. **使用GPU**: 如果有CUDA设备，建议使用 `device="cuda"` 以获得更好的性能
2. **调整帧率**: 根据需要调整输出视频的帧率
3. **批处理**: 对于多个视频，可以使用集成的评估函数进行批处理

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案: 使用 device="cpu" 或者减小输入视频的分辨率
   ```

2. **CoTracker模型加载失败**
   ```
   解决方案: 检查网络连接，确保能访问torch.hub
   ```

3. **输出视频无法播放**
   ```
   解决方案: 检查OpenCV是否正确安装，尝试不同的视频编解码器
   ```

## 示例输出

可视化结果将显示：
- 原始视频内容
- 覆盖的特征点（绿色圆点）
- 特征点之间的连线（网格模式）或轨迹（轨迹模式）
- 当前帧数和可见特征点数量

## 自定义开发

如果需要自定义可视化效果，可以修改 `CameraPredict` 类中的以下方法：
- `visualize_and_save_tracks()`: 自定义轨迹模式的可视化
- `visualize_grid_tracks()`: 自定义网格模式的可视化

例如，修改颜色、点的大小、线条粗细等：

```python
# 修改特征点颜色和大小
cv2.circle(frame, (x, y), radius=5, color=(0, 255, 255), thickness=-1)

# 修改连线样式
cv2.line(frame, pt1, pt2, color=(255, 0, 0), thickness=3)
``` 