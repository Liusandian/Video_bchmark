# 基于深度估计的相机运动检测系统

## 概述

本系统是对VBench camera motion维度评测的重大改进，通过集成深度估计技术来分离前景背景，仅基于背景特征点计算运镜类型，从而显著提高检测准确性。

## 🎯 核心优势

### 1. **深度估计增强**
- 使用MiDaS/DPT模型进行相对深度估计
- 自动分离前景运动物体和背景静态场景
- 避免前景运动干扰相机运动检测

### 2. **背景特征点分析**
- 仅使用背景区域的特征点进行运镜分析
- 重新计算边缘关键特征点的运动方向
- 更准确的上下左右边缘运动检测

### 3. **增强的运镜类型检测**
- 支持传统运镜：pan, tilt, zoom, static
- 新增高级运镜：dolly, orbits, oblique
- 复合运镜检测：dolly+zoom组合

### 4. **详细的召回率分析**
- 每种运镜类型的召回率统计
- 深度方法使用率分析
- 性能改进建议生成

## 📁 文件结构

```
├── camera_motion_with_depth.py              # 主要的深度增强检测模块
├── depth_camera_motion_recall_analysis.py   # 召回率分析工具
├── README_depth_camera_motion.md            # 使用说明文档
└── examples/                                # 使用示例
```

## 🚀 快速开始

### 1. 环境依赖

```bash
# 基础依赖
pip install torch torchvision opencv-python numpy scipy tqdm decord

# 深度估计依赖
pip install timm  # 用于MiDaS模型

# VBench依赖
# 确保已安装VBench相关依赖
```

### 2. 基本使用

```python
from camera_motion_with_depth import DepthBasedCameraPredict
from depth_camera_motion_recall_analysis import enhanced_camera_motion_evaluation_with_depth

# 配置
device = "cuda"  # 或 "cpu"
submodules_dict = {
    "repo": "facebookresearch/co-tracker",
    "model": "cotracker2_online"
}

# 评测单个视频
camera = DepthBasedCameraPredict(device, submodules_dict)
results = camera.predict(video_tensor, fps=30, end_frame=-1)
print(f"检测到的运镜类型: {results}")

# 批量评测与召回率分析
json_dir = "path/to/vbench/data"
score, detailed_results = enhanced_camera_motion_evaluation_with_depth(
    json_dir, device, submodules_dict, 
    save_visualizations=True,
    detailed_output=True
)
```

### 3. 测试功能

```python
from camera_motion_with_depth import test_depth_based_camera_motion

# 测试单个视频的深度估计效果
test_depth_based_camera_motion(
    video_path="test_video.mp4",
    output_dir="./test_results",
    device="cuda"
)
```

## 🔧 核心组件

### 1. DepthBasedCameraPredict 类

```python
class DepthBasedCameraPredict:
    def __init__(self, device, submodules_list):
        # 初始化CoTracker和深度估计模型
        
    def estimate_depth(self, frame):
        # 估计单帧深度图
        
    def filter_background_feature_points(self, pred_tracks, pred_visibility, video_frames):
        # 基于深度分离前景背景特征点
        
    def background_camera_classify(self, pred_tracks, background_point_ids, video_frames):
        # 基于背景特征点进行运镜分类
        
    def predict(self, video, fps, end_frame):
        # 主预测函数
```

### 2. 深度估计模型

支持的模型：
- **MiDaS** (推荐)：轻量级，速度快
- **DPT-Large**：精度高，计算量大

自动模型选择机制：
```python
# 优先级顺序
1. MiDaS -> 2. DPT -> 3. 传统方法回退
```

### 3. 背景分割策略

#### 自适应阈值
```python
# 基于深度直方图的双峰检测
hist, bins = np.histogram(depth_normalized.flatten(), bins=50)
peaks = find_peaks(hist, height=np.max(hist) * 0.1)
threshold = bins[peaks[0]] + (bins[peaks[1]] - bins[peaks[0]]) * 0.3
```

#### 形态学后处理
```python
# 去除噪声并平滑边界
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
```

## 📊 召回率分析功能

### 1. 详细指标计算

```python
recall_metrics = {
    'overall_accuracy': 0.85,           # 总体准确率
    'depth_method_usage_rate': 0.78,    # 深度方法使用率
    'traditional_fallback_rate': 0.22,  # 传统方法回退率
    'per_type_recall': {                # 各类型召回率
        'pan_left': 0.90,
        'zoom_in': 0.85,
        'dolly_out': 0.75
    }
}
```

### 2. 性能分析报告

```python
# 生成详细的性能报告
print_depth_camera_recall_report(recall_metrics, detailed=True)

# 输出示例：
🎬 基于深度估计的相机运动检测召回率报告
============================================================
📊 总体性能指标:
   总体准确率: 0.850
   深度方法使用率: 0.780
   传统方法回退率: 0.220

🎯 各运镜类型召回率:
   pan_left        | 召回率: 0.900 | 深度使用: 0.850 | 🟢 优秀
   zoom_in         | 召回率: 0.850 | 深度使用: 0.800 | 🟢 优秀
   dolly_out       | 召回率: 0.750 | 深度使用: 0.650 | 🟡 良好
```

### 3. 改进建议生成

```python
💡 改进建议:
   1. 🔥 考虑优化深度估计模型或阈值参数，特别针对这些运镜类型
      影响类型: tilt_up, static
   2. ⚠️  这些运动类型理论上应该从深度估计中获益更多，建议检查前景背景分割质量
      影响类型: zoom_out, dolly_in
```

## 🎬 支持的运镜类型

### 传统运镜
- **pan_left/pan_right**: 水平摇摄
- **tilt_up/tilt_down**: 垂直俯仰
- **zoom_in/zoom_out**: 推拉镜头
- **static**: 静态镜头

### 高级运镜
- **dolly_in/dolly_out**: 移动推拉
- **orbits**: 环绕运动
- **oblique**: 斜向运动

### 复合运镜
- **dolly_in_zoom_out**: 前移后拉（希区柯克效果）
- **dolly_out_zoom_in**: 后拉前推
- **hitchcock_zoom**: 经典变焦效果

## 🔍 可视化功能

系统提供了丰富的可视化功能：

### 1. 深度增强可视化
```python
# 保存包含深度信息的可视化视频
camera.infer(video, fps, end_frame, save_video=True, save_dir="./visualizations")
```

特征：
- 绿色点：背景特征点
- 红色点：前景特征点
- 绿色半透明覆盖：背景区域
- 实时统计信息显示

### 2. 特征点轨迹可视化
- 轨迹线连接：显示特征点运动轨迹
- 网格可视化：显示网格特征点分布
- 边缘点高亮：突出显示边缘区域特征点

## ⚙️ 参数配置

### 深度估计参数
```python
# 在初始化时配置
camera = DepthBasedCameraPredict(device, submodules_dict)
camera.depth_threshold_percentile = 30     # 深度阈值百分位数
camera.use_adaptive_threshold = True       # 是否使用自适应阈值
camera.background_erosion_size = 3         # 背景mask腐蚀kernel大小
```

### 运动检测参数
```python
# 运动阈值（相对于最小分辨率）
motion_threshold = min(height, width) * 0.02

# 边缘区域阈值
edge_threshold = min(height, width) * 0.15

# 置信度阈值
confidence_threshold = 0.6
```

## 📈 性能优化建议

### 1. 硬件配置
- **推荐GPU**: RTX 3080及以上
- **显存要求**: 至少8GB
- **CPU**: 支持多核并行处理

### 2. 模型选择策略
```python
# 根据硬件条件选择模型
if gpu_memory >= 8:
    model_type = "DPT_Large"    # 高精度
else:
    model_type = "MiDaS"        # 平衡性能
```

### 3. 批处理优化
```python
# 对于大量视频的批量处理
batch_size = min(4, available_gpu_memory // 2)
```

## 🔧 故障排除

### 常见问题

#### 1. 深度估计模型加载失败
```python
# 错误信息：深度估计模型加载失败
# 解决方案：
pip install timm transformers
# 或手动下载模型文件
```

#### 2. 内存不足
```python
# 减少批处理大小或使用CPU模式
device = "cpu"
torch.cuda.empty_cache()
```

#### 3. 深度方法使用率过低
```python
# 调整深度阈值参数
camera.depth_threshold_percentile = 40  # 增加到40
camera.use_adaptive_threshold = False   # 禁用自适应
```

## 📋 使用示例

### 完整评测流程
```python
from camera_motion_with_depth import *
from depth_camera_motion_recall_analysis import *

# 1. 配置参数
json_dir = "VBench-2.0/prompts"
device = "cuda"
submodules_dict = {
    "repo": "facebookresearch/co-tracker",
    "model": "cotracker2_online"
}

# 2. 执行评测
score, results = enhanced_camera_motion_evaluation_with_depth(
    json_dir=json_dir,
    device=device,
    submodules_dict=submodules_dict,
    save_visualizations=True,
    detailed_output=True
)

# 3. 保存结果
save_depth_camera_analysis_report(results, "analysis_report.json")

print(f"总体得分: {score:.3f}")
print(f"深度方法使用率: {results['recall_metrics']['depth_method_usage_rate']:.3f}")
```

### 单视频测试
```python
# 测试单个视频
video_path = "test_video.mp4"
output_dir = "./test_output"

test_depth_based_camera_motion(video_path, output_dir, device="cuda")
```

## 🚀 未来改进方向

1. **多帧深度融合**: 使用多帧信息提高深度估计精度
2. **语义分割集成**: 结合语义分割进一步细化前景背景分离
3. **实时处理优化**: 针对实时应用的性能优化
4. **自适应参数调整**: 根据视频内容自动调整参数
5. **更多复合运镜**: 支持更复杂的电影运镜技法检测

## 📞 技术支持

如有问题或建议，请参考：
1. 查看详细的错误日志
2. 检查GPU内存使用情况
3. 验证输入视频格式
4. 确认模型文件完整性

---

**注意**: 本系统需要稳定的网络连接来下载深度估计模型，首次使用时请确保网络畅通。 