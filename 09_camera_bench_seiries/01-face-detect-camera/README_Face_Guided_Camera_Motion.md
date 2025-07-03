# Face-Guided Camera Motion Analysis

本项目集成了DBFace人脸检测算法到VBench-2.0的相机运动分析中，通过分析人脸占画面比例的变化来更准确地判断zoom in/out运镜方式。

## 🌟 功能特点

- **智能人脸检测**: 使用DBFace模型检测视频中的人脸
- **精确运镜判断**: 结合人脸比例变化分析zoom in/out
- **多重分析融合**: 传统光流分析 + 人脸检测分析
- **用户友好建议**: 提供清晰的运镜方式推荐和置信度评估

## 📋 主要改进

### 1. 集成DBFace人脸检测
- 在`CameraPredict`类中集成DBFace模型
- 支持单帧和视频序列的人脸检测
- 自动处理模型加载和GPU加速

### 2. 人脸比例分析
- 计算视频首尾帧人脸面积占比
- 基于比例变化判断zoom in/out
- 提供置信度评估

### 3. 综合分析系统
- 传统运镜检测 + 人脸分析
- 智能结果融合和优先级判断
- 详细的分析报告和建议

## 🚀 快速开始

### 1. 环境准备

确保已安装以下依赖：
```bash
pip install torch torchvision opencv-python decord numpy
```

### 2. 模型文件

确保DBFace模型文件位于正确位置：
```
VBench-2.0/vbench2/third_party/DBFace-master/model/dbface.pth
```

### 3. 基本使用

```python
from vbench2.camera_motion import analyze_single_video_with_face_guidance
import torch

# 配置
device = "cuda" if torch.cuda.is_available() else "cpu"
submodules_dict = {"repo": "facebook/co-tracker", "model": "cotracker2"}

# 分析视频
results = analyze_single_video_with_face_guidance(
    video_path="path/to/your/video.mp4",
    device=device,
    submodules_dict=submodules_dict
)

# 查看结果
print(results["user_guidance"]["message"])
for suggestion in results["user_guidance"]["suggestions"]:
    print(f"  - {suggestion}")
```

### 4. 使用示例脚本

```bash
# 分析单个视频
python example_face_guided_camera_analysis.py --video path/to/video.mp4

# 运行演示
python example_face_guided_camera_analysis.py --demo

# 指定设备
python example_face_guided_camera_analysis.py --video path/to/video.mp4 --device cuda
```

## 📊 分析结果说明

### 结果结构
```python
{
    "video_path": "视频路径",
    "traditional_motion_analysis": ["传统分析结果"],
    "face_analysis": {
        "face_motion": "zoom_in/zoom_out/None",
        "confidence": 0.85,
        "start_ratio": 0.05,
        "end_ratio": 0.12,
        "ratio_change_percent": 140.0,
        "start_faces_count": 1,
        "end_faces_count": 1
    },
    "final_recommendations": [
        {
            "motion_type": "zoom_in",
            "confidence": "High",
            "source": "Face Analysis",
            "description": "Based on face detection: Face area changed by +140.0%"
        }
    ],
    "user_guidance": {
        "message": "Primary motion detected: zoom_in",
        "suggestions": ["Recommended: zoom_in"]
    }
}
```

### 置信度级别
- **High (高)**: 置信度 > 0.7，强烈推荐
- **Medium (中)**: 置信度 0.4-0.7，一般推荐  
- **Low (低)**: 置信度 < 0.4，仅供参考

## 🔧 核心功能详解

### 1. 人脸检测 (`detect_faces_in_frame`)
- 使用DBFace模型检测单帧中的人脸
- 支持置信度和NMS阈值调整
- 返回人脸边界框和关键点信息

### 2. 人脸比例计算 (`calculate_face_ratio`)
- 计算人脸面积占总画面的比例
- 支持多人脸场景
- 提供归一化的比例值

### 3. 运镜分析 (`analyze_face_zoom_motion`)
- 对比视频首尾帧的人脸比例
- 计算比例变化百分比
- 基于阈值判断zoom in/out

### 4. 综合分析 (`get_camera_motion_recommendations`)
- 融合传统光流分析和人脸分析
- 提供详细的分析报告
- 生成用户友好的建议

## 📈 优势对比

| 特性 | 传统方法 | 人脸引导方法 |
|------|----------|------------|
| 准确性 | 中等 | 高 |
| 语义理解 | 无 | 有 |
| Zoom检测 | 一般 | 优秀 |
| 干扰抗性 | 弱 | 强 |
| 计算复杂度 | 低 | 中等 |

## ⚙️ 参数调优

### 人脸检测参数
```python
# 在 detect_faces_in_frame 中调整
threshold=0.4        # 人脸检测置信度阈值
nms_iou=0.5         # NMS IoU阈值
```

### 运镜判断参数
```python
# 在 analyze_face_zoom_motion 中调整
ratio_change_threshold = 20  # 比例变化阈值（百分比）
confidence_threshold = 0.3   # 置信度阈值
```

## 🚨 注意事项

1. **模型文件**: 确保DBFace模型文件(`dbface.pth`)存在于指定路径
2. **GPU内存**: 人脸检测需要额外的GPU内存，建议4GB以上
3. **视频质量**: 低质量或模糊的视频会影响人脸检测效果
4. **多人场景**: 多人脸场景会计算所有人脸的总面积
5. **光照条件**: 极暗或过亮的场景可能影响检测准确性

## 🛠️ 故障排除

### 常见问题

**Q: 提示"DBFace model not found"**
A: 检查模型文件路径是否正确，确保`dbface.pth`存在

**Q: 人脸检测失败**
A: 检查视频中是否有清晰的人脸，尝试调低置信度阈值

**Q: 分析结果不准确**
A: 可能需要调整比例变化阈值或置信度阈值

**Q: GPU内存不足**
A: 尝试使用CPU模式或减小视频分辨率

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个功能！

## 📄 许可证

遵循VBench项目的许可证协议。 