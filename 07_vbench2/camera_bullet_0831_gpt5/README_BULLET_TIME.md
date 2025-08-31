# 子弹时间运镜检测 - 快速开始

## 概述
本项目实现了基于VGGT头部姿态估计和CoTracker轨迹分析的子弹时间运镜检测功能。通过分析视频中人物头部的yaw角度旋转和相机的环绕运动来识别子弹时间效果。

## 核心文件
- `bullet_time.py` - 主要检测模块
- `camera_motion_ori.py` - 增强的相机运动检测（集成子弹时间）
- `example_bullet_time.py` - 使用示例
- `BULLET_TIME_DETECTION.md` - 详细技术文档

## 快速使用

### 1. 环境准备
```bash
pip install torch torchvision opencv-python decord numpy tqdm dlib
```

### 2. 配置VGGT
将VGGT模型代码放置在：
```
VBench-2.0/vbench2/third_party/vggt-main/vggt-main/
```

### 3. 基本使用
```python
from vbench2.bullet_time import BulletTimeDetector

# 配置
submodules_dict = {
    'repo': 'facebookresearch/co-tracker',
    'model': 'cotracker_stride_4_wind_8',
    'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main'
}

# 检测
detector = BulletTimeDetector(device, submodules_dict)
result = detector.detect_bullet_time('video.mp4')

print(f"子弹时间: {result['is_bullet_time']}")
print(f"置信度: {result['confidence']:.2f}")
```

### 4. 集成使用
```python
from vbench2.camera_motion_ori import CameraPredict

camera = CameraPredict(device, submodules_dict)
result = camera.predict_with_bullet_time('video.mp4', fps=30, end_frame=-1)

print(f"运镜类型: {result['enhanced_motions']}")
print(f"子弹时间: {result['bullet_time']['detected']}")
```

## 检测原理
1. **头部姿态分析**：使用VGGT提取yaw、pitch、roll角度
2. **Yaw旋转检测**：计算总旋转角度，要求≥75°且运动一致性≥0.7
3. **环绕运动检测**：使用CoTracker分析特征点轨迹
4. **综合判定**：头部旋转 + 环绕运动 = 子弹时间

## 参数调整
- `yaw_threshold=75.0` - yaw角度阈值
- `consistency_threshold=0.7` - 运动一致性阈值  
- `min_valid_frames=10` - 最少有效帧数

## 运行示例
```bash
python VBench-2.0/vbench2/example_bullet_time.py
```

详细文档请参考 `BULLET_TIME_DETECTION.md`
