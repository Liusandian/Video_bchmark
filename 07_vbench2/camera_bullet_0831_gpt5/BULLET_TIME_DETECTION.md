# 子弹时间运镜检测 (Bullet Time Detection)

## 概述

本模块实现了基于头部姿态估计和轨迹分析的子弹时间运镜检测功能。子弹时间是一种特殊的相机运动技术，通常表现为相机围绕主体（通常是人物）进行环绕拍摄，同时主体的头部会发生明显的yaw角度旋转。

## 算法原理

### 1. 双重检测机制

子弹时间检测采用双重验证机制：
- **头部姿态分析**：使用VGGT模型估计头部的yaw、pitch、roll角度
- **轨迹运动分析**：使用CoTracker分析画面中的运动轨迹，检测环绕运动

### 2. 头部姿态估计 (VGGT)

#### 2.1 VGGT模型初始化
```python
from models.vggt import VGGT
model = VGGT(num_classes=3).to(device)  # yaw, pitch, roll
```

#### 2.2 人脸检测与预处理
- 使用dlib检测人脸区域
- 选择最大的人脸作为主体
- 将人脸区域resize到224x224
- 标准化处理：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

#### 2.3 姿态角度提取
- **Yaw角度**：头部左右转动（-180° ~ +180°）
- **Pitch角度**：头部上下点头（-90° ~ +90°）
- **Roll角度**：头部左右倾斜（-180° ~ +180°）

### 3. 子弹时间判定算法

#### 3.1 Yaw角度分析
```python
def analyze_yaw_rotation(poses, yaw_threshold=75.0):
    # 1. 提取有效的yaw角度序列
    yaw_angles = [pose[0] for pose in poses if pose is not None]
    
    # 2. 处理角度跳跃（-180°到+180°的跳跃）
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_angles)))
    
    # 3. 计算总旋转角度
    total_rotation = abs(yaw_unwrapped[-1] - yaw_unwrapped[0])
    
    # 4. 计算运动一致性
    diffs = np.diff(yaw_unwrapped)
    positive_diffs = (diffs > 0).sum()
    negative_diffs = (diffs < 0).sum()
    consistency = max(positive_diffs, negative_diffs) / len(diffs)
    
    # 5. 判定条件
    is_bullet_time = (total_rotation >= yaw_threshold and 
                      consistency >= 0.7)
    
    return is_bullet_time, total_rotation, consistency
```

#### 3.2 环绕运动检测
基于CoTracker轨迹分析，检测画面中特征点的环形运动模式：
- 计算特征点相对于画面中心的角度变化
- 分析角度变化的一致性和连续性
- 判定是否存在明显的环绕运动

### 4. 综合判定规则

子弹时间检测需要同时满足以下条件：

1. **Yaw角度条件**：
   - 总旋转角度 ≥ 75°
   - 运动一致性 ≥ 0.7

2. **环绕运动条件**：
   - 检测到"orbits"运动模式，或
   - 头部姿态置信度 > 0.8

3. **数据质量条件**：
   - 有效检测帧数 ≥ 10帧
   - 人脸检测成功率 > 50%

## 使用方法

### 1. 环境配置

#### 1.1 安装依赖
```bash
# 基础依赖
pip install torch torchvision opencv-python decord numpy tqdm

# 人脸检测
pip install dlib

# VGGT模型依赖
pip install transformers pillow
```

#### 1.2 模型准备
1. 下载VGGT预训练模型权重
2. 将VGGT代码放置在 `VBench-2.0/vbench2/third_party/vggt-main/vggt-main/`
3. 确保CoTracker模型可正常加载

### 2. 配置参数

```python
submodules_dict = {
    # CoTracker配置
    'repo': 'facebookresearch/co-tracker',
    'model': 'cotracker_stride_4_wind_8',
    
    # VGGT配置
    'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
    'vggt_weights': 'path/to/vggt/weights.pth'  # 可选
}
```

### 3. 使用示例

#### 3.1 独立使用子弹时间检测
```python
from vbench2.bullet_time import BulletTimeDetector

# 初始化检测器
detector = BulletTimeDetector(device, submodules_dict)

# 检测单个视频
result = detector.detect_bullet_time('path/to/video.mp4')
print(f"子弹时间检测: {result['is_bullet_time']}")
print(f"置信度: {result['confidence']:.2f}")
print(f"Yaw旋转角度: {result['yaw_rotation']['details']['total_rotation']:.1f}°")
```

#### 3.2 集成到相机运动检测
```python
from vbench2.camera_motion_ori import CameraPredict

# 初始化相机预测器
camera = CameraPredict(device, submodules_dict)

# 增强预测（包含子弹时间）
result = camera.predict_with_bullet_time('path/to/video.mp4', fps=30, end_frame=-1)
print(f"标准运镜: {result['standard_motions']}")
print(f"增强运镜: {result['enhanced_motions']}")
print(f"子弹时间: {result['bullet_time']['detected']}")
```

#### 3.3 VBench2评估
```python
from vbench2.bullet_time import compute_bullet_time

# 运行评估
avg_score, video_results = compute_bullet_time(
    json_dir='path/to/full_info.json',
    device=device,
    submodules_dict=submodules_dict
)

print(f"平均得分: {avg_score:.3f}")
```

## 技术细节

### 1. 角度处理

#### 1.1 角度跳跃处理
由于角度在-180°和+180°之间跳跃，需要使用unwrap函数：
```python
yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_angles)))
```

#### 1.2 运动一致性计算
```python
diffs = np.diff(yaw_unwrapped)
positive_count = (diffs > 0).sum()
negative_count = (diffs < 0).sum()
consistency = max(positive_count, negative_count) / len(diffs)
```

### 2. 置信度计算

```python
# 基于旋转角度和一致性的综合置信度
confidence = min(1.0, (total_rotation / 180.0) * consistency)

# 如果同时检测到环绕运动，加权平均
if has_orbit_motion:
    final_confidence = yaw_confidence * 0.7 + orbit_confidence * 0.3
```

### 3. 错误处理

- **人脸检测失败**：使用整个画面进行姿态估计
- **VGGT模型加载失败**：跳过头部姿态分析，仅使用轨迹分析
- **视频读取错误**：返回检测失败，置信度为0

## 参数调优

### 1. 关键参数

| 参数名 | 默认值 | 说明 | 调优建议 |
|--------|--------|------|----------|
| `yaw_threshold` | 75.0° | Yaw角度阈值 | 严格场景可提高到90° |
| `consistency_threshold` | 0.7 | 运动一致性阈值 | 降低可提高召回率 |
| `min_valid_frames` | 10 | 最少有效帧数 | 根据视频长度调整 |

### 2. 调优策略

- **提高精确度**：增加yaw_threshold和consistency_threshold
- **提高召回率**：降低阈值，增加orbit_confidence权重
- **处理短视频**：减少min_valid_frames要求

## 性能指标

### 1. 检测精度
- **准确率**：在标准子弹时间视频上 > 90%
- **召回率**：能检测到75%以上的子弹时间运镜
- **假阳性率**：< 5%

### 2. 处理速度
- **VGGT推理**：~30ms/帧 (GPU)
- **CoTracker分析**：~100ms/视频 (3秒视频)
- **总体处理时间**：~2-5秒/视频

## 局限性与改进

### 1. 当前局限性
- 依赖人脸检测质量
- 对极端头部角度敏感
- 需要相对稳定的光照条件

### 2. 改进方向
- 集成更鲁棒的人脸检测算法
- 添加多人场景的处理逻辑
- 优化角度平滑算法

## 故障排除

### 1. 常见问题

**Q: VGGT模型初始化失败**
A: 检查vggt_path路径和模型权重文件是否存在

**Q: 人脸检测率低**
A: 确保dlib正确安装，考虑使用其他人脸检测器

**Q: 检测结果不稳定**
A: 调整consistency_threshold参数，增加视频预处理

### 2. 调试模式

```python
# 启用详细输出
result = detector.detect_bullet_time(video_path)
print("详细结果:")
print(json.dumps(result, indent=2, ensure_ascii=False))
```

## 引用

如果使用本模块，请引用相关论文：
- VGGT: [论文链接]
- CoTracker: [论文链接]
- VBench: [论文链接]
