## 运镜检测增强：Zoom/Dolly + 通用子弹时间（基于CoTracker轨迹分析 + 可选人脸姿态）

### 功能概述
在 `camera_motion.py` 的 `CameraPredict` 中新增：
- **人像分割**：优先使用 SegFormer ADE20K，如不可用自动回退为中心掩码
- **前景特征点筛选**：基于人像掩码筛选CoTracker轨迹中的前景区域特征点
- **人脸姿态估计**：使用 MediaPipe Face Mesh 提取人脸 yaw/pitch/roll 角度
- **综合运镜检测**：结合**平均距离变化趋势**和**径向投影**检测运镜类型：
  - `zoom_in` / `zoom_out`：基于前景点到中心距离的线性变化趋势
  - `dolly_in` / `dolly_out`：基于前景点径向运动投影
  - `no_dolly`：无明显运镜模式
- **通用子弹时间检测**：主要基于**CoTracker环形运动轨迹分析**，可选**人脸角度变化**辅助：
  - 支持人像、建筑、风景等各种场景的子弹时间检测
  - `bullet_time_90` / `bullet_time_180` / `bullet_time_270` / `bullet_time_360`：不同角度的子弹时间效果
  - `no_bullet_time`：无子弹时间运镜
  - **新增**：不依赖人脸检测的纯轨迹分析模式
- **增强标准分类**：保留原有运动分类，并自动融合检测到的zoom和子弹时间运镜

### 核心算法
#### 1. 前景特征点提取
- `segment_video_masks(video)`: 为视频每帧生成前景（人物）掩码
- 基于掩码筛选CoTracker特征点，确保分析焦点在前景对象上
- 如前景点不足，智能回退到中心区域或全部可见点

#### 2. 距离变化趋势分析
- 计算前景特征点到画面中心 `(w/2, h/2)` 的平均距离序列
- 使用线性拟合分析距离随时间的变化趋势 `distance_trend`
- 计算首末帧距离变化比例 `distance_change_ratio`

#### 3. 径向投影分析
- 计算特征点运动在径向方向（指向/远离中心）的投影分量
- 统计径向投影的平均值和一致性

#### 4. 人脸姿态估计
- `estimate_face_pose(frame)`: 使用MediaPipe Face Mesh估计单帧人脸姿态
- 提取关键点：鼻尖、下巴、左右眼角、左右嘴角
- 计算三个角度：
  - **Yaw (左右转头)**: 基于面部中心相对画面中心的水平偏移
  - **Pitch (上下点头)**: 基于鼻尖-下巴垂直关系
  - **Roll (侧倾)**: 基于左右眼连线的倾斜角度

#### 5. 通用子弹时间检测算法
- **环形运动分析**: 分析CoTracker特征点相对画面中心的角度变化
- **多尺度一致性检测**: 分析内、中、外三个环形区域的运动一致性
- **径向稳定性分析**: 检测径向运动变化（子弹时间中应保持稳定）
- **角速度一致性**: 分析角速度变化的稳定性
- **运动周期性检测**: 通过自相关函数检测运动的周期性特征
- **可选人脸角度辅助**: 如果检测到人脸，提供额外的角度变化支持
- **综合判断逻辑**:
```
# 主要权重分配
motion_confidence = (
    circular_direction_consistency * 0.40 +    # 环形运动一致性
    radial_stability * 0.25 +                  # 径向稳定性
    angular_consistency * 0.20 +               # 角速度一致性
    region_consistency * 0.15                  # 区域一致性
)

# 人脸辅助加成（如果可用）
if face_angle_support > 0:
    motion_confidence = motion_confidence * 0.8 + face_angle_support * 0.2

# 角度阈值（适应不同场景）
if total_rotation >= 75°:
    if >= 320° → bullet_time_360
    elif >= 240° → bullet_time_270  
    elif >= 150° → bullet_time_180
    else → bullet_time_90
else:
    → no_bullet_time
```

#### 6. Zoom/Dolly 运镜类型判断逻辑
```
if |distance_change_ratio| > 0.05:
    if distance_change_ratio > 0 and distance_trend > 0:
        → zoom_out  # 距离增加且趋势向上
    elif distance_change_ratio < 0 and distance_trend < 0:
        → zoom_in   # 距离减少且趋势向下
    else:
        → dolly_in/out  # 距离变化但趋势不一致，可能是dolly
else:
    基于径向投影判断 dolly_in/out
```

### 新增API
- `infer_with_visibility(video, fps, end_frame)`: 返回 `tracks(T×N×2)` 与 `visibility(T×N)`
- `estimate_face_pose(frame)`: 估计单帧人脸姿态，返回`(yaw, pitch, roll)`角度
- `extract_video_face_poses(video)`: 提取视频每帧的人脸姿态序列
- `detect_dolly_motion(tracks, visibility, h, w, masks)`: 
  - 输入：轨迹、可见性、尺寸、前景掩码
  - 输出：`(motion_type, confidence, details)`
  - details包含距离序列、投影序列、一致性等调试信息
- `detect_bullet_time_motion(tracks, visibility, h, w, face_poses=None)`:
  - 输入：轨迹、可见性、尺寸、可选人脸姿态序列
  - 输出：`(bullet_time_type, confidence, details)`
  - details包含环形运动、区域一致性、周期性等分析数据
  - **通用性**：face_poses为None时，使用纯轨迹分析模式
- `predict_with_segmentation(video, fps, end_frame)`: 综合输出：
  - `standard_motions`: 原有运动类别 + 检测到的zoom和子弹时间
  - `primary_motion`: 主要运镜类型（优先级：子弹时间>zoom/dolly>标准运动）
  - `primary_confidence`: 主要运镜的置信度
  - `motion_type`: zoom_in/zoom_out/dolly_in/dolly_out/no_dolly
  - `motion_confidence`: zoom/dolly置信度
  - `bullet_time_type`: bullet_time_90/180/270/360/no_bullet_time
  - `bullet_time_confidence`: 子弹时间置信度
  - `face_pose_count`: 成功检测到人脸的帧数
  - `foreground_mask_coverage`: 各帧前景覆盖率
- `predict_universal_bullet_time(video, fps, end_frame)`: **新增**通用子弹时间检测：
  - 专门针对建筑、风景等非人像场景
  - 不依赖人脸检测和前景分割
  - 返回标准运动、子弹时间类型和详细分析数据

### 使用方式

#### 1. 通用子弹时间检测（推荐用于建筑、风景场景）
```python
from vbench2.camera_motion import CameraPredict

submodules = {"repo": "facebookresearch/co-tracker", "model": "cotracker_w8"}
camera = CameraPredict(device="cuda", submodules_list=submodules)

# 通用子弹时间检测（不依赖人脸）
result = camera.predict_universal_bullet_time(video, fps=30)
print(f"子弹时间类型: {result['bullet_time_type']}")
print(f"置信度: {result['bullet_time_confidence']:.3f}")

if result['bullet_time_type'] != 'no_bullet_time':
    details = result['bullet_time_details']
    print(f"总角度变化: {details['total_rotation_deg']:.1f}°")
    print(f"环形运动一致性: {details['circular_direction_consistency']:.3f}")
    print(f"径向稳定性: {details['radial_stability']:.3f}")
```

#### 2. 综合检测（人像场景，包含前景分割和人脸分析）
```python
# 综合检测（zoom/dolly + 子弹时间 + 人脸分析）

# 准备 decord 读取的视频张量: video (1,T,C,H,W) float
result = camera.predict_with_segmentation(video, fps=30, end_frame=-1)

# 基本信息
print(f"标准运动: {result['standard_motions']}")
print(f"主要运镜: {result['primary_motion']} (置信度: {result['primary_confidence']:.3f})")
print(f"子弹时间: {result['bullet_time_type']} (置信度: {result['bullet_time_confidence']:.3f})")
print(f"Zoom/Dolly: {result['motion_type']} (置信度: {result['motion_confidence']:.3f})")

# 人脸检测统计
print(f"人脸检测: {result['face_pose_count']}/{result['total_frames']} 帧")
print(f"前景覆盖率: {np.mean(result['foreground_mask_coverage']):.3f}")

# 子弹时间详细分析
if result['bullet_time_type'] != 'no_bullet_time':
    bullet_details = result['bullet_time_details']
    print(f"\n=== 子弹时间分析 ===")
    print(f"Yaw角度变化: {bullet_details['yaw_range']:.1f}°")
    print(f"Pitch角度变化: {bullet_details['pitch_range']:.1f}°")
    print(f"Roll角度变化: {bullet_details['roll_range']:.1f}°")
    print(f"环形运动一致性: {bullet_details['circular_consistency']:.3f}")
    print(f"总角度变化: {bullet_details['total_angle_change']:.1f}°")

# Zoom/Dolly详细分析
if result['motion_type'] != 'no_dolly':
    motion_details = result['motion_details']
    print(f"\n=== Zoom/Dolly分析 ===")
    print(f"距离变化趋势: {motion_details['distance_trend']:.4f}")
    print(f"距离变化比例: {motion_details['distance_change_ratio']:.4f}")
    print(f"径向投影均值: {motion_details['mean_proj']:.4f}")
```

### 调参说明
```python
# 在CameraPredict.__init__()中可调整：
self.dolly_proj_threshold = 0.01      # 径向投影阈值
self.dolly_consistency_threshold = 0.65  # 运动一致性阈值
self.bullet_time_angle_threshold = 15.0  # 子弹时间角度变化阈值（度）
self.bullet_time_consistency_threshold = 0.7  # 子弹时间一致性阈值

# 在detect_dolly_motion()中可调整：
th_dist = 0.05  # 距离变化阈值（判断zoom的主要参数）

# 在detect_bullet_time_motion()中可调整：
angle_90/180/270/360 = 90/180/270/360 - self.bullet_time_angle_threshold  # 各级别角度阈值
```

### 依赖与兼容性
- **必需依赖**：`numpy`, `torch`, `cv2`, `decord`（VBench原有依赖）
- **可选依赖**：
  - `transformers`（`pip install transformers`）- 用于人像分割
  - `mediapipe`（`pip install mediapipe`）- 用于人脸姿态估计（子弹时间检测）
  - `dlib`（`pip install dlib`）- MediaPipe不可用时的备用人脸检测
- **自动降级**：
  - 无transformers时使用中心区域掩码
  - 无MediaPipe时子弹时间检测返回`no_bullet_time`
- **向后兼容**：原有`predict()`方法保持不变
- **其他依赖**：沿用VBench原有配置（co-tracker等）

### 适用场景与注意事项

#### Zoom/Dolly检测
- **适用**：包含明显前景对象（人物）的视频，zoom/dolly运镜明显
- **限制**：纯风景、抽象场景可能降级为中心区域分析
- **建议**：输入视频避免严重变形，保持合理的纵横比
- **调试**：通过`motion_details`查看距离序列和投影数据，便于参数调优

#### 通用子弹时间检测 🌟
- **适用场景**：
  - **建筑场景**：围绕建筑物、雕塑的环形拍摄
  - **风景场景**：山川、海景等自然场景的旋转视角
  - **物体特写**：产品展示、艺术品等的360度展示
  - **人像场景**：人物肖像的环形运镜
  - **城市景观**：街景、广场等的环形航拍
- **技术优势**：
  - **无依赖**：不需要人脸检测、前景分割等预处理
  - **鲁棒性**：基于纯轨迹分析，适应性强
  - **多尺度**：分析内、中、外三个区域的运动一致性
  - **周期性检测**：自动识别运动的周期性特征
- **性能特点**：
  - 最小帧数要求：8帧（相比原来的10帧更宽松）
  - 最小特征点：3个（适应稀疏场景）
  - 置信度阈值：0.6（平衡准确率和召回率）
  - 角度阈值：降低15°，更好适应不同场景
- **最佳实践**：
  - 视频帧率 ≥ 15fps，总时长 ≥ 0.5秒
  - 相机运动相对平滑，避免剧烈抖动
  - 场景中有足够的特征点（纹理丰富）
  - 避免严重的运动模糊
- **调试指标**：
  - `circular_direction_consistency`: 环形运动一致性 > 0.7
  - `radial_stability`: 径向稳定性 > 0.6
  - `region_consistency`: 区域一致性 > 0.5
  - `motion_periodicity`: 运动周期性 > 0.3

#### 传统子弹时间检测（人脸辅助）
- **适用**：包含清晰人脸的视频，提供额外的角度变化验证
- **限制**：人脸检测失败时降级为通用模式
- **优势**：人脸角度变化可以提供额外的置信度支持

