## Zoom/Dolly 运镜检测增强（基于前景人像分割 + CoTracker）

### 功能概述
在 `camera_motion.py` 的 `CameraPredict` 中新增：
- **人像分割**：优先使用 SegFormer ADE20K，如不可用自动回退为中心掩码
- **前景特征点筛选**：基于人像掩码筛选CoTracker轨迹中的前景区域特征点
- **综合运镜检测**：结合**平均距离变化趋势**和**径向投影**检测运镜类型：
  - `zoom_in` / `zoom_out`：基于前景点到中心距离的线性变化趋势
  - `dolly_in` / `dolly_out`：基于前景点径向运动投影
  - `no_dolly`：无明显运镜模式
- **增强标准分类**：保留原有运动分类，并自动融合检测到的zoom运镜

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

#### 4. 运镜类型判断逻辑
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
- `detect_dolly_motion(tracks, visibility, h, w, masks)`: 
  - 输入：轨迹、可见性、尺寸、前景掩码
  - 输出：`(motion_type, confidence, details)`
  - details包含距离序列、投影序列、一致性等调试信息
- `predict_with_segmentation(video, fps, end_frame)`: 综合输出：
  - `standard_motions`: 原有运动类别 + 检测到的zoom（如置信度>0.6）
  - `motion_type`: zoom_in/zoom_out/dolly_in/dolly_out/no_dolly
  - `motion_confidence`: [0,1] 置信度
  - `motion_details`: 详细分析数据
  - `foreground_mask_coverage`: 各帧前景覆盖率

### 使用方式
```python
from vbench2.camera_motion import CameraPredict

submodules = {"repo": "facebookresearch/co-tracker", "model": "cotracker_w8"}
camera = CameraPredict(device="cuda", submodules_list=submodules)

# 准备 decord 读取的视频张量: video (1,T,C,H,W) float
result = camera.predict_with_segmentation(video, fps=30, end_frame=-1)
print(f"标准运动: {result['standard_motions']}")
print(f"检测运镜: {result['motion_type']} (置信度: {result['motion_confidence']:.3f})")
print(f"前景覆盖率: {np.mean(result['foreground_mask_coverage']):.3f}")

# 获取详细分析数据
details = result['motion_details']
print(f"距离变化趋势: {details['distance_trend']:.4f}")
print(f"距离变化比例: {details['distance_change_ratio']:.4f}")
print(f"径向投影均值: {details['mean_proj']:.4f}")
```

### 调参说明
```python
# 在CameraPredict.__init__()中可调整：
self.dolly_proj_threshold = 0.01      # 径向投影阈值
self.dolly_consistency_threshold = 0.65  # 运动一致性阈值

# 在detect_dolly_motion()中可调整：
th_dist = 0.05  # 距离变化阈值（判断zoom的主要参数）
```

### 依赖与兼容性
- **可选依赖**：`transformers`（`pip install transformers`）
- **自动降级**：无transformers时使用中心区域掩码
- **向后兼容**：原有`predict()`方法保持不变
- **其他依赖**：沿用VBench原有配置（decord、torch、co-tracker等）

### 适用场景与注意事项
- **适用**：包含明显前景对象（人物）的视频，zoom/dolly运镜明显
- **限制**：纯风景、抽象场景可能降级为中心区域分析
- **建议**：输入视频避免严重变形，保持合理的纵横比
- **调试**：通过`motion_details`查看距离序列和投影数据，便于参数调优

