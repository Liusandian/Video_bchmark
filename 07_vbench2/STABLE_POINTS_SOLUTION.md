# 稳定特征点维度冲突问题解决方案

## 问题描述

在相机运动分析中，当使用稳定特征点交集进行边缘运动分析时，遇到了**reshape维度不匹配**的问题：

- **期望维度**: 10×10 = 100个特征点（网格结构）
- **实际维度**: 72+个稳定特征点（筛选后的子集）
- **冲突**: 无法将72个点reshape为(10, 10, 2)的网格

## 问题根源

### 原有方法的限制
```python
# ❌ 原有方法依赖固定网格结构
track = pred_tracks[frame_idx].reshape(10, 10, 2)  # 维度冲突！
top = [track[0, i, :] for i in range(start, end)]   # 固定索引访问
```

原有的`get_edge_point()`方法假设：
1. 特征点总数固定为100个（10×10网格）
2. 可以通过网格索引访问边缘点（如`track[0, i, :]`表示顶部边缘）
3. 每个边缘区域的点数固定

## 解决方案

### 核心思路
**不依赖固定网格结构，改为基于坐标位置的动态分类**

### 新增方法

#### 1. 稳定特征点边缘提取
```python
def get_edge_point_from_stable_points(self, pred_tracks, stable_point_ids, frame_idx):
    """基于坐标位置提取边缘点，无需reshape"""
    stable_points = pred_tracks[frame_idx, stable_point_ids, :]  # (N_stable, 2)
    edge_threshold = min(self.width, self.height) * 0.15
    
    # 基于坐标动态分类
    top_points = [point for point in stable_points if point[1] <= edge_threshold]
    down_points = [point for point in stable_points if point[1] >= self.height - edge_threshold]
    # ... 左右边缘类似
```

#### 2. 边缘运动方向分析
```python
def get_edge_direction_from_stable_points(self, pred_tracks, stable_point_ids, frame1_idx, frame2_idx):
    """分析稳定特征点的边缘运动方向"""
    edge_points1 = self.get_edge_point_from_stable_points(pred_tracks, stable_point_ids, frame1_idx)
    edge_points2 = self.get_edge_point_from_stable_points(pred_tracks, stable_point_ids, frame2_idx)
    # 计算运动向量并分类
```

#### 3. 相机运动分类
```python
def camera_classify_from_stable_points(self, pred_tracks, stable_point_ids):
    """基于稳定特征点进行相机运动分类"""
    top, down, left, right = self.get_edge_direction_from_stable_points(...)
    # 使用现有的分类逻辑
```

### 关键优势

| 特性 | 原有方法 | 新方法 |
|------|----------|--------|
| **维度要求** | 固定100个点(10×10) | 任意数量稳定点 |
| **边缘定义** | 网格索引(第0行=顶部) | 坐标位置(y≤阈值=顶部) |
| **点数灵活性** | 固定每边缘2个点 | 动态数量，最少保证2个 |
| **适应性** | 仅适用完整网格 | 适用任意特征点集合 |

## 使用方法

### 集成到现有流程
```python
def predict_with_stable_points(self, pred_tracks, pred_visibility, stability_threshold=0.8):
    # 1. 找到稳定特征点
    stable_point_ids = self.find_stable_feature_points(pred_visibility, stability_threshold)
    
    # 2. 使用新方法分析边缘运动（解决维度问题）
    edge_results = self.camera_classify_from_stable_points(pred_tracks, stable_point_ids)
    
    # 3. 结合其他分析方法
    zoom_results = self.analyze_zoom_motion(pred_tracks, stable_point_ids)
    pan_tilt_results = self.analyze_pan_tilt_motion(pred_tracks, stable_point_ids)
    
    # 4. 融合所有结果
    return self.merge_motion_results(edge_results, zoom_results, pan_tilt_results)
```

### 测试新功能
```python
# 测试稳定特征点边缘分析
from vbench2.camera_motion import test_stable_point_edge_analysis

test_stable_point_edge_analysis(
    video_path="your_video.mp4",
    output_dir="./analysis_results",
    device="cuda",
    stability_threshold=0.8
)
```

## 技术细节

### 边缘区域定义
- **顶部边缘**: `y ≤ height × 0.15`
- **底部边缘**: `y ≥ height × 0.85`  
- **左侧边缘**: `x ≤ width × 0.15`
- **右侧边缘**: `x ≥ width × 0.85`

### 点数保证机制
当某个边缘区域点数不足时，使用距离排序补充：
```python
if len(top_points) < min_points_per_edge:
    y_coords = stable_points[:, 1]
    top_indices = np.argsort(y_coords)[:min_points_per_edge]
    top_points = [stable_points[i] for i in top_indices]
```

## 向后兼容性

- 保留原有方法作为后备方案
- 新方法优先使用，失败时自动回退
- 保持相同的输出格式和接口

## 总结

通过**坐标位置分类**替代**网格索引访问**，彻底解决了稳定特征点数量与固定网格维度不匹配的问题，同时保持了算法的准确性和灵活性。 