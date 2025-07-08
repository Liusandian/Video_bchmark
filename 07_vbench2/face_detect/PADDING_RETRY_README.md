# DBFace Padding重试功能说明

## 功能简介

本次更新为DBFace人脸检测模型添加了**Padding重试机制**。当模型在第一次检测时没有发现人脸，系统会自动进行以下操作：

1. 对原图像添加padding，使人脸占比变小
2. 将padding后的图像resize回原始尺寸
3. 重新进行人脸检测
4. 如果检测成功，将坐标转换回原图像坐标系

## 新增功能

### 1. 核心函数

- `pad_with_ratio(image, padding_ratio=0.5)`: 对图像进行指定比例的padding
- `adjust_bbox_coordinates(objs, scale_factor, target_w, target_h)`: 调整检测框坐标
- `detect_single_attempt(model, image, threshold=0.4, nms_iou=0.5)`: 单次检测尝试
- `detect(model, image, threshold=0.4, nms_iou=0.5, use_padding_retry=True, padding_ratios=[0.3, 0.6, 1.0])`: 改进的主检测函数

### 2. 参数说明

#### `detect()` 函数新增参数：
- `use_padding_retry`: 是否启用padding重试机制（默认：True）
- `padding_ratios`: padding比例列表，按顺序尝试（默认：[0.3, 0.6, 1.0]）

#### `pad_with_ratio()` 函数参数：
- `padding_ratio`: padding比例，0.5表示在每个方向添加原尺寸50%的padding

### 3. 工作流程

```
原图像 → 第一次检测
    ↓
检测到人脸？ → 是 → 返回结果
    ↓ 否
应用padding(0.3倍) → resize回原尺寸 → 检测
    ↓
检测到人脸？ → 是 → 坐标转换 → 返回结果
    ↓ 否
应用padding(0.6倍) → resize回原尺寸 → 检测
    ↓
检测到人脸？ → 是 → 坐标转换 → 返回结果
    ↓ 否
应用padding(1.0倍) → resize回原尺寸 → 检测
    ↓
检测到人脸？ → 是 → 坐标转换 → 返回结果
    ↓ 否
返回空列表
```

## 使用方法

### 1. 基本使用

```python
# 启用padding重试（默认）
objs = detect(model, image)

# 禁用padding重试
objs = detect(model, image, use_padding_retry=False)

# 自定义padding比例
objs = detect(model, image, padding_ratios=[0.2, 0.5, 0.8])
```

### 2. 图像检测演示

```python
# 正常模式
detect_image(dbface, "image.jpg", use_padding_retry=False)

# padding重试模式
detect_image(dbface, "image.jpg", use_padding_retry=True)
```

### 3. 摄像头演示

```python
# 启动摄像头演示，运行时按't'键可切换padding重试模式
camera_demo()
```

### 4. 功能测试

```python
# 测试padding重试功能的效果
test_padding_feature()
```

## 演示程序

运行 `python main.py` 后，程序会提供以下选项：

1. **图像演示**: 对示例图像进行检测，比较正常模式和padding重试模式的效果
2. **摄像头演示**: 实时摄像头检测，可按't'键切换模式
3. **功能测试**: 专门测试padding重试功能的效果
4. **运行所有演示**: 依次运行上述所有演示

## 优势

1. **提高检测率**: 对于人脸占比较大的图像，通过padding可以显著提高检测成功率
2. **自动重试**: 无需手动调整，系统自动尝试多种padding比例
3. **坐标精确**: 自动进行坐标转换，确保结果准确
4. **可配置**: 可以自定义padding比例列表和是否启用该功能
5. **向后兼容**: 现有代码无需修改即可使用

## 适用场景

- 人脸占图像比例过大导致检测失败
- 图像边缘的人脸检测
- 需要提高检测召回率的应用
- 对检测精度要求较高的场景

## 注意事项

1. padding重试会增加计算时间，建议根据实际需求调整`padding_ratios`列表
2. 对于已经能正常检测到人脸的图像，该功能不会产生额外效果
3. 坐标转换可能会有轻微的精度损失，但在实际应用中通常可以忽略
4. 建议根据具体应用场景调整`threshold`和`padding_ratios`参数 