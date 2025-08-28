## 亮度突变与清晰度检测指南

### 功能概述
- 逐帧统计亮度，检测亮度突变（骤亮/骤暗）
- 基于方差拉普拉斯的模糊检测（阈值可配置）
- 支持单视频与目录批量分析，支持JSON导出

### 代码位置
- `vbench2/liandu_detect.py`

### 使用方式
- 单视频分析：
```bash
python vbench2/liandu_detect.py --video path/to/video.mp4 \
  --brightness_abs 25 --brightness_rel 0.25 --blur_thresh 100 --stride 1 \
  --save_json report.json
```
- 目录批量分析：
```bash
python vbench2/liandu_detect.py --dir path/to/videos --ext mp4 \
  --brightness_abs 25 --brightness_rel 0.25 --blur_thresh 100 --stride 1 \
  --save_json report.json
```

### 检测原理
- 亮度统计：
  - 将帧转换到YCbCr空间，取Y通道均值作为感知亮度
  - 相邻帧亮度变化：`|Y_t - Y_{t-1}| >= brightness_abs` 或 `|ΔY|/Y_{t-1} >= brightness_rel`
  - 支持连续帧计数（`min_consecutive`）抑制偶发抖动
- 模糊检测：
  - 方差拉普拉斯（Variance of Laplacian），数值越低越模糊
  - `blur_ratio = (#(V<blur_thresh))/N`，若 `blur_ratio >= min_blur_ratio` 判为模糊

### 关键参数建议
- `--brightness_abs`：25（8bit强度，场景偏暗时可降到15-20）
- `--brightness_rel`：0.25（相对比例，强曝光变化可调到0.3）
- `--blur_thresh`：100（运动多或低光场景可调到70-120）
- `--stride`：1（提速可设为2/3，但会降低精度）

### 输出字段说明（单视频）
```json
{
  "video_path": "...",
  "has_brightness_jump": true,
  "jump_indices": [15, 16],
  "brightness_series": [ ... ],
  "is_blurry": false,
  "blur_ratio": 0.22,
  "laplacian_series": [ ... ],
  "fps": 24.0,
  "frame_count": 300
}
```

### 实践建议
- 对生成视频先用 `--stride 2` 快速筛查，再对可疑样本用 `--stride 1` 复检
- 结合 `jump_indices` 回放定位突变帧，便于进一步诊断
- 不同数据域（暗光、运动、压缩强）需做一次小样本标定后确定阈值
