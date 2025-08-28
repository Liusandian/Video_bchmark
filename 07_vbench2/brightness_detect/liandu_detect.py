"""
亮度突变与清晰度检测模块

功能：
1) 逐帧统计亮度，检测突变（骤亮/骤暗）
2) 模糊检测（方差拉普拉斯）与噪声辅助（可选）
3) 提供命令行工具，支持单视频或目录批量筛选

依赖：opencv-python, numpy
用法：
  python liandu_detect.py --video path/to/video.mp4
  python liandu_detect.py --dir path/to/videos --ext mp4 --save_json report.json
"""

import os
import cv2
import json
import glob
import argparse
import numpy as np
from typing import Dict, List, Tuple


def compute_frame_brightness(frame: np.ndarray, method: str = "y_mean") -> float:
    """计算单帧亮度。
    method:
      - y_mean: 转YCbCr后取Y通道均值（更贴近感知亮度）
      - gray_mean: 灰度均值
    """
    if method == "y_mean":
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        return float(np.mean(yuv[..., 0]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def detect_brightness_jumps(
    video_path: str,
    brightness_delta_abs: float = 25.0,
    brightness_delta_rel: float = 0.25,
    min_consecutive: int = 1,
    sample_stride: int = 1,
) -> Dict:
    """检测视频亮度突变。
    - brightness_delta_abs: 相邻帧亮度绝对变化阈值（0-255）
    - brightness_delta_rel: 相邻帧亮度相对变化阈值（相对上一帧亮度的比例）
    - min_consecutive: 连续满足阈值的最小帧数（>1可抑制偶发抖动）
    - sample_stride: 取帧步距，>1可加速但精度略降
    返回：
      {
        'video_path', 'fps', 'frame_count',
        'brightness_series': [...],
        'jump_indices': [idx, ...],
        'has_jump': bool
      }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    brightness_series: List[float] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx % sample_stride) == 0:
            brightness_series.append(compute_frame_brightness(frame, method="y_mean"))
        idx += 1
    cap.release()

    jumps: List[int] = []
    consec = 0
    for i in range(1, len(brightness_series)):
        prev, cur = brightness_series[i - 1], brightness_series[i]
        delta_abs = abs(cur - prev)
        delta_rel = delta_abs / max(1e-6, prev)
        if (delta_abs >= brightness_delta_abs) or (delta_rel >= brightness_delta_rel):
            consec += 1
            if consec >= min_consecutive:
                jumps.append(i)
        else:
            consec = 0

    return {
        'video_path': video_path,
        'fps': fps,
        'frame_count': frame_count,
        'brightness_series': brightness_series,
        'jump_indices': jumps,
        'has_jump': len(jumps) > 0,
    }


def variance_of_laplacian(frame: np.ndarray) -> float:
    """拉普拉斯方差，衡量清晰度（数值越低越模糊）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_blur(
    video_path: str,
    blur_thresh: float = 100.0,
    sample_stride: int = 1,
    min_blur_ratio: float = 0.5,
) -> Dict:
    """视频模糊检测。
    - blur_thresh: 拉普拉斯方差阈值（经验：<100较模糊，场景依赖可调）
    - sample_stride: 取帧步距
    - min_blur_ratio: 低于阈值的帧比例超过该值则判定视频模糊
    返回：
      {
        'video_path', 'fps', 'frame_count',
        'laplacian_series': [...],
        'blur_ratio': 0~1,
        'is_blurry': bool
      }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    vals: List[float] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx % sample_stride) == 0:
            vals.append(variance_of_laplacian(frame))
        idx += 1
    cap.release()

    if len(vals) == 0:
        return {
            'video_path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'laplacian_series': [],
            'blur_ratio': 1.0,
            'is_blurry': True,
        }

    blur_ratio = float(np.mean(np.array(vals) < blur_thresh))
    return {
        'video_path': video_path,
        'fps': fps,
        'frame_count': frame_count,
        'laplacian_series': vals,
        'blur_ratio': blur_ratio,
        'is_blurry': blur_ratio >= min_blur_ratio,
    }


def analyze_video(
    video_path: str,
    brightness_delta_abs: float = 25.0,
    brightness_delta_rel: float = 0.25,
    blur_thresh: float = 100.0,
    stride: int = 1,
) -> Dict:
    """综合分析：亮度突变 + 模糊检测。"""
    jump = detect_brightness_jumps(
        video_path,
        brightness_delta_abs=brightness_delta_abs,
        brightness_delta_rel=brightness_delta_rel,
        min_consecutive=1,
        sample_stride=stride,
    )
    blur = detect_blur(
        video_path,
        blur_thresh=blur_thresh,
        sample_stride=stride,
        min_blur_ratio=0.5,
    )
    return {
        'video_path': video_path,
        'has_brightness_jump': jump['has_jump'],
        'jump_indices': jump['jump_indices'],
        'brightness_series': jump['brightness_series'],
        'is_blurry': blur['is_blurry'],
        'blur_ratio': blur['blur_ratio'],
        'laplacian_series': blur['laplacian_series'],
        'fps': jump['fps'],
        'frame_count': jump['frame_count'],
    }


def analyze_dir(
    directory: str,
    ext: str = "mp4",
    brightness_delta_abs: float = 25.0,
    brightness_delta_rel: float = 0.25,
    blur_thresh: float = 100.0,
    stride: int = 1,
) -> List[Dict]:
    """批量分析目录内视频。"""
    paths = sorted(glob.glob(os.path.join(directory, f"**/*.{ext}"), recursive=True))
    results = []
    for p in paths:
        try:
            results.append(analyze_video(
                p,
                brightness_delta_abs=brightness_delta_abs,
                brightness_delta_rel=brightness_delta_rel,
                blur_thresh=blur_thresh,
                stride=stride,
            ))
        except Exception as e:
            results.append({'video_path': p, 'error': str(e)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Brightness jump and blur detection")
    parser.add_argument('--video', type=str, default="", help='Path to video file')
    parser.add_argument('--dir', type=str, default="", help='Directory of videos')
    parser.add_argument('--ext', type=str, default="mp4", help='Video extension for directory mode')
    parser.add_argument('--stride', type=int, default=1, help='Frame sampling stride')
    parser.add_argument('--brightness_abs', type=float, default=25.0, help='Absolute brightness delta threshold (0-255)')
    parser.add_argument('--brightness_rel', type=float, default=0.25, help='Relative brightness delta threshold (0-1)')
    parser.add_argument('--blur_thresh', type=float, default=100.0, help='Variance of Laplacian threshold')
    parser.add_argument('--save_json', type=str, default="", help='Save results to json path')

    args = parser.parse_args()

    if args.video:
        result = analyze_video(
            args.video,
            brightness_delta_abs=args.brightness_abs,
            brightness_delta_rel=args.brightness_rel,
            blur_thresh=args.blur_thresh,
            stride=args.stride,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return

    if args.dir:
        results = analyze_dir(
            args.dir,
            ext=args.ext,
            brightness_delta_abs=args.brightness_abs,
            brightness_delta_rel=args.brightness_rel,
            blur_thresh=args.blur_thresh,
            stride=args.stride,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        return

    parser.print_help()


if __name__ == "__main__":
    main()