"""Simple background-based anomaly detection for short videos (~3s @ 16 FPS).

Detects sudden, persistent foreground regions such as black blocks in the
background or newly appearing islands on the sea.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def read_video_frames(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    """Read frames from a video using OpenCV.

    Returns frames in BGR order and the source FPS.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 16.0
    frames: List[np.ndarray] = []
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target = total if max_frames is None else min(total, max_frames)

    while True:
        if max_frames is not None and len(frames) >= target:
            break
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()
    return frames, fps


def compute_background_gray(frames: List[np.ndarray], warmup_frames: int) -> np.ndarray:
    """Compute a robust grayscale background as temporal median of warmup frames."""
    warmup = max(1, min(warmup_frames, len(frames)))
    stack = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in range(warmup)], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    return median


def preprocess_gray(image_gray: np.ndarray) -> np.ndarray:
    """Apply gentle denoising to stabilize difference maps."""
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    return blurred


def diff_mask(frame_gray: np.ndarray, bg_gray: np.ndarray, diff_thresh: int) -> np.ndarray:
    """Compute binary foreground mask from absolute difference."""
    absd = cv2.absdiff(frame_gray, bg_gray)
    _, mask = cv2.threshold(absd, diff_thresh, 255, cv2.THRESH_BINARY)
    return mask


def clean_mask(mask: np.ndarray, erode_kernel: int, dilate_kernel: int) -> np.ndarray:
    """Clean binary mask using morphology to remove noise and fill holes."""
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, erode_kernel), max(1, erode_kernel)))
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, dilate_kernel), max(1, dilate_kernel)))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ek, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, dk, iterations=2)
    return cleaned


def find_bounding_boxes(mask: np.ndarray, min_area: float) -> List[Tuple[int, int, int, int]]:
    """Find bounding boxes from a binary mask, filtering by area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if float(w * h) >= min_area:
            boxes.append((x, y, w, h))
    return boxes


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes defined as (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0
    area_a = aw * ah
    area_b = bw * bh
    union = float(area_a + area_b - inter_area)
    return float(inter_area) / union if union > 0 else 0.0


def box_centroid(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Compute centroid of a box (x, y, w, h)."""
    x, y, w, h = box
    return (float(x + w / 2.0), float(y + h / 2.0))


class Track:
    """Minimal track to accumulate persistence and drift for a region."""

    def __init__(self, track_id: int, box: Tuple[int, int, int, int], frame_index: int) -> None:
        self.track_id = track_id
        self.start_frame = frame_index
        self.end_frame = frame_index
        self.box = box
        self.num_observed_frames = 1
        self.total_centroid_shift = 0.0
        self.prev_centroid = box_centroid(box)

    def update(self, box: Tuple[int, int, int, int], frame_index: int) -> None:
        self.end_frame = frame_index
        current_centroid = box_centroid(box)
        dx = current_centroid[0] - self.prev_centroid[0]
        dy = current_centroid[1] - self.prev_centroid[1]
        self.total_centroid_shift += math.hypot(dx, dy)
        self.prev_centroid = current_centroid
        self.box = box
        self.num_observed_frames += 1

    def mean_shift_per_frame(self) -> float:
        if self.num_observed_frames <= 1:
            return 0.0
        return self.total_centroid_shift / float(self.num_observed_frames - 1)

    def to_result(self, fps: float) -> Dict[str, float]:
        x, y, w, h = self.box
        return {
            "track_id": int(self.track_id),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "start_time": float(self.start_frame) / float(fps),
            "end_time": float(self.end_frame) / float(fps),
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
        }


def match_boxes_to_tracks(
    boxes: List[Tuple[int, int, int, int]],
    tracks: List[Track],
    iou_threshold: float,
    max_centroid_dist: float,
) -> Tuple[List[Tuple[int, Tuple[int, int, int, int]]], List[Tuple[int, int, int, int]]]:
    """Greedy matching of boxes to existing tracks by IoU then centroid distance.

    Returns list of (track_index, box) matches and list of unmatched boxes.
    """
    matches: List[Tuple[int, Tuple[int, int, int, int]]] = []
    unmatched_boxes = boxes[:]
    used_track_indices: set = set()

    # First pass: IoU-based
    for t_idx, track in enumerate(tracks):
        if t_idx in used_track_indices:
            continue
        best_iou = 0.0
        best_bi = -1
        for bi, box in enumerate(unmatched_boxes):
            score = iou(track.box, box)
            if score > best_iou:
                best_iou = score
                best_bi = bi
        if best_bi >= 0 and best_iou >= iou_threshold:
            matches.append((t_idx, unmatched_boxes[best_bi]))
            used_track_indices.add(t_idx)
            unmatched_boxes.pop(best_bi)

    # Second pass: centroid distance for remaining tracks
    for t_idx, track in enumerate(tracks):
        if t_idx in used_track_indices:
            continue
        tcx, tcy = box_centroid(track.box)
        best_dist = float("inf")
        best_bi = -1
        for bi, box in enumerate(unmatched_boxes):
            bcx, bcy = box_centroid(box)
            dist = math.hypot(bcx - tcx, bcy - tcy)
            if dist < best_dist:
                best_dist = dist
                best_bi = bi
        if best_bi >= 0 and best_dist <= max_centroid_dist:
            matches.append((t_idx, unmatched_boxes[best_bi]))
            used_track_indices.add(t_idx)
            unmatched_boxes.pop(best_bi)

    return matches, unmatched_boxes


def detect_anomalies(
    video_path: str,
    out_vis_path: Optional[str] = None,
    out_json_path: Optional[str] = None,
    warmup_frames: int = 16,
    diff_thresh: int = 25,
    min_area_ratio: float = 0.0025,
    persistence_frames: int = 6,
    iou_threshold: float = 0.3,
    max_centroid_shift_per_frame: float = 3.5,
    erode_kernel: int = 3,
    dilate_kernel: int = 7,
) -> Dict[str, object]:
    """Run anomaly detection on a short video and optionally save outputs.

    Returns a dict containing a high-level result and per-anomaly details.
    """
    frames, fps = read_video_frames(video_path)
    if len(frames) == 0:
        raise RuntimeError("No frames decoded from video.")

    h, w = frames[0].shape[:2]
    min_area = float(h * w) * float(min_area_ratio)

    bg_gray = compute_background_gray(frames, warmup_frames)
    bg_gray = preprocess_gray(bg_gray)

    writer = None
    if out_vis_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_vis_path, fourcc, fps, (w, h))

    tracks: List[Track] = []
    next_track_id = 1
    confirmed: List[Track] = []

    for fi, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_gray(gray)
        mask = diff_mask(gray, bg_gray, diff_thresh)
        mask = clean_mask(mask, erode_kernel, dilate_kernel)

        boxes = find_bounding_boxes(mask, min_area=min_area)
        matches, unmatched_boxes = match_boxes_to_tracks(
            boxes=boxes,
            tracks=tracks,
            iou_threshold=iou_threshold,
            max_centroid_dist=max_centroid_shift_per_frame * 2.5,
        )

        for t_idx, box in matches:
            tracks[t_idx].update(box, fi)

        for box in unmatched_boxes:
            tracks.append(Track(next_track_id, box, fi))
            next_track_id += 1

        persisted_tracks: List[Track] = []
        for track in tracks:
            if track.num_observed_frames >= persistence_frames and track.mean_shift_per_frame() <= max_centroid_shift_per_frame:
                persisted_tracks.append(track)

        for track in persisted_tracks:
            if track not in confirmed:
                confirmed.append(track)

        if writer is not None:
            vis = frame.copy()
            for x, y, w1, h1 in boxes:
                cv2.rectangle(vis, (x, y), (x + w1, y + h1), (0, 255, 255), 2)
            for tr in confirmed:
                x, y, w1, h1 = tr.box
                cv2.rectangle(vis, (x, y), (x + w1, y + h1), (0, 0, 255), 2)
                cv2.putText(
                    vis,
                    f"ID {tr.track_id}",
                    (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(vis)

    if writer is not None:
        writer.release()

    results = [tr.to_result(fps) for tr in confirmed if tr.start_frame >= min(warmup_frames - 1, len(frames) - 1)]

    output: Dict[str, object] = {
        "video": os.path.abspath(video_path),
        "fps": fps,
        "width": w,
        "height": h,
        "num_frames": len(frames),
        "num_anomalies": len(results),
        "anomalies": results,
    }

    if out_json_path:
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def build_argparser() -> argparse.ArgumentParser:
    """Create argument parser for CLI usage."""
    parser = argparse.ArgumentParser(description="Anomaly detection on short videos using background subtraction.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (3s @ 16fps recommended)")
    parser.add_argument("--out_vis", type=str, default=None, help="Optional output annotated video path (e.g., out.mp4)")
    parser.add_argument("--out_json", type=str, default=None, help="Optional output JSON path for detection results")
    parser.add_argument("--warmup_frames", type=int, default=16, help="Number of first frames to build background")
    parser.add_argument("--diff_thresh", type=int, default=25, help="Absolute difference threshold (0-255)")
    parser.add_argument("--min_area_ratio", type=float, default=0.0025, help="Min area ratio of frame to keep a region")
    parser.add_argument("--persistence_frames", type=int, default=6, help="Frames a region must persist to be an anomaly")
    parser.add_argument("--iou_threshold", type=float, default=0.3, help="IoU threshold for tracking association")
    parser.add_argument("--max_centroid_shift", type=float, default=3.5, help="Max mean centroid shift per frame (pixels)")
    parser.add_argument("--erode_kernel", type=int, default=3, help="Erode kernel size for noise removal")
    parser.add_argument("--dilate_kernel", type=int, default=7, help="Dilate/close kernel size for filling gaps")
    return parser


def main() -> None:
    """Entry point for CLI."""
    parser = build_argparser()
    args = parser.parse_args()

    result = detect_anomalies(
        video_path=args.video,
        out_vis_path=args.out_vis,
        out_json_path=args.out_json,
        warmup_frames=args.warmup_frames,
        diff_thresh=args.diff_thresh,
        min_area_ratio=args.min_area_ratio,
        persistence_frames=args.persistence_frames,
        iou_threshold=args.iou_threshold,
        max_centroid_shift_per_frame=args.max_centroid_shift,
        erode_kernel=args.erode_kernel,
        dilate_kernel=args.dilate_kernel,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()