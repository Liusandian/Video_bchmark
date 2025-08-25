#!/usr/bin/env python3
"""
Zoom/Dolly运镜检测示例脚本
基于前景人像分割和CoTracker特征点的综合分析
"""

import cv2
import numpy as np
import torch
import decord
from camera_motion import CameraPredict

def load_video_tensor(video_path):
    """加载视频为pytorch张量格式"""
    try:
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader)))
        # 转换为 B T C H W 格式
        video = video.permute(0, 3, 1, 2)[None].float()
        if torch.cuda.is_available():
            video = video.cuda()
        return video
    except Exception as e:
        print(f"视频加载失败: {e}")
        return None

def analyze_video_motion(video_path, device="cuda"):
    """分析视频的运镜类型"""
    
    # 初始化检测器
    submodules = {
        "repo": "facebookresearch/co-tracker", 
        "model": "cotracker_w8"
    }
    
    try:
        camera = CameraPredict(device=device, submodules_list=submodules)
        print("✓ 相机运动检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        return None
    
    # 加载视频
    video = load_video_tensor(video_path)
    if video is None:
        return None
    
    print(f"✓ 视频加载成功: {video.shape}")
    
    # 获取视频FPS
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    try:
        # 使用增强版检测
        result = camera.predict_with_segmentation(video, fps=fps, end_frame=-1)
        
        # 输出结果
        print("\n=== 运镜分析结果 ===")
        print(f"标准运动类型: {result['standard_motions']}")
        print(f"检测运镜类型: {result['motion_type']}")
        print(f"运镜置信度: {result['motion_confidence']:.3f}")
        
        # 前景覆盖率统计
        mask_coverage = result['foreground_mask_coverage']
        print(f"前景覆盖率: 平均{np.mean(mask_coverage):.3f}, 范围[{np.min(mask_coverage):.3f}, {np.max(mask_coverage):.3f}]")
        
        # 详细分析数据
        details = result['motion_details']
        if details:
            print("\n=== 详细分析数据 ===")
            print(f"距离变化趋势: {details.get('distance_trend', 0):.4f}")
            print(f"距离变化比例: {details.get('distance_change_ratio', 0):.4f}")
            print(f"径向投影均值: {details.get('mean_proj', 0):.4f}")
            print(f"投影一致性: {details.get('proj_consistency', 0):.3f}")
            print(f"距离一致性: {details.get('distance_consistency', 0):.3f}")
            print(f"综合一致性: {details.get('overall_consistency', 0):.3f}")
            
            # 前景特征点数量统计
            fg_points = details.get('num_foreground_points', [])
            if fg_points:
                print(f"前景特征点数: 平均{np.mean(fg_points):.1f}, 范围[{np.min(fg_points)}, {np.max(fg_points)}]")
        
        return result
        
    except Exception as e:
        print(f"✗ 运镜分析失败: {e}")
        return None

def batch_analyze_videos(video_list, device="cuda"):
    """批量分析多个视频"""
    results = []
    
    for i, video_path in enumerate(video_list):
        print(f"\n{'='*50}")
        print(f"分析视频 {i+1}/{len(video_list)}: {video_path}")
        print(f"{'='*50}")
        
        result = analyze_video_motion(video_path, device)
        if result:
            results.append({
                'video_path': video_path,
                'motion_type': result['motion_type'],
                'confidence': result['motion_confidence'],
                'standard_motions': result['standard_motions']
            })
        else:
            results.append({
                'video_path': video_path,
                'motion_type': 'error',
                'confidence': 0.0,
                'standard_motions': []
            })
    
    # 汇总统计
    print(f"\n{'='*50}")
    print("批量分析汇总")
    print(f"{'='*50}")
    
    motion_types = [r['motion_type'] for r in results if r['motion_type'] != 'error']
    if motion_types:
        from collections import Counter
        type_counts = Counter(motion_types)
        print("运镜类型分布:")
        for motion_type, count in type_counts.most_common():
            print(f"  {motion_type}: {count}个视频")
        
        # 平均置信度
        avg_confidence = np.mean([r['confidence'] for r in results if r['confidence'] > 0])
        print(f"平均置信度: {avg_confidence:.3f}")
    
    return results

def main():
    """主函数 - 使用示例"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <video_path>        # 分析单个视频")
        print(f"  python {sys.argv[0]} <video1> <video2> ... # 批量分析")
        return
    
    video_paths = sys.argv[1:]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if len(video_paths) == 1:
        # 单个视频分析
        analyze_video_motion(video_paths[0], device)
    else:
        # 批量分析
        batch_analyze_videos(video_paths, device)

if __name__ == "__main__":
    main()
