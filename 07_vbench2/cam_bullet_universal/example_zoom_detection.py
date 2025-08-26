#!/usr/bin/env python3
"""
运镜检测示例脚本：Zoom/Dolly + 子弹时间
基于前景人像分割、CoTracker特征点和人脸姿态的综合分析
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
        print(f"主要运镜: {result['primary_motion']} (置信度: {result['primary_confidence']:.3f})")
        print(f"子弹时间: {result['bullet_time_type']} (置信度: {result['bullet_time_confidence']:.3f})")
        print(f"Zoom/Dolly: {result['motion_type']} (置信度: {result['motion_confidence']:.3f})")
        
        # 人脸检测统计
        print(f"人脸检测: {result['face_pose_count']}/{result['total_frames']} 帧 ({result['face_pose_count']/result['total_frames']*100:.1f}%)")
        
        # 前景覆盖率统计
        mask_coverage = result['foreground_mask_coverage']
        print(f"前景覆盖率: 平均{np.mean(mask_coverage):.3f}, 范围[{np.min(mask_coverage):.3f}, {np.max(mask_coverage):.3f}]")
        
        # 子弹时间详细分析
        if result['bullet_time_type'] != 'no_bullet_time':
            bullet_details = result['bullet_time_details']
            print("\n=== 子弹时间分析 ===")
            print(f"Yaw角度变化: {bullet_details.get('yaw_range', 0):.1f}°")
            print(f"Pitch角度变化: {bullet_details.get('pitch_range', 0):.1f}°")
            print(f"Roll角度变化: {bullet_details.get('roll_range', 0):.1f}°")
            print(f"环形运动一致性: {bullet_details.get('circular_consistency', 0):.3f}")
            print(f"总角度变化: {bullet_details.get('total_angle_change', 0):.1f}°")
            print(f"人脸检测比例: {bullet_details.get('face_pose_ratio', 0):.3f}")
        
        # Zoom/Dolly详细分析
        if result['motion_type'] != 'no_dolly':
            motion_details = result['motion_details']
            print("\n=== Zoom/Dolly分析 ===")
            print(f"距离变化趋势: {motion_details.get('distance_trend', 0):.4f}")
            print(f"距离变化比例: {motion_details.get('distance_change_ratio', 0):.4f}")
            print(f"径向投影均值: {motion_details.get('mean_proj', 0):.4f}")
            print(f"投影一致性: {motion_details.get('proj_consistency', 0):.3f}")
            print(f"距离一致性: {motion_details.get('distance_consistency', 0):.3f}")
            print(f"综合一致性: {motion_details.get('overall_consistency', 0):.3f}")
            
            # 前景特征点数量统计
            fg_points = motion_details.get('num_foreground_points', [])
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
            'primary_motion': result['primary_motion'],
            'primary_confidence': result['primary_confidence'],
            'bullet_time_type': result['bullet_time_type'],
            'bullet_time_confidence': result['bullet_time_confidence'],
            'motion_type': result['motion_type'],
            'motion_confidence': result['motion_confidence'],
            'standard_motions': result['standard_motions'],
            'face_detection_ratio': result['face_pose_count'] / result['total_frames']
        })
        else:
            results.append({
                'video_path': video_path,
                'primary_motion': 'error',
                'primary_confidence': 0.0,
                'bullet_time_type': 'error',
                'bullet_time_confidence': 0.0,
                'motion_type': 'error',
                'motion_confidence': 0.0,
                'standard_motions': [],
                'face_detection_ratio': 0.0
            })
    
    # 汇总统计
    print(f"\n{'='*50}")
    print("批量分析汇总")
    print(f"{'='*50}")
    
    valid_results = [r for r in results if r['primary_motion'] != 'error']
    if valid_results:
        from collections import Counter
        
        # 主要运镜类型分布
        primary_types = [r['primary_motion'] for r in valid_results]
        primary_counts = Counter(primary_types)
        print("主要运镜类型分布:")
        for motion_type, count in primary_counts.most_common():
            print(f"  {motion_type}: {count}个视频")
        
        # 子弹时间检测统计
        bullet_types = [r['bullet_time_type'] for r in valid_results if r['bullet_time_type'] != 'no_bullet_time']
        if bullet_types:
            bullet_counts = Counter(bullet_types)
            print("\n子弹时间检测分布:")
            for bullet_type, count in bullet_counts.most_common():
                print(f"  {bullet_type}: {count}个视频")
        
        # Zoom/Dolly检测统计
        zoom_dolly_types = [r['motion_type'] for r in valid_results if r['motion_type'] != 'no_dolly']
        if zoom_dolly_types:
            zoom_dolly_counts = Counter(zoom_dolly_types)
            print("\nZoom/Dolly检测分布:")
            for motion_type, count in zoom_dolly_counts.most_common():
                print(f"  {motion_type}: {count}个视频")
        
        # 平均置信度和人脸检测率
        avg_primary_confidence = np.mean([r['primary_confidence'] for r in valid_results if r['primary_confidence'] > 0])
        avg_face_detection = np.mean([r['face_detection_ratio'] for r in valid_results])
        print(f"\n平均主要运镜置信度: {avg_primary_confidence:.3f}")
        print(f"平均人脸检测率: {avg_face_detection:.3f} ({avg_face_detection*100:.1f}%)")
        
        # 子弹时间检测成功率
        bullet_success_count = len([r for r in valid_results if r['bullet_time_type'] != 'no_bullet_time'])
        bullet_success_rate = bullet_success_count / len(valid_results)
        print(f"子弹时间检测成功率: {bullet_success_rate:.3f} ({bullet_success_rate*100:.1f}%)")
    
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

