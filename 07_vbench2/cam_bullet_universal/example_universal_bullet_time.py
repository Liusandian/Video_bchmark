#!/usr/bin/env python3
"""
通用子弹时间运镜检测示例脚本
适用于人像、建筑、风景等各种场景的子弹时间效果检测
主要基于CoTracker特征点轨迹分析，不依赖人脸检测
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

def analyze_universal_bullet_time(video_path, device="cuda"):
    """分析视频的通用子弹时间运镜效果"""
    
    # 初始化检测器
    submodules = {
        "repo": "facebookresearch/co-tracker", 
        "model": "cotracker_w8"
    }
    
    try:
        camera = CameraPredict(device=device, submodules_list=submodules)
        print("✓ 通用运镜检测器初始化成功")
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
        # 使用通用子弹时间检测
        result = camera.predict_universal_bullet_time(video, fps=fps, end_frame=-1)
        
        # 输出结果
        print("\n=== 通用子弹时间分析结果 ===")
        print(f"标准运动类型: {result['standard_motions']}")
        print(f"主要运镜: {result['primary_motion']} (置信度: {result['primary_confidence']:.3f})")
        print(f"子弹时间类型: {result['bullet_time_type']}")
        print(f"子弹时间置信度: {result['bullet_time_confidence']:.3f}")
        print(f"总帧数: {result['total_frames']}")
        
        # 详细分析数据
        details = result['bullet_time_details']
        if details and result['bullet_time_type'] != 'no_bullet_time':
            print("\n=== 子弹时间详细分析 ===")
            print(f"总角度变化: {details.get('total_rotation_deg', 0):.1f}°")
            print(f"环形运动方向一致性: {details.get('circular_direction_consistency', 0):.3f}")
            print(f"径向运动稳定性: {details.get('radial_stability', 0):.3f}")
            print(f"角速度一致性: {details.get('angular_consistency', 0):.3f}")
            print(f"区域运动一致性: {details.get('region_consistency', 0):.3f}")
            print(f"运动周期性: {details.get('motion_periodicity', 0):.3f}")
            print(f"有效运动帧数: {details.get('num_motion_frames', 0)}")
            print(f"运动帧比例: {details.get('motion_frame_ratio', 0):.3f}")
            
            # 人脸辅助信息（如果有）
            if details.get('face_detection_ratio', 0) > 0:
                print(f"\n=== 人脸辅助信息 ===")
                print(f"人脸角度变化: {details.get('face_angle_range', 0):.1f}°")
                print(f"人脸检测率: {details.get('face_detection_ratio', 0):.3f}")
                print(f"人脸角度支持度: {details.get('face_angle_support', 0):.3f}")
        
        return result
        
    except Exception as e:
        print(f"✗ 子弹时间分析失败: {e}")
        return None

def batch_analyze_videos(video_list, device="cuda"):
    """批量分析多个视频的子弹时间效果"""
    results = []
    
    for i, video_path in enumerate(video_list):
        print(f"\n{'='*60}")
        print(f"分析视频 {i+1}/{len(video_list)}: {video_path}")
        print(f"{'='*60}")
        
        result = analyze_universal_bullet_time(video_path, device)
        if result:
            results.append({
                'video_path': video_path,
                'bullet_time_type': result['bullet_time_type'],
                'bullet_time_confidence': result['bullet_time_confidence'],
                'primary_motion': result['primary_motion'],
                'primary_confidence': result['primary_confidence'],
                'total_rotation': result['bullet_time_details'].get('total_rotation_deg', 0) if result['bullet_time_details'] else 0,
                'motion_consistency': result['bullet_time_details'].get('circular_direction_consistency', 0) if result['bullet_time_details'] else 0
            })
        else:
            results.append({
                'video_path': video_path,
                'bullet_time_type': 'error',
                'bullet_time_confidence': 0.0,
                'primary_motion': 'error',
                'primary_confidence': 0.0,
                'total_rotation': 0.0,
                'motion_consistency': 0.0
            })
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("批量分析汇总")
    print(f"{'='*60}")
    
    valid_results = [r for r in results if r['bullet_time_type'] != 'error']
    if valid_results:
        from collections import Counter
        
        # 子弹时间类型分布
        bullet_types = [r['bullet_time_type'] for r in valid_results]
        bullet_counts = Counter(bullet_types)
        print("子弹时间类型分布:")
        for bullet_type, count in bullet_counts.most_common():
            print(f"  {bullet_type}: {count}个视频")
        
        # 成功检测统计
        bullet_detected = len([r for r in valid_results if r['bullet_time_type'] != 'no_bullet_time'])
        detection_rate = bullet_detected / len(valid_results)
        print(f"\n子弹时间检测成功率: {detection_rate:.3f} ({detection_rate*100:.1f}%)")
        
        # 平均指标
        if bullet_detected > 0:
            detected_results = [r for r in valid_results if r['bullet_time_type'] != 'no_bullet_time']
            avg_confidence = np.mean([r['bullet_time_confidence'] for r in detected_results])
            avg_rotation = np.mean([r['total_rotation'] for r in detected_results])
            avg_consistency = np.mean([r['motion_consistency'] for r in detected_results])
            
            print(f"平均置信度: {avg_confidence:.3f}")
            print(f"平均角度变化: {avg_rotation:.1f}°")
            print(f"平均运动一致性: {avg_consistency:.3f}")
        
        # 按角度范围分类
        print(f"\n按角度范围分类:")
        angle_ranges = {
            '90°-180°': len([r for r in detected_results if 75 <= r['total_rotation'] < 150]),
            '180°-270°': len([r for r in detected_results if 150 <= r['total_rotation'] < 240]),
            '270°-360°': len([r for r in detected_results if 240 <= r['total_rotation'] < 320]),
            '360°+': len([r for r in detected_results if r['total_rotation'] >= 320])
        }
        
        for range_name, count in angle_ranges.items():
            if count > 0:
                print(f"  {range_name}: {count}个视频")
    
    return results

def compare_with_traditional_detection(video_path, device="cuda"):
    """对比通用检测和传统检测的效果"""
    submodules = {
        "repo": "facebookresearch/co-tracker", 
        "model": "cotracker_w8"
    }
    
    camera = CameraPredict(device=device, submodules_list=submodules)
    video = load_video_tensor(video_path)
    
    if video is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"\n{'='*60}")
    print(f"对比分析: {video_path}")
    print(f"{'='*60}")
    
    # 通用检测
    print("\n--- 通用子弹时间检测 ---")
    universal_result = camera.predict_universal_bullet_time(video, fps=fps)
    print(f"检测结果: {universal_result['bullet_time_type']}")
    print(f"置信度: {universal_result['bullet_time_confidence']:.3f}")
    
    # 传统检测（带人脸）
    print("\n--- 传统检测（含人脸分析） ---")
    traditional_result = camera.predict_with_segmentation(video, fps=fps)
    print(f"检测结果: {traditional_result['bullet_time_type']}")
    print(f"置信度: {traditional_result['bullet_time_confidence']:.3f}")
    print(f"人脸检测: {traditional_result['face_pose_count']}/{traditional_result['total_frames']} 帧")
    
    return {
        'universal': universal_result,
        'traditional': traditional_result
    }

def main():
    """主函数 - 使用示例"""
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <video_path>                    # 单个视频分析")
        print(f"  python {sys.argv[0]} <video1> <video2> ...          # 批量分析")
        print(f"  python {sys.argv[0]} --compare <video_path>         # 对比分析")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    if sys.argv[1] == "--compare" and len(sys.argv) == 3:
        # 对比分析
        compare_with_traditional_detection(sys.argv[2], device)
    elif len(sys.argv) == 2:
        # 单个视频分析
        analyze_universal_bullet_time(sys.argv[1], device)
    else:
        # 批量分析
        video_paths = sys.argv[1:]
        batch_analyze_videos(video_paths, device)

if __name__ == "__main__":
    main()
