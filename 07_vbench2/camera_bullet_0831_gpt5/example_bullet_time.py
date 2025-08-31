#!/usr/bin/env python3
"""
子弹时间检测使用示例
演示如何使用VGGT和CoTracker进行子弹时间运镜检测
"""

import os
import sys
import torch
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from vbench2.bullet_time import BulletTimeDetector, compute_bullet_time
from vbench2.camera_motion_ori import CameraPredict

def example_single_video_detection():
    """单个视频的子弹时间检测示例"""
    print("=== 单个视频子弹时间检测示例 ===")
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置参数
    submodules_dict = {
        # CoTracker配置
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8',
        
        # VGGT配置
        'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
        'vggt_weights': None  # 如果有预训练权重，请提供路径
    }
    
    # 初始化检测器
    try:
        detector = BulletTimeDetector(device, submodules_dict)
        print("✓ 子弹时间检测器初始化成功")
    except Exception as e:
        print(f"✗ 检测器初始化失败: {e}")
        return
    
    # 测试视频路径（请替换为实际视频路径）
    test_video = "path/to/your/bullet_time_video.mp4"
    
    if not os.path.exists(test_video):
        print(f"✗ 测试视频不存在: {test_video}")
        print("请将 test_video 变量设置为实际的视频文件路径")
        return
    
    print(f"正在检测视频: {test_video}")
    
    # 执行检测
    try:
        result = detector.detect_bullet_time(test_video)
        
        # 输出结果
        print("\n检测结果:")
        print(f"  子弹时间检测: {'✓ 是' if result['is_bullet_time'] else '✗ 否'}")
        print(f"  综合置信度: {result['confidence']:.3f}")
        
        # Yaw旋转分析
        yaw_info = result['yaw_rotation']
        print(f"\n头部Yaw角度分析:")
        print(f"  检测到Yaw旋转: {'✓ 是' if yaw_info['detected'] else '✗ 否'}")
        print(f"  Yaw旋转置信度: {yaw_info['confidence']:.3f}")
        
        if 'details' in yaw_info:
            details = yaw_info['details']
            print(f"  总旋转角度: {details.get('total_yaw_rotation', 0):.1f}°")
            print(f"  运动一致性: {details.get('direction_consistency', 0):.3f}")
            print(f"  有效帧数: {details.get('valid_frames', 0)}/{details.get('total_frames', 0)}")
        
        # 环绕运动分析
        orbit_info = result['orbit_motion']
        print(f"\n环绕运动分析:")
        print(f"  检测到环绕运动: {'✓ 是' if orbit_info['detected'] else '✗ 否'}")
        print(f"  环绕运动置信度: {orbit_info['confidence']:.3f}")
        
    except Exception as e:
        print(f"✗ 检测过程出错: {e}")

def example_camera_motion_integration():
    """集成到相机运动检测的示例"""
    print("\n=== 集成相机运动检测示例 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    submodules_dict = {
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8',
        'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
        'vggt_weights': None
    }
    
    # 初始化相机预测器
    try:
        camera = CameraPredict(device, submodules_dict)
        print("✓ 相机运动预测器初始化成功")
    except Exception as e:
        print(f"✗ 预测器初始化失败: {e}")
        return
    
    test_video = "path/to/your/video.mp4"
    
    if not os.path.exists(test_video):
        print(f"✗ 测试视频不存在: {test_video}")
        return
    
    print(f"正在分析视频: {test_video}")
    
    try:
        # 使用增强预测方法
        result = camera.predict_with_bullet_time(test_video, fps=30, end_frame=-1)
        
        print("\n相机运动分析结果:")
        print(f"  标准运镜类型: {result['standard_motions']}")
        print(f"  增强运镜类型: {result['enhanced_motions']}")
        print(f"  检测到环绕运动: {'✓ 是' if result['has_orbit_motion'] else '✗ 否'}")
        
        bullet_info = result['bullet_time']
        print(f"\n子弹时间分析:")
        print(f"  检测结果: {'✓ 是' if bullet_info['detected'] else '✗ 否'}")
        print(f"  置信度: {bullet_info['confidence']:.3f}")
        
        if 'details' in bullet_info:
            details = bullet_info['details']
            print(f"  总Yaw旋转: {details.get('total_yaw_rotation', 0):.1f}°")
            print(f"  运动一致性: {details.get('direction_consistency', 0):.3f}")
        
    except Exception as e:
        print(f"✗ 分析过程出错: {e}")

def example_batch_evaluation():
    """批量评估示例"""
    print("\n=== 批量评估示例 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    submodules_dict = {
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8',
        'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
        'vggt_weights': None
    }
    
    # VBench2评估配置文件路径
    json_dir = "path/to/VBench2_full_info.json"
    
    if not os.path.exists(json_dir):
        print(f"✗ 评估配置文件不存在: {json_dir}")
        print("请提供正确的VBench2配置文件路径")
        return
    
    print(f"使用配置文件: {json_dir}")
    
    try:
        # 运行批量评估
        avg_score, video_results = compute_bullet_time(
            json_dir=json_dir,
            device=device,
            submodules_dict=submodules_dict
        )
        
        print(f"\n批量评估结果:")
        print(f"  平均得分: {avg_score:.3f}")
        print(f"  处理视频数量: {len(video_results)}")
        
        # 统计结果
        successful_detections = sum(1 for r in video_results if r['video_results'] > 0)
        print(f"  检测到子弹时间的视频: {successful_detections}/{len(video_results)}")
        
        # 显示前几个结果
        print(f"\n前5个视频结果:")
        for i, result in enumerate(video_results[:5]):
            video_path = os.path.basename(result['video_path'])
            score = result['video_results']
            status = "✓ 检测到" if score > 0 else "✗ 未检测到"
            print(f"  {i+1}. {video_path}: {status} (得分: {score})")
        
    except Exception as e:
        print(f"✗ 批量评估出错: {e}")

def print_configuration_guide():
    """打印配置指南"""
    print("\n=== 配置指南 ===")
    print("1. 确保已安装必要依赖:")
    print("   pip install torch torchvision opencv-python decord numpy tqdm dlib")
    print()
    print("2. 准备VGGT模型:")
    print("   - 将VGGT代码放置在: VBench-2.0/vbench2/third_party/vggt-main/vggt-main/")
    print("   - 下载预训练权重文件（可选）")
    print()
    print("3. 测试视频要求:")
    print("   - 包含人脸的视频")
    print("   - 视频长度建议 > 2秒")
    print("   - 分辨率建议 > 224x224")
    print()
    print("4. 子弹时间特征:")
    print("   - 相机围绕主体环绕拍摄")
    print("   - 主体头部发生明显yaw角度旋转 (> 75°)")
    print("   - 运动具有连续性和一致性")

if __name__ == "__main__":
    print("子弹时间检测 - 使用示例")
    print("=" * 50)
    
    # 打印配置指南
    print_configuration_guide()
    
    # 运行示例
    try:
        # 示例1: 单个视频检测
        example_single_video_detection()
        
        # 示例2: 集成相机运动检测
        example_camera_motion_integration()
        
        # 示例3: 批量评估
        example_batch_evaluation()
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行出错: {e}")
    
    print("\n示例执行完成！")
