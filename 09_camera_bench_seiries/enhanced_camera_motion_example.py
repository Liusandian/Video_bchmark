"""
示例：如何在现有的VBench camera motion评测中添加召回率统计

这个文件展示了如何最小化修改现有代码来添加召回率分析功能
"""

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import cv2
import decord
import os
import json


def enhanced_camera_motion_with_recall(prompt_dict_ls, camera, save_visualizations=False, vis_output_dir="./camera_motion_visualizations"):
    """
    增强版的camera_motion函数，在原有功能基础上添加召回率统计
    
    这个函数保持与原camera_motion函数相同的接口，但额外返回召回率信息
    
    Returns:
        avg_score: 平均得分（与原函数相同）
        video_results: 视频结果列表（扩展了信息）
        recall_metrics: 召回率统计信息（新增）
    """
    from vbench2.utils import split_video_into_scenes
    
    # 用于统计召回率的数据结构
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'predictions': []})
    sim = []
    video_results = []

    if save_visualizations:
        os.makedirs(vis_output_dir, exist_ok=True)

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']  # ground truth运镜类型
        video_paths = prompt_dict['video_list']
        
        for idx, video_path in enumerate(video_paths):
            try:
                # === 原有的视频处理逻辑 ===
                end_frame = -1
                scene_list = split_video_into_scenes(video_path, 5.0)
                if len(scene_list) != 0:
                    end_frame = int(scene_list[0][1].get_frames())
                    
                video_reader = decord.VideoReader(video_path)
                video = video_reader.get_batch(range(len(video_reader))) 
                frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
                video = video.permute(0, 3, 1, 2)[None].float().cuda()
                
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                # 可视化保存（可选）
                if save_visualizations:
                    vis_save_dir = os.path.join(vis_output_dir, f"video_{idx}_{label}")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    camera.infer(video, fps, end_frame, save_video=True, save_dir=vis_save_dir, visualization_type="grid")
                
                # 预测运镜类型
                predict_results = camera.predict(video, fps, end_frame)
                video_score = 1.0 if label in predict_results else 0.0
                
                # === 原有的结果记录 ===
                sim.append(video_score)
                
                # === 扩展的结果记录（添加更多信息用于召回率分析）===
                enhanced_result = {
                    'video_path': video_path,
                    'video_results': video_score,
                    'ground_truth': label,           # 新增：ground truth标签
                    'predictions': predict_results,  # 新增：完整的预测结果
                    'video_info': {                  # 新增：视频元信息
                        'frame_count': frame_count,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'end_frame': end_frame
                    }
                }
                video_results.append(enhanced_result)
                
                # === 召回率统计 ===
                type_stats[label]['total'] += 1
                type_stats[label]['predictions'].append({
                    'video_path': video_path,
                    'predictions': predict_results,
                    'is_correct': video_score > 0.5
                })
                if video_score > 0.5:
                    type_stats[label]['correct'] += 1
                    
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue
    
    # === 计算原有的平均得分 ===
    avg_score = np.mean(sim)
    
    # === 计算召回率指标 ===
    recall_metrics = calculate_recall_from_stats(type_stats)
    
    # === 打印召回率报告 ===
    print_simple_recall_report(recall_metrics)
    
    return avg_score, video_results, recall_metrics


def calculate_recall_from_stats(type_stats):
    """
    从统计数据计算召回率指标
    """
    recall_by_type = {}
    total_correct = 0
    total_samples = 0
    
    for motion_type, stats in type_stats.items():
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        # 收集错误预测的例子
        wrong_predictions = [p for p in stats['predictions'] if not p['is_correct']]
        
        recall_by_type[motion_type] = {
            'recall': recall,
            'correct': stats['correct'],
            'total': stats['total'],
            'wrong_count': stats['total'] - stats['correct'],
            'sample_predictions': stats['predictions'][:3],  # 保留几个预测例子
            'error_examples': wrong_predictions[:2]  # 保留几个错误例子
        }
        
        total_correct += stats['correct']
        total_samples += stats['total']
    
    overall_recall = total_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'overall_recall': overall_recall,
        'total_samples': total_samples,
        'total_correct': total_correct,
        'recall_by_type': recall_by_type,
        'type_count': len(recall_by_type)
    }


def print_simple_recall_report(recall_metrics):
    """
    打印简洁的召回率报告
    """
    print(f"\n{'='*60}")
    print("Camera Motion 召回率统计")
    print(f"{'='*60}")
    
    print(f"总体召回率: {recall_metrics['overall_recall']:.3f}")
    print(f"运镜类型总数: {recall_metrics['type_count']}")
    print(f"总样本数: {recall_metrics['total_samples']}")
    print(f"正确预测数: {recall_metrics['total_correct']}")
    
    print(f"\n按运镜类型的召回率:")
    print(f"{'类型':<20} {'召回率':<10} {'正确/总数':<10}")
    print(f"{'-'*45}")
    
    # 按召回率排序
    sorted_types = sorted(recall_metrics['recall_by_type'].items(), 
                         key=lambda x: x[1]['recall'], reverse=True)
    
    for motion_type, stats in sorted_types:
        ratio_str = f"{stats['correct']}/{stats['total']}"
        print(f"{motion_type:<20} {stats['recall']:<10.3f} {ratio_str:<10}")
    
    # 统计性能分级
    high_count = sum(1 for _, stats in sorted_types if stats['recall'] >= 0.8)
    mid_count = sum(1 for _, stats in sorted_types if 0.5 <= stats['recall'] < 0.8)
    low_count = sum(1 for _, stats in sorted_types if stats['recall'] < 0.5)
    
    print(f"\n性能分级: 高性能({high_count}种) | 中等({mid_count}种) | 低性能({low_count}种)")
    
    # 显示需要改进的运镜类型
    need_improvement = [item for item in sorted_types if item[1]['recall'] < 0.6]
    if need_improvement:
        print(f"\n需要改进的运镜类型:")
        for motion_type, stats in need_improvement[:3]:
            print(f"  {motion_type}: {stats['recall']:.3f}")


def compute_camera_motion_with_recall(json_dir, device, submodules_dict, save_visualizations=False, **kwargs):
    """
    修改后的compute_camera_motion函数，包含召回率分析
    
    这个函数可以直接替换原有的compute_camera_motion函数
    
    Returns:
        all_results: 总体结果得分
        detailed_results: 包含video_results和recall_metrics的详细结果
    """
    from vbench2.camera_motion import CameraPredict
    from vbench2.utils import load_dimension_info
    
    # 初始化
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    
    # 执行增强版评测
    avg_score, video_results, recall_metrics = enhanced_camera_motion_with_recall(
        prompt_dict_ls, camera, save_visualizations=save_visualizations, **kwargs
    )
    
    # 保存召回率分析结果
    save_recall_results(recall_metrics, "camera_motion_recall_results.json")
    
    # 返回结果（保持与原函数接口兼容）
    detailed_results = {
        'video_results': video_results,
        'recall_metrics': recall_metrics
    }
    
    return avg_score, detailed_results


def save_recall_results(recall_metrics, filename):
    """
    保存召回率结果到JSON文件
    """
    # 准备可序列化的数据
    serializable_data = {
        'summary': {
            'overall_recall': recall_metrics['overall_recall'],
            'total_samples': recall_metrics['total_samples'],
            'total_correct': recall_metrics['total_correct'],
            'type_count': recall_metrics['type_count']
        },
        'by_type': {}
    }
    
    for motion_type, stats in recall_metrics['recall_by_type'].items():
        serializable_data['by_type'][motion_type] = {
            'recall': stats['recall'],
            'correct': stats['correct'],
            'total': stats['total'],
            'wrong_count': stats['wrong_count']
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"召回率结果已保存到: {filename}")


# 使用示例
if __name__ == "__main__":
    """
    使用示例：如何在现有项目中使用这个增强版函数
    """
    
    # 方法1：直接替换原函数
    print("=== 方法1：直接替换原compute_camera_motion函数 ===")
    
    # 配置参数（根据实际情况调整）
    json_dir = "VBench-2.0/prompts"
    device = "cuda"
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 调用增强版函数
    # avg_score, detailed_results = compute_camera_motion_with_recall(
    #     json_dir, device, submodules_dict, save_visualizations=False
    # )
    
    # print(f"平均得分: {avg_score:.3f}")
    # print(f"总体召回率: {detailed_results['recall_metrics']['overall_recall']:.3f}")
    
    print("示例代码已注释，请根据实际环境取消注释运行")
    
    print("\n=== 方法2：在现有代码中添加召回率分析 ===")
    print("如果已经有video_results，可以使用camera_motion_recall_utils.py中的calculate_recall_metrics函数")
    
    # 示例：假设已经有了video_results
    # from camera_motion_recall_utils import calculate_recall_metrics, print_recall_report
    # recall_metrics = calculate_recall_metrics(video_results)
    # print_recall_report(recall_metrics) 