import numpy as np
from collections import defaultdict, Counter
import json


def calculate_recall_metrics(video_results):
    """
    基于video_results计算每种运镜类型的召回率
    
    Args:
        video_results: 来自camera_motion函数的video_results列表
                      每个元素包含: {'video_path': str, 'video_results': float, 'ground_truth': str, 'predictions': list}
    
    Returns:
        recall_metrics: 包含各种召回率统计的字典
    """
    
    # 统计每种运镜类型的结果
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong_predictions': []})
    
    for result in video_results:
        if 'ground_truth' not in result:
            continue
            
        gt_type = result['ground_truth']
        is_correct = result['video_results'] > 0.5  # 假设>0.5为正确
        predictions = result.get('predictions', [])
        
        type_stats[gt_type]['total'] += 1
        if is_correct:
            type_stats[gt_type]['correct'] += 1
        else:
            type_stats[gt_type]['wrong_predictions'].append({
                'video_path': result['video_path'],
                'predictions': predictions
            })
    
    # 计算召回率
    recall_by_type = {}
    total_correct = 0
    total_samples = 0
    
    for motion_type, stats in type_stats.items():
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        recall_by_type[motion_type] = {
            'recall': recall,
            'correct': stats['correct'], 
            'total': stats['total'],
            'wrong_count': stats['total'] - stats['correct'],
            'error_examples': stats['wrong_predictions'][:3]  # 保留前3个错误例子
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


def enhanced_camera_motion_evaluation(prompt_dict_ls, camera, save_visualizations=False, **kwargs):
    """
    增强版的camera motion评测函数，返回详细的召回率信息
    这个函数是对原camera_motion函数的扩展
    
    Args:
        prompt_dict_ls: prompt字典列表
        camera: CameraPredict实例
        save_visualizations: 是否保存可视化
        **kwargs: 其他参数
    
    Returns:
        overall_score: 总体得分
        detailed_results: 详细结果，包含召回率信息
    """
    from tqdm import tqdm
    import cv2
    import decord
    import os
    from vbench2.utils import split_video_into_scenes
    
    all_video_results = []
    
    if save_visualizations:
        vis_output_dir = "./camera_motion_enhanced_visualizations"
        os.makedirs(vis_output_dir, exist_ok=True)

    for prompt_dict in tqdm(prompt_dict_ls, desc="评测运镜类型"):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        
        for idx, video_path in enumerate(video_paths):
            try:
                # 视频预处理（与原函数保持一致）
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
                
                # 可视化（可选）
                if save_visualizations:
                    vis_save_dir = os.path.join(vis_output_dir, f"{label}_video_{idx}")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    camera.infer(video, fps, end_frame, save_video=True, save_dir=vis_save_dir)
                
                # 预测
                predict_results = camera.predict(video, fps, end_frame)
                video_score = 1.0 if label in predict_results else 0.0
                
                # 保存详细结果
                video_result = {
                    'video_path': video_path,
                    'video_results': video_score,
                    'ground_truth': label,
                    'predictions': predict_results,
                    'video_info': {
                        'frame_count': frame_count,
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    }
                }
                all_video_results.append(video_result)
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue
    
    # 计算召回率指标
    recall_metrics = calculate_recall_metrics(all_video_results)
    
    # 打印召回率报告
    print_recall_report(recall_metrics)
    
    # 返回结果
    overall_score = recall_metrics['overall_recall']
    detailed_results = {
        'video_results': all_video_results,
        'recall_metrics': recall_metrics
    }
    
    return overall_score, detailed_results


def print_recall_report(recall_metrics):
    """
    打印格式化的召回率报告
    
    Args:
        recall_metrics: calculate_recall_metrics函数的返回结果
    """
    print(f"\n{'='*80}")
    print("Camera Motion 召回率分析报告")
    print(f"{'='*80}")
    
    print(f"总体召回率: {recall_metrics['overall_recall']:.3f}")
    print(f"总样本数: {recall_metrics['total_samples']}")
    print(f"正确预测数: {recall_metrics['total_correct']}")
    print(f"运镜类型数: {recall_metrics['type_count']}")
    
    print(f"\n{'运镜类型':<20} {'总数':<8} {'正确':<8} {'错误':<8} {'召回率':<10}")
    print(f"{'-'*70}")
    
    # 按召回率排序显示
    sorted_types = sorted(recall_metrics['recall_by_type'].items(), 
                         key=lambda x: x[1]['recall'], reverse=True)
    
    for motion_type, stats in sorted_types:
        print(f"{motion_type:<20} {stats['total']:<8} {stats['correct']:<8} "
              f"{stats['wrong_count']:<8} {stats['recall']:<10.3f}")
    
    print(f"{'-'*70}")
    
    # 性能分级统计
    high_perf = sum(1 for _, stats in sorted_types if stats['recall'] >= 0.8)
    mid_perf = sum(1 for _, stats in sorted_types if 0.5 <= stats['recall'] < 0.8)
    low_perf = sum(1 for _, stats in sorted_types if stats['recall'] < 0.5)
    
    print(f"\n性能分级:")
    print(f"  高性能 (≥80%): {high_perf} 种")
    print(f"  中等性能 (50%-80%): {mid_perf} 种") 
    print(f"  低性能 (<50%): {low_perf} 种")
    
    # 显示表现最差的运镜类型
    worst_types = [item for item in sorted_types if item[1]['recall'] < 0.5]
    if worst_types:
        print(f"\n表现较差的运镜类型:")
        for motion_type, stats in worst_types[:5]:  # 最多显示5种
            print(f"  {motion_type}: {stats['recall']:.3f} (正确: {stats['correct']}/{stats['total']})")
            # 显示错误预测示例
            if stats['error_examples']:
                example = stats['error_examples'][0]
                print(f"    错误预测示例: {example['predictions']}")


def save_recall_analysis_to_json(recall_metrics, filename="camera_motion_recall_analysis.json"):
    """
    将召回率分析结果保存为JSON文件
    
    Args:
        recall_metrics: 召回率分析结果
        filename: 保存的文件名
    """
    # 准备可序列化的数据
    serializable_data = {
        'summary': {
            'overall_recall': recall_metrics['overall_recall'],
            'total_samples': recall_metrics['total_samples'],
            'total_correct': recall_metrics['total_correct'],
            'type_count': recall_metrics['type_count']
        },
        'recall_by_type': {}
    }
    
    # 转换recall_by_type数据
    for motion_type, stats in recall_metrics['recall_by_type'].items():
        serializable_data['recall_by_type'][motion_type] = {
            'recall': stats['recall'],
            'correct': stats['correct'],
            'total': stats['total'],
            'wrong_count': stats['wrong_count'],
            'error_examples': [
                {
                    'video_path': ex.get('video_path', ''),
                    'predictions': ex.get('predictions', [])
                }
                for ex in stats.get('error_examples', [])
            ]
        }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"召回率分析结果已保存到: {filename}")


# 使用示例函数
def example_usage():
    """
    使用示例
    """
    # 假设已经有了camera对象和prompt_dict_ls
    # from vbench2.camera_motion import CameraPredict
    # from vbench2.utils import load_dimension_info
    
    # device = "cuda"
    # submodules_dict = {
    #     "repo": "facebookresearch/co-tracker", 
    #     "model": "cotracker2_online"
    # }
    # camera = CameraPredict(device, submodules_dict)
    # _, prompt_dict_ls = load_dimension_info("VBench-2.0/prompts", dimension='camera_motion', lang='en')
    
    # # 使用增强版评测函数
    # overall_score, detailed_results = enhanced_camera_motion_evaluation(
    #     prompt_dict_ls, camera, save_visualizations=False
    # )
    
    # # 保存分析结果
    # save_recall_analysis_to_json(detailed_results['recall_metrics'])
    
    print("示例用法请参考注释中的代码")


if __name__ == "__main__":
    example_usage() 