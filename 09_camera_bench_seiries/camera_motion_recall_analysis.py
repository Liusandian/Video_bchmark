import numpy as np
from collections import defaultdict
from tqdm import tqdm
import json
import os


def compute_camera_motion_recall(json_dir, device, submodules_dict, save_visualizations=False, detailed_output=False, **kwargs):
    """
    计算VBench camera motion维度评测中每种运镜类型的召回率
    
    Args:
        json_dir: 包含评测数据的JSON目录
        device: 设备类型 (cuda/cpu)
        submodules_dict: CoTracker模型配置字典
        save_visualizations: 是否保存可视化结果
        detailed_output: 是否输出详细的预测结果分析
        **kwargs: 其他参数
        
    Returns:
        overall_recall: 总体召回率
        recall_by_type: 各运镜类型的召回率字典
        detailed_results: 详细的预测结果（如果detailed_output=True）
    """
    # 导入必要的模块
    from vbench2.camera_motion import CameraPredict
    from vbench2.utils import load_dimension_info
    
    # 初始化相机预测器
    camera = CameraPredict(device, submodules_dict)
    
    # 加载评测数据
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    
    # 统计数据结构
    type_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'predictions': []})
    all_predictions = []
    
    if save_visualizations:
        vis_output_dir = "./camera_motion_recall_visualizations"
        os.makedirs(vis_output_dir, exist_ok=True)

    print(f"开始评测 {len(prompt_dict_ls)} 种运镜类型...")
    
    for prompt_dict in tqdm(prompt_dict_ls, desc="评测运镜类型"):
        ground_truth_label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        
        print(f"\n评测运镜类型: {ground_truth_label}")
        print(f"视频数量: {len(video_paths)}")
        
        for idx, video_path in enumerate(tqdm(video_paths, desc=f"处理{ground_truth_label}视频", leave=False)):
            try:
                # 视频预处理
                import decord
                import cv2
                from vbench2.utils import split_video_into_scenes
                
                # 场景分割
                end_frame = -1
                scene_list = split_video_into_scenes(video_path, 5.0)
                if len(scene_list) != 0:
                    end_frame = int(scene_list[0][1].get_frames())
                
                # 读取视频
                video_reader = decord.VideoReader(video_path)
                video = video_reader.get_batch(range(len(video_reader))) 
                frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
                video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
                
                # 获取FPS
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                # 如果需要保存可视化
                if save_visualizations:
                    vis_save_dir = os.path.join(vis_output_dir, f"{ground_truth_label}_video_{idx}")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    camera.infer(video, fps, end_frame, save_video=True, save_dir=vis_save_dir, visualization_type="grid")
                
                # 预测运镜类型
                predict_results = camera.predict(video, fps, end_frame)
                
                # 判断预测是否正确
                is_correct = ground_truth_label in predict_results
                
                # 统计结果
                type_stats[ground_truth_label]['total'] += 1
                if is_correct:
                    type_stats[ground_truth_label]['correct'] += 1
                
                # 记录详细预测结果
                prediction_detail = {
                    'video_path': video_path,
                    'ground_truth': ground_truth_label,
                    'predictions': predict_results,
                    'is_correct': is_correct,
                    'video_info': {
                        'frame_count': frame_count,
                        'resolution': f"{width}x{height}",
                        'fps': fps,
                        'end_frame': end_frame
                    }
                }
                
                type_stats[ground_truth_label]['predictions'].append(prediction_detail)
                all_predictions.append(prediction_detail)
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue
    
    # 计算召回率
    recall_by_type = {}
    total_correct = 0
    total_samples = 0
    
    print(f"\n{'='*60}")
    print(f"{'运镜类型':<20} {'总数':<8} {'正确':<8} {'召回率':<10} {'错误预测示例'}")
    print(f"{'='*60}")
    
    for motion_type, stats in type_stats.items():
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        recall_by_type[motion_type] = {
            'recall': recall,
            'correct': stats['correct'],
            'total': stats['total'],
            'sample_count': stats['total']
        }
        
        total_correct += stats['correct']
        total_samples += stats['total']
        
        # 找一些错误预测的例子
        wrong_predictions = [p for p in stats['predictions'] if not p['is_correct']]
        error_examples = []
        for wrong_pred in wrong_predictions[:3]:  # 最多显示3个错误例子
            error_examples.append(f"{wrong_pred['predictions']}")
        
        error_str = "; ".join(error_examples) if error_examples else "无"
        
        print(f"{motion_type:<20} {stats['total']:<8} {stats['correct']:<8} {recall:<10.3f} {error_str}")
    
    overall_recall = total_correct / total_samples if total_samples > 0 else 0.0
    
    print(f"{'='*60}")
    print(f"{'总体召回率':<20} {total_samples:<8} {total_correct:<8} {overall_recall:<10.3f}")
    print(f"{'='*60}")
    
    # 分析预测模式
    print(f"\n{'='*60}")
    print("预测模式分析:")
    print(f"{'='*60}")
    
    # 统计所有出现的预测类型
    all_predicted_types = set()
    for prediction in all_predictions:
        all_predicted_types.update(prediction['predictions'])
    
    print(f"检测到的所有运镜类型: {sorted(all_predicted_types)}")
    
    # 混淆矩阵式的分析
    confusion_analysis = defaultdict(lambda: defaultdict(int))
    for prediction in all_predictions:
        gt = prediction['ground_truth']
        for pred in prediction['predictions']:
            confusion_analysis[gt][pred] += 1
    
    print(f"\n运镜类型混淆分析 (Ground Truth -> Predictions):")
    for gt_type in sorted(confusion_analysis.keys()):
        pred_counts = confusion_analysis[gt_type]
        total_gt = type_stats[gt_type]['total']
        print(f"\n{gt_type} (总计 {total_gt}):")
        for pred_type, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_gt * 100
            print(f"  -> {pred_type}: {count} ({percentage:.1f}%)")
    
    # 准备返回结果
    results = {
        'overall_recall': overall_recall,
        'recall_by_type': recall_by_type,
        'total_samples': total_samples,
        'total_correct': total_correct
    }
    
    if detailed_output:
        results['detailed_predictions'] = all_predictions
        results['confusion_analysis'] = dict(confusion_analysis)
        results['type_statistics'] = dict(type_stats)
    
    return results


def analyze_camera_motion_performance(results, save_report=True, output_file="camera_motion_recall_report.json"):
    """
    深入分析camera motion预测性能
    
    Args:
        results: compute_camera_motion_recall函数的返回结果
        save_report: 是否保存分析报告
        output_file: 报告文件名
        
    Returns:
        analysis_report: 性能分析报告
    """
    
    recall_by_type = results['recall_by_type']
    overall_recall = results['overall_recall']
    
    # 按召回率排序
    sorted_types = sorted(recall_by_type.items(), key=lambda x: x[1]['recall'], reverse=True)
    
    # 性能分类
    high_performance = []  # 召回率 >= 0.8
    medium_performance = []  # 0.5 <= 召回率 < 0.8  
    low_performance = []  # 召回率 < 0.5
    
    for motion_type, stats in sorted_types:
        recall = stats['recall']
        if recall >= 0.8:
            high_performance.append((motion_type, recall))
        elif recall >= 0.5:
            medium_performance.append((motion_type, recall))
        else:
            low_performance.append((motion_type, recall))
    
    # 样本数量分析
    sample_distribution = {motion_type: stats['total'] for motion_type, stats in recall_by_type.items()}
    
    # 生成分析报告
    analysis_report = {
        'summary': {
            'overall_recall': overall_recall,
            'total_motion_types': len(recall_by_type),
            'total_samples': results['total_samples'],
            'total_correct': results['total_correct']
        },
        'performance_classification': {
            'high_performance': {
                'types': high_performance,
                'count': len(high_performance),
                'description': '召回率 >= 80%'
            },
            'medium_performance': {
                'types': medium_performance,
                'count': len(medium_performance),
                'description': '50% <= 召回率 < 80%'
            },
            'low_performance': {
                'types': low_performance,
                'count': len(low_performance),
                'description': '召回率 < 50%'
            }
        },
        'sample_distribution': sample_distribution,
        'detailed_recall': recall_by_type
    }
    
    # 打印分析报告
    print(f"\n{'='*80}")
    print("Camera Motion 性能分析报告")
    print(f"{'='*80}")
    
    print(f"总体召回率: {overall_recall:.3f}")
    print(f"运镜类型总数: {len(recall_by_type)}")
    print(f"总样本数: {results['total_samples']}")
    print(f"正确预测数: {results['total_correct']}")
    
    print(f"\n性能分级:")
    print(f"  高性能 (召回率≥80%): {len(high_performance)} 种")
    for motion_type, recall in high_performance:
        print(f"    - {motion_type}: {recall:.3f}")
    
    print(f"  中等性能 (50%≤召回率<80%): {len(medium_performance)} 种")
    for motion_type, recall in medium_performance:
        print(f"    - {motion_type}: {recall:.3f}")
    
    print(f"  低性能 (召回率<50%): {len(low_performance)} 种")
    for motion_type, recall in low_performance:
        print(f"    - {motion_type}: {recall:.3f}")
    
    print(f"\n样本分布:")
    sorted_samples = sorted(sample_distribution.items(), key=lambda x: x[1], reverse=True)
    for motion_type, count in sorted_samples:
        print(f"  {motion_type}: {count} 个样本")
    
    if save_report:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        print(f"\n分析报告已保存到: {output_file}")
    
    return analysis_report


def main():
    """
    主函数，演示如何使用召回率分析功能
    """
    # 配置参数
    json_dir = "VBench-2.0/prompts"  # 根据实际路径调整
    device = "cuda"
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 计算召回率
    print("开始计算Camera Motion召回率...")
    results = compute_camera_motion_recall(
        json_dir=json_dir,
        device=device,
        submodules_dict=submodules_dict,
        save_visualizations=False,  # 设为True可保存可视化结果
        detailed_output=True
    )
    
    # 分析性能
    print("\n开始性能分析...")
    analysis_report = analyze_camera_motion_performance(
        results=results,
        save_report=True,
        output_file="camera_motion_recall_analysis.json"
    )
    
    print(f"\n分析完成！总体召回率: {results['overall_recall']:.3f}")


if __name__ == "__main__":
    main() 