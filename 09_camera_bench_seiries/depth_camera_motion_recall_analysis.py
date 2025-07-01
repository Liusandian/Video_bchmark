import numpy as np
from collections import defaultdict, Counter
import json
import os
from camera_motion_with_depth import DepthBasedCameraPredict, camera_motion_with_depth


def calculate_depth_camera_recall_metrics(video_results):
    """
    基于深度估计相机运动检测结果计算召回率
    
    Args:
        video_results: 来自depth-based camera motion函数的video_results列表
                      每个元素包含: {'video_path': str, 'video_results': float, 'ground_truth': str, 'predictions': list}
    
    Returns:
        recall_metrics: 包含各种召回率统计的字典
    """
    
    # 统计每种运镜类型的结果
    type_stats = defaultdict(lambda: {
        'total': 0, 
        'correct': 0, 
        'wrong_predictions': [],
        'depth_background_success': 0,  # 成功使用深度估计的次数
        'traditional_fallback': 0       # 回退到传统方法的次数
    })
    
    # 全局统计
    total_videos = len(video_results)
    total_correct = 0
    depth_method_used = 0
    traditional_fallback = 0
    
    for result in video_results:
        if 'ground_truth' not in result:
            continue
            
        gt_type = result['ground_truth']
        is_correct = result['video_results'] > 0.5
        predictions = result.get('predictions', [])
        
        # 统计该类型
        type_stats[gt_type]['total'] += 1
        if is_correct:
            type_stats[gt_type]['correct'] += 1
            total_correct += 1
        else:
            type_stats[gt_type]['wrong_predictions'].append(predictions)
        
        # 检查是否使用了深度估计（通过预测结果的复杂度判断）
        if len(predictions) > 1 or any('dolly' in p for p in predictions):
            type_stats[gt_type]['depth_background_success'] += 1
            depth_method_used += 1
        else:
            type_stats[gt_type]['traditional_fallback'] += 1
            traditional_fallback += 1
    
    # 计算各种指标
    recall_metrics = {
        'overall_accuracy': total_correct / total_videos if total_videos > 0 else 0,
        'depth_method_usage_rate': depth_method_used / total_videos if total_videos > 0 else 0,
        'traditional_fallback_rate': traditional_fallback / total_videos if total_videos > 0 else 0,
        'per_type_recall': {},
        'per_type_stats': {},
        'confusion_matrix': {},
        'depth_enhancement_effectiveness': {}
    }
    
    # 计算每种类型的召回率
    for motion_type, stats in type_stats.items():
        recall = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        depth_success_rate = stats['depth_background_success'] / stats['total'] if stats['total'] > 0 else 0
        
        recall_metrics['per_type_recall'][motion_type] = recall
        recall_metrics['per_type_stats'][motion_type] = {
            'total_samples': stats['total'],
            'correct_predictions': stats['correct'],
            'recall': recall,
            'depth_method_success_rate': depth_success_rate,
            'traditional_fallback_count': stats['traditional_fallback'],
            'wrong_prediction_patterns': Counter([tuple(sorted(pred)) for pred in stats['wrong_predictions']])
        }
    
    # 分析深度估计的增强效果
    recall_metrics['depth_enhancement_effectiveness'] = analyze_depth_enhancement_effect(type_stats)
    
    return recall_metrics


def analyze_depth_enhancement_effect(type_stats):
    """
    分析深度估计对不同运镜类型检测的增强效果
    
    Args:
        type_stats: 运镜类型统计数据
        
    Returns:
        enhancement_analysis: 深度增强效果分析
    """
    enhancement_analysis = {
        'high_benefit_types': [],      # 从深度估计获益较大的运镜类型
        'medium_benefit_types': [],    # 中等获益的运镜类型
        'low_benefit_types': [],       # 获益较小的运镜类型
        'depth_sensitive_motions': [], # 对深度信息敏感的运动类型
        'recommendations': []          # 改进建议
    }
    
    for motion_type, stats in type_stats.items():
        if stats['total'] == 0:
            continue
            
        depth_success_rate = stats['depth_background_success'] / stats['total']
        recall = stats['correct'] / stats['total']
        
        # 分析深度估计的效果
        if depth_success_rate > 0.7 and recall > 0.8:
            enhancement_analysis['high_benefit_types'].append({
                'type': motion_type,
                'depth_usage': depth_success_rate,
                'recall': recall,
                'samples': stats['total']
            })
        elif depth_success_rate > 0.4 and recall > 0.6:
            enhancement_analysis['medium_benefit_types'].append({
                'type': motion_type,
                'depth_usage': depth_success_rate,
                'recall': recall,
                'samples': stats['total']
            })
        else:
            enhancement_analysis['low_benefit_types'].append({
                'type': motion_type,
                'depth_usage': depth_success_rate,
                'recall': recall,
                'samples': stats['total'],
                'issues': stats['wrong_predictions'][:3]  # 显示前3个错误案例
            })
        
        # 识别对深度信息敏感的运动类型
        if motion_type in ['zoom_in', 'zoom_out', 'dolly_in', 'dolly_out', 'pan_left', 'pan_right']:
            enhancement_analysis['depth_sensitive_motions'].append({
                'type': motion_type,
                'expected_depth_benefit': 'high',
                'actual_depth_usage': depth_success_rate,
                'performance_gap': max(0, 0.8 - recall)  # 期望召回率与实际的差距
            })
    
    # 生成改进建议
    enhancement_analysis['recommendations'] = generate_improvement_recommendations(enhancement_analysis)
    
    return enhancement_analysis


def generate_improvement_recommendations(enhancement_analysis):
    """
    基于分析结果生成改进建议
    """
    recommendations = []
    
    # 检查低效益类型
    if enhancement_analysis['low_benefit_types']:
        low_benefit_types = [item['type'] for item in enhancement_analysis['low_benefit_types']]
        recommendations.append({
            'issue': 'depth_estimation_low_benefit',
            'affected_types': low_benefit_types,
            'suggestion': '考虑优化深度估计模型或阈值参数，特别针对这些运镜类型',
            'priority': 'high'
        })
    
    # 检查深度敏感运动的性能
    depth_sensitive_issues = [
        item for item in enhancement_analysis['depth_sensitive_motions'] 
        if item['performance_gap'] > 0.2
    ]
    if depth_sensitive_issues:
        recommendations.append({
            'issue': 'depth_sensitive_motion_underperforming',
            'affected_types': [item['type'] for item in depth_sensitive_issues],
            'suggestion': '这些运动类型理论上应该从深度估计中获益更多，建议检查前景背景分割质量',
            'priority': 'medium'
        })
    
    # 检查总体深度使用率
    avg_depth_usage = np.mean([
        item['depth_usage'] for item_list in [
            enhancement_analysis['high_benefit_types'],
            enhancement_analysis['medium_benefit_types'],
            enhancement_analysis['low_benefit_types']
        ] for item in item_list
    ])
    
    if avg_depth_usage < 0.5:
        recommendations.append({
            'issue': 'low_depth_method_usage',
            'suggestion': '深度估计方法使用率较低，可能需要降低深度分割的阈值或改进模型加载',
            'priority': 'high'
        })
    
    return recommendations


def print_depth_camera_recall_report(recall_metrics, detailed=True):
    """
    打印基于深度估计的相机运动召回率报告
    
    Args:
        recall_metrics: 召回率指标字典
        detailed: 是否显示详细信息
    """
    
    print("\n" + "="*60)
    print("🎬 基于深度估计的相机运动检测召回率报告")
    print("="*60)
    
    # 总体指标
    print(f"\n📊 总体性能指标:")
    print(f"   总体准确率: {recall_metrics['overall_accuracy']:.3f}")
    print(f"   深度方法使用率: {recall_metrics['depth_method_usage_rate']:.3f}")
    print(f"   传统方法回退率: {recall_metrics['traditional_fallback_rate']:.3f}")
    
    # 各运镜类型的召回率
    print(f"\n🎯 各运镜类型召回率:")
    per_type_recall = recall_metrics['per_type_recall']
    per_type_stats = recall_metrics['per_type_stats']
    
    # 按召回率排序
    sorted_types = sorted(per_type_recall.items(), key=lambda x: x[1], reverse=True)
    
    for motion_type, recall in sorted_types:
        stats = per_type_stats[motion_type]
        depth_rate = stats['depth_method_success_rate']
        
        # 性能等级标识
        if recall >= 0.8:
            level = "🟢 优秀"
        elif recall >= 0.6:
            level = "🟡 良好"
        else:
            level = "🔴 需改进"
            
        print(f"   {motion_type:15} | 召回率: {recall:.3f} | 深度使用: {depth_rate:.3f} | {level}")
        
        if detailed and stats['total_samples'] > 0:
            print(f"                     样本数: {stats['total_samples']}, "
                  f"正确数: {stats['correct_predictions']}, "
                  f"回退数: {stats['traditional_fallback_count']}")
    
    # 深度增强效果分析
    enhancement = recall_metrics['depth_enhancement_effectiveness']
    
    print(f"\n🔍 深度估计增强效果分析:")
    
    if enhancement['high_benefit_types']:
        print(f"   高效益运镜类型 ({len(enhancement['high_benefit_types'])}种):")
        for item in enhancement['high_benefit_types']:
            print(f"     - {item['type']}: 召回率 {item['recall']:.3f}, 深度使用率 {item['depth_usage']:.3f}")
    
    if enhancement['medium_benefit_types']:
        print(f"   中等效益运镜类型 ({len(enhancement['medium_benefit_types'])}种):")
        for item in enhancement['medium_benefit_types']:
            print(f"     - {item['type']}: 召回率 {item['recall']:.3f}, 深度使用率 {item['depth_usage']:.3f}")
    
    if enhancement['low_benefit_types']:
        print(f"   低效益运镜类型 ({len(enhancement['low_benefit_types'])}种):")
        for item in enhancement['low_benefit_types']:
            print(f"     - {item['type']}: 召回率 {item['recall']:.3f}, 深度使用率 {item['depth_usage']:.3f}")
    
    # 改进建议
    if enhancement['recommendations']:
        print(f"\n💡 改进建议:")
        for i, rec in enumerate(enhancement['recommendations'], 1):
            priority_icon = "🔥" if rec['priority'] == 'high' else "⚠️" if rec['priority'] == 'medium' else "💭"
            print(f"   {i}. {priority_icon} {rec['suggestion']}")
            if 'affected_types' in rec:
                print(f"      影响类型: {', '.join(rec['affected_types'])}")
    
    print("\n" + "="*60)


def enhanced_camera_motion_evaluation_with_depth(json_dir, device, submodules_dict, 
                                                 save_visualizations=False, 
                                                 detailed_output=True, 
                                                 **kwargs):
    """
    增强版的深度估计相机运动评测，包含详细的召回率分析
    
    Args:
        json_dir: 测试数据目录
        device: 设备类型
        submodules_dict: 模型配置
        save_visualizations: 是否保存可视化
        detailed_output: 是否输出详细分析结果
        **kwargs: 其他参数
        
    Returns:
        overall_score: 总体得分
        detailed_results: 详细结果包含召回率分析
    """
    
    # 执行深度估计相机运动评测
    avg_score, video_results = compute_camera_motion_with_depth(
        json_dir, device, submodules_dict, save_visualizations=save_visualizations, **kwargs)
    
    # 计算详细的召回率指标
    recall_metrics = calculate_depth_camera_recall_metrics(video_results)
    
    # 输出报告
    if detailed_output:
        print_depth_camera_recall_report(recall_metrics, detailed=True)
    
    # 保存详细结果到文件
    detailed_results = {
        'overall_score': avg_score,
        'recall_metrics': recall_metrics,
        'video_results': video_results,
        'method': 'depth_enhanced_camera_motion',
        'total_videos': len(video_results),
        'depth_enhancement_summary': {
            'depth_method_usage_rate': recall_metrics['depth_method_usage_rate'],
            'high_benefit_types_count': len(recall_metrics['depth_enhancement_effectiveness']['high_benefit_types']),
            'improvement_recommendations_count': len(recall_metrics['depth_enhancement_effectiveness']['recommendations'])
        }
    }
    
    return avg_score, detailed_results


def compare_traditional_vs_depth_enhanced(json_dir, device, submodules_dict, **kwargs):
    """
    比较传统方法与深度增强方法的性能差异
    
    Args:
        json_dir: 测试数据目录
        device: 设备类型
        submodules_dict: 模型配置
        **kwargs: 其他参数
        
    Returns:
        comparison_results: 比较结果
    """
    
    print("🔬 开始比较传统方法与深度增强方法...")
    
    # 执行深度增强方法评测
    depth_score, depth_results = enhanced_camera_motion_evaluation_with_depth(
        json_dir, device, submodules_dict, detailed_output=False, **kwargs)
    
    # 这里可以添加传统方法的评测（需要导入原始模块）
    # traditional_score, traditional_results = compute_camera_motion(...)
    
    # 暂时返回深度增强的结果，等待传统方法的对比
    comparison_results = {
        'depth_enhanced': {
            'score': depth_score,
            'details': depth_results
        },
        # 'traditional': {
        #     'score': traditional_score,
        #     'details': traditional_results
        # },
        'improvement_analysis': {
            'depth_method_usage': depth_results['recall_metrics']['depth_method_usage_rate'],
            'performance_gain': depth_score  # 相对于基线的改进
        }
    }
    
    print(f"\n📈 方法比较结果:")
    print(f"   深度增强方法得分: {depth_score:.3f}")
    print(f"   深度方法使用率: {depth_results['recall_metrics']['depth_method_usage_rate']:.3f}")
    
    return comparison_results


def save_depth_camera_analysis_report(results, output_file="depth_camera_analysis_report.json"):
    """
    保存深度相机分析报告到JSON文件
    
    Args:
        results: 分析结果字典
        output_file: 输出文件路径
    """
    
    # 转换numpy类型为Python类型以便JSON序列化
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_compatible_results = convert_for_json(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 分析报告已保存到: {output_file}")


# 使用示例
if __name__ == "__main__":
    print("深度估计相机运动召回率分析模块")
    print("主要功能:")
    print("1. 分析深度估计方法对不同运镜类型的检测效果")
    print("2. 计算各运镜类型的召回率和深度方法使用率")
    print("3. 提供性能改进建议")
    print("4. 生成详细的分析报告")
    
    # 示例调用
    # json_dir = "path/to/test/data"
    # device = "cuda"
    # submodules_dict = {"repo": "facebookresearch/co-tracker", "model": "cotracker2_online"}
    # 
    # score, results = enhanced_camera_motion_evaluation_with_depth(
    #     json_dir, device, submodules_dict, save_visualizations=True)
    # 
    # save_depth_camera_analysis_report(results, "depth_camera_analysis.json") 