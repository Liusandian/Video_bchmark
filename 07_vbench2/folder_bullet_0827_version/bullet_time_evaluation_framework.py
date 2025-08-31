"""
子弹时间运镜综合评估框架
集成多种姿态估计和运镜检测方法，提供完整的评估解决方案
"""

import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
import json
import os
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# 导入各个模块
from .pose_estimation_bullet_time import HumanPoseEstimator, BulletTimeCameraMotionEvaluator
from .optical_flow_rotation import MultiMethodRotationEstimator
from .depth_3d_pose_estimation import Depth3DPoseEstimator, AdvancedBulletTimeAnalyzer
from .enhanced_camera_motion import EnhancedCameraPredict
from .utils import load_dimension_info, split_video_into_scenes


class BulletTimeEvaluationFramework:
    """
    子弹时间运镜综合评估框架
    """
    
    def __init__(self, device='cuda', config: Optional[Dict] = None):
        self.device = device
        self.config = config or self._get_default_config()
        
        # 初始化各个评估模块
        self.pose_estimator = None
        self.optical_flow_estimator = None
        self.depth_3d_estimator = None
        self.camera_motion_predictor = None
        self.advanced_analyzer = None
        
        self._initialize_modules()
    
    def _get_default_config(self) -> Dict:
        """
        获取默认配置
        """
        return {
            'enable_pose_estimation': True,
            'enable_optical_flow': True,
            'enable_depth_3d': True,
            'enable_camera_motion': True,
            'fusion_strategy': 'weighted_average',
            'confidence_threshold': 0.5,
            'smoothing_window': 5,
            'output_visualization': True,
            'save_intermediate_results': False
        }
    
    def _initialize_modules(self):
        """
        初始化各个评估模块
        """
        try:
            if self.config['enable_pose_estimation']:
                self.pose_estimator = HumanPoseEstimator(self.device)
                print("✓ Pose estimation module initialized")
        except Exception as e:
            print(f"✗ Failed to initialize pose estimation: {e}")
        
        try:
            if self.config['enable_optical_flow']:
                self.optical_flow_estimator = MultiMethodRotationEstimator()
                print("✓ Optical flow estimation module initialized")
        except Exception as e:
            print(f"✗ Failed to initialize optical flow estimation: {e}")
        
        try:
            if self.config['enable_depth_3d']:
                self.depth_3d_estimator = Depth3DPoseEstimator(self.device)
                self.advanced_analyzer = AdvancedBulletTimeAnalyzer(self.device)
                print("✓ Depth-based 3D estimation module initialized")
        except Exception as e:
            print(f"✗ Failed to initialize depth 3D estimation: {e}")
        
        # Camera motion predictor 需要 submodules_dict，暂时跳过
        print("Camera motion predictor will be initialized when needed")
    
    def evaluate_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict:
        """
        评估单个视频的子弹时间运镜效果
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录（可选）
        
        Returns:
            evaluation_result: 完整的评估结果
        """
        print(f"Evaluating video: {video_path}")
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return {'success': False, 'error': f'Video file not found: {video_path}'}
        
        # 创建输出目录
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 读取视频
        video_data = self._load_video(video_path)
        if not video_data['success']:
            return video_data
        
        # 多方法评估
        evaluation_results = {
            'video_path': video_path,
            'video_info': video_data['info'],
            'methods': {},
            'fusion_result': {},
            'visualization_paths': []
        }
        
        # 方法1: 基于姿态估计
        if self.pose_estimator:
            pose_result = self._evaluate_with_pose_estimation(video_data)
            evaluation_results['methods']['pose_estimation'] = pose_result
        
        # 方法2: 基于光流
        if self.optical_flow_estimator:
            optical_flow_result = self._evaluate_with_optical_flow(video_data)
            evaluation_results['methods']['optical_flow'] = optical_flow_result
        
        # 方法3: 基于深度3D
        if self.depth_3d_estimator:
            depth_3d_result = self._evaluate_with_depth_3d(video_data)
            evaluation_results['methods']['depth_3d'] = depth_3d_result
        
        # 融合多种方法的结果
        fusion_result = self._fuse_evaluation_results(evaluation_results['methods'])
        evaluation_results['fusion_result'] = fusion_result
        
        # 生成可视化
        if self.config['output_visualization'] and output_dir:
            viz_paths = self._generate_visualizations(evaluation_results, output_dir)
            evaluation_results['visualization_paths'] = viz_paths
        
        # 保存结果
        if output_dir:
            result_path = os.path.join(output_dir, 'evaluation_result.json')
            self._save_evaluation_result(evaluation_results, result_path)
        
        return evaluation_results
    
    def _load_video(self, video_path: str) -> Dict:
        """
        加载视频数据
        """
        try:
            # 使用OpenCV获取基本信息
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 读取所有帧
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            # 限制帧数（如果视频太长）
            max_frames = 300  # 约10秒（30fps）
            if len(frames) > max_frames:
                # 均匀采样
                indices = np.linspace(0, len(frames)-1, max_frames, dtype=int)
                frames = [frames[i] for i in indices]
            
            return {
                'success': True,
                'frames': frames,
                'info': {
                    'fps': fps,
                    'frame_count': len(frames),
                    'original_frame_count': frame_count,
                    'width': width,
                    'height': height,
                    'duration': duration
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _evaluate_with_pose_estimation(self, video_data: Dict) -> Dict:
        """
        使用姿态估计方法评估
        """
        try:
            frames = video_data['frames']
            pose_sequence = []
            
            for frame_idx, frame in enumerate(tqdm(frames, desc="Pose estimation")):
                # 转换颜色空间
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # MediaPipe姿态检测
                results = self.pose_estimator.pose.process(rgb_frame)
                
                frame_info = {
                    'frame': frame_idx,
                    'yaw': None,
                    'confidence': 0.0
                }
                
                if results.pose_landmarks:
                    # 提取关键点
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.visibility])
                    landmarks = np.array(landmarks)
                    
                    # 估计yaw角度
                    yaw_angle = self.pose_estimator.estimate_yaw_from_keypoints(landmarks)
                    if yaw_angle is not None:
                        frame_info['yaw'] = yaw_angle
                        frame_info['confidence'] = 0.8
                
                pose_sequence.append(frame_info)
            
            # 分析旋转模式
            analysis_result = self._analyze_pose_sequence(pose_sequence)
            
            return {
                'success': True,
                'method': 'pose_estimation',
                'pose_sequence': pose_sequence,
                'analysis': analysis_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'method': 'pose_estimation'}
    
    def _evaluate_with_optical_flow(self, video_data: Dict) -> Dict:
        """
        使用光流方法评估
        """
        try:
            frames = video_data['frames']
            if len(frames) < 2:
                return {'success': False, 'error': 'Need at least 2 frames', 'method': 'optical_flow'}
            
            rotation_sequence = []
            cumulative_rotation = 0.0
            
            for i in tqdm(range(len(frames) - 1), desc="Optical flow analysis"):
                frame1 = frames[i]
                frame2 = frames[i + 1]
                
                # 估计帧间旋转
                rotation_result = self.optical_flow_estimator.estimate_rotation_multi_method(
                    frame1, frame2, mask=None)
                
                frame_rotation = rotation_result.get('final_rotation', 0.0)
                if frame_rotation is not None:
                    cumulative_rotation += frame_rotation
                else:
                    frame_rotation = 0.0
                
                rotation_info = {
                    'frame_pair': (i, i + 1),
                    'rotation': frame_rotation,
                    'cumulative_rotation': cumulative_rotation,
                    'confidence': rotation_result.get('confidence', 0.0)
                }
                
                rotation_sequence.append(rotation_info)
            
            # 分析旋转模式
            analysis_result = self._analyze_optical_flow_sequence(rotation_sequence)
            
            return {
                'success': True,
                'method': 'optical_flow',
                'rotation_sequence': rotation_sequence,
                'analysis': analysis_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'method': 'optical_flow'}
    
    def _evaluate_with_depth_3d(self, video_data: Dict) -> Dict:
        """
        使用深度3D方法评估
        """
        try:
            frames = video_data['frames']
            
            # 首先使用姿态估计获取2D关键点
            keypoints_sequence = []
            for frame in tqdm(frames, desc="Extracting keypoints for 3D"):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_estimator.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.visibility])
                    keypoints_sequence.append(np.array(landmarks))
                else:
                    keypoints_sequence.append(np.array([]))
            
            # 使用高级分析器
            analysis_result = self.advanced_analyzer.analyze_bullet_time_with_depth(
                frames, keypoints_sequence)
            
            return {
                'success': True,
                'method': 'depth_3d',
                'analysis': analysis_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'method': 'depth_3d'}
    
    def _analyze_pose_sequence(self, pose_sequence: List[Dict]) -> Dict:
        """
        分析姿态序列
        """
        valid_yaws = [frame['yaw'] for frame in pose_sequence if frame['yaw'] is not None]
        
        if len(valid_yaws) < 5:
            return {'is_bullet_time': False, 'reason': 'Insufficient pose data'}
        
        # 计算角度变化
        angle_changes = []
        for i in range(1, len(valid_yaws)):
            diff = valid_yaws[i] - valid_yaws[i-1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            angle_changes.append(diff)
        
        total_rotation = sum(angle_changes)
        mean_velocity = np.mean(np.abs(angle_changes))
        velocity_std = np.std(angle_changes)
        
        # 方向一致性
        direction_changes = sum(1 for i in range(1, len(angle_changes)) 
                              if np.sign(angle_changes[i]) != np.sign(angle_changes[i-1]))
        direction_consistency = 1.0 - (direction_changes / max(1, len(angle_changes) - 1))
        
        # 判断是否为子弹时间
        is_bullet_time = (
            abs(total_rotation) > 90 and
            direction_consistency > 0.7 and
            velocity_std < 15  # 相对平滑
        )
        
        motion_type = 'static'
        if is_bullet_time:
            if total_rotation > 0:
                motion_type = 'bullet_time_clockwise'
            else:
                motion_type = 'bullet_time_counterclockwise'
        
        return {
            'is_bullet_time': is_bullet_time,
            'motion_type': motion_type,
            'total_rotation': total_rotation,
            'direction_consistency': direction_consistency,
            'smoothness': 1.0 / (1.0 + velocity_std / 10.0),
            'confidence': direction_consistency * (1.0 / (1.0 + velocity_std / 10.0))
        }
    
    def _analyze_optical_flow_sequence(self, rotation_sequence: List[Dict]) -> Dict:
        """
        分析光流旋转序列
        """
        rotations = [r['rotation'] for r in rotation_sequence if r['rotation'] is not None]
        
        if len(rotations) < 3:
            return {'is_bullet_time': False, 'reason': 'Insufficient optical flow data'}
        
        total_rotation = sum(rotations)
        rotation_std = np.std(rotations)
        
        # 方向一致性
        positive_count = sum(1 for r in rotations if r > 0)
        negative_count = sum(1 for r in rotations if r < 0)
        direction_consistency = max(positive_count, negative_count) / len(rotations)
        
        is_bullet_time = (
            abs(total_rotation) > 45 and
            direction_consistency > 0.7 and
            rotation_std < 10
        )
        
        motion_type = 'static'
        if is_bullet_time:
            motion_type = 'bullet_time_clockwise' if total_rotation > 0 else 'bullet_time_counterclockwise'
        
        return {
            'is_bullet_time': is_bullet_time,
            'motion_type': motion_type,
            'total_rotation': total_rotation,
            'direction_consistency': direction_consistency,
            'smoothness': 1.0 / (1.0 + rotation_std / 5.0),
            'confidence': direction_consistency * (1.0 / (1.0 + rotation_std / 5.0))
        }
    
    def _fuse_evaluation_results(self, method_results: Dict) -> Dict:
        """
        融合多种方法的评估结果
        """
        valid_methods = [result for result in method_results.values() if result['success']]
        
        if not valid_methods:
            return {'success': False, 'error': 'No valid evaluation methods'}
        
        # 收集各方法的结果
        bullet_time_votes = []
        motion_types = []
        confidences = []
        total_rotations = []
        
        for method_result in valid_methods:
            analysis = method_result.get('analysis', {})
            
            if analysis.get('is_bullet_time', False):
                bullet_time_votes.append(1)
                motion_types.append(analysis.get('motion_type', 'unknown'))
                confidences.append(analysis.get('confidence', 0.0))
                total_rotations.append(analysis.get('total_rotation', 0.0))
            else:
                bullet_time_votes.append(0)
                confidences.append(0.0)
        
        # 投票决定是否为子弹时间
        bullet_time_score = np.mean(bullet_time_votes)
        is_bullet_time = bullet_time_score >= 0.5
        
        # 确定运镜类型
        if is_bullet_time and motion_types:
            # 选择最常见的运镜类型
            motion_type = max(set(motion_types), key=motion_types.count)
        else:
            motion_type = 'static'
        
        # 计算融合置信度
        if confidences:
            fusion_confidence = np.mean(confidences)
        else:
            fusion_confidence = 0.0
        
        # 计算平均总旋转
        avg_total_rotation = np.mean(total_rotations) if total_rotations else 0.0
        
        return {
            'success': True,
            'is_bullet_time': is_bullet_time,
            'motion_type': motion_type,
            'confidence': fusion_confidence,
            'bullet_time_score': bullet_time_score,
            'average_total_rotation': avg_total_rotation,
            'num_valid_methods': len(valid_methods),
            'method_agreement': bullet_time_score
        }
    
    def _generate_visualizations(self, evaluation_results: Dict, output_dir: str) -> List[str]:
        """
        生成可视化图表
        """
        viz_paths = []
        
        try:
            # 1. 姿态角度变化图
            if 'pose_estimation' in evaluation_results['methods']:
                pose_result = evaluation_results['methods']['pose_estimation']
                if pose_result['success']:
                    viz_path = self._plot_pose_angles(pose_result, output_dir)
                    if viz_path:
                        viz_paths.append(viz_path)
            
            # 2. 光流旋转图
            if 'optical_flow' in evaluation_results['methods']:
                flow_result = evaluation_results['methods']['optical_flow']
                if flow_result['success']:
                    viz_path = self._plot_optical_flow_rotation(flow_result, output_dir)
                    if viz_path:
                        viz_paths.append(viz_path)
            
            # 3. 综合评估结果图
            viz_path = self._plot_fusion_result(evaluation_results, output_dir)
            if viz_path:
                viz_paths.append(viz_path)
                
        except Exception as e:
            print(f"Visualization generation failed: {e}")
        
        return viz_paths
    
    def _plot_pose_angles(self, pose_result: Dict, output_dir: str) -> Optional[str]:
        """
        绘制姿态角度变化图
        """
        try:
            pose_sequence = pose_result['pose_sequence']
            frames = [p['frame'] for p in pose_sequence]
            yaw_angles = [p['yaw'] if p['yaw'] is not None else np.nan for p in pose_sequence]
            
            plt.figure(figsize=(12, 6))
            plt.plot(frames, yaw_angles, 'b-', linewidth=2, label='Yaw Angle')
            plt.xlabel('Frame')
            plt.ylabel('Yaw Angle (degrees)')
            plt.title('Human Pose Yaw Angle Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加分析信息
            analysis = pose_result['analysis']
            info_text = f"Motion Type: {analysis['motion_type']}\n"
            info_text += f"Total Rotation: {analysis['total_rotation']:.1f}°\n"
            info_text += f"Confidence: {analysis['confidence']:.3f}"
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'pose_angles.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            print(f"Failed to plot pose angles: {e}")
            return None
    
    def _plot_optical_flow_rotation(self, flow_result: Dict, output_dir: str) -> Optional[str]:
        """
        绘制光流旋转图
        """
        try:
            rotation_sequence = flow_result['rotation_sequence']
            frames = [r['frame_pair'][0] for r in rotation_sequence]
            rotations = [r['rotation'] for r in rotation_sequence]
            cumulative = [r['cumulative_rotation'] for r in rotation_sequence]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 帧间旋转
            ax1.plot(frames, rotations, 'r-', linewidth=2, label='Frame-to-frame Rotation')
            ax1.set_xlabel('Frame')
            ax1.set_ylabel('Rotation (degrees)')
            ax1.set_title('Optical Flow Frame-to-frame Rotation')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 累积旋转
            ax2.plot(frames, cumulative, 'g-', linewidth=2, label='Cumulative Rotation')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Cumulative Rotation (degrees)')
            ax2.set_title('Optical Flow Cumulative Rotation')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 添加分析信息
            analysis = flow_result['analysis']
            info_text = f"Motion Type: {analysis['motion_type']}\n"
            info_text += f"Total Rotation: {analysis['total_rotation']:.1f}°\n"
            info_text += f"Confidence: {analysis['confidence']:.3f}"
            
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'optical_flow_rotation.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            print(f"Failed to plot optical flow rotation: {e}")
            return None
    
    def _plot_fusion_result(self, evaluation_results: Dict, output_dir: str) -> Optional[str]:
        """
        绘制融合结果图
        """
        try:
            fusion_result = evaluation_results['fusion_result']
            methods = evaluation_results['methods']
            
            # 创建条形图显示各方法的置信度
            method_names = []
            confidences = []
            
            for method_name, method_result in methods.items():
                if method_result['success']:
                    method_names.append(method_name.replace('_', ' ').title())
                    analysis = method_result.get('analysis', {})
                    confidences.append(analysis.get('confidence', 0.0))
            
            plt.figure(figsize=(10, 6))
            
            # 子图1: 方法置信度对比
            plt.subplot(1, 2, 1)
            bars = plt.bar(method_names, confidences, 
                          color=['skyblue', 'lightcoral', 'lightgreen'][:len(method_names)])
            plt.xlabel('Evaluation Method')
            plt.ylabel('Confidence Score')
            plt.title('Method Confidence Comparison')
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, conf in zip(bars, confidences):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom')
            
            # 子图2: 融合结果
            plt.subplot(1, 2, 2)
            result_text = f"Bullet Time Detection: {'YES' if fusion_result['is_bullet_time'] else 'NO'}\n\n"
            result_text += f"Motion Type: {fusion_result['motion_type']}\n"
            result_text += f"Fusion Confidence: {fusion_result['confidence']:.3f}\n"
            result_text += f"Method Agreement: {fusion_result['method_agreement']:.3f}\n"
            result_text += f"Average Rotation: {fusion_result['average_total_rotation']:.1f}°"
            
            plt.text(0.1, 0.5, result_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title('Fusion Result')
            
            plt.tight_layout()
            viz_path = os.path.join(output_dir, 'fusion_result.png')
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            print(f"Failed to plot fusion result: {e}")
            return None
    
    def _save_evaluation_result(self, evaluation_results: Dict, output_path: str):
        """
        保存评估结果到JSON文件
        """
        try:
            # 转换numpy类型为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            converted_results = convert_numpy(evaluation_results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
                
            print(f"Evaluation results saved to: {output_path}")
            
        except Exception as e:
            print(f"Failed to save evaluation results: {e}")
    
    def evaluate_batch(self, video_list: List[str], output_base_dir: str) -> Dict:
        """
        批量评估多个视频
        """
        batch_results = {
            'total_videos': len(video_list),
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'bullet_time_detected': 0,
            'results': []
        }
        
        for i, video_path in enumerate(tqdm(video_list, desc="Batch evaluation")):
            print(f"\nEvaluating video {i+1}/{len(video_list)}: {os.path.basename(video_path)}")
            
            # 为每个视频创建输出目录
            video_name = Path(video_path).stem
            output_dir = os.path.join(output_base_dir, f"video_{i+1:03d}_{video_name}")
            
            # 评估单个视频
            result = self.evaluate_video(video_path, output_dir)
            
            if result.get('success', True):  # 如果没有success字段，认为成功
                batch_results['successful_evaluations'] += 1
                if result.get('fusion_result', {}).get('is_bullet_time', False):
                    batch_results['bullet_time_detected'] += 1
            else:
                batch_results['failed_evaluations'] += 1
            
            batch_results['results'].append({
                'video_path': video_path,
                'result': result
            })
        
        # 保存批量结果
        batch_summary_path = os.path.join(output_base_dir, 'batch_evaluation_summary.json')
        self._save_evaluation_result(batch_results, batch_summary_path)
        
        # 生成批量统计
        self._generate_batch_statistics(batch_results, output_base_dir)
        
        return batch_results
    
    def _generate_batch_statistics(self, batch_results: Dict, output_dir: str):
        """
        生成批量评估统计
        """
        try:
            # 统计信息
            total = batch_results['total_videos']
            success = batch_results['successful_evaluations']
            bullet_time = batch_results['bullet_time_detected']
            
            # 创建统计图
            plt.figure(figsize=(12, 8))
            
            # 子图1: 评估成功率
            plt.subplot(2, 2, 1)
            labels = ['Successful', 'Failed']
            sizes = [success, total - success]
            colors = ['lightgreen', 'lightcoral']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Evaluation Success Rate')
            
            # 子图2: 子弹时间检测率
            plt.subplot(2, 2, 2)
            labels = ['Bullet Time', 'Other Motion']
            sizes = [bullet_time, success - bullet_time]
            colors = ['gold', 'lightblue']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('Bullet Time Detection Rate')
            
            # 子图3: 置信度分布
            plt.subplot(2, 2, 3)
            confidences = []
            for result_item in batch_results['results']:
                result = result_item['result']
                if result.get('success', True):
                    conf = result.get('fusion_result', {}).get('confidence', 0.0)
                    confidences.append(conf)
            
            if confidences:
                plt.hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                plt.xlabel('Confidence Score')
                plt.ylabel('Number of Videos')
                plt.title('Confidence Score Distribution')
            
            # 子图4: 运镜类型统计
            plt.subplot(2, 2, 4)
            motion_types = []
            for result_item in batch_results['results']:
                result = result_item['result']
                if result.get('success', True):
                    motion_type = result.get('fusion_result', {}).get('motion_type', 'unknown')
                    motion_types.append(motion_type)
            
            if motion_types:
                type_counts = {}
                for motion_type in motion_types:
                    type_counts[motion_type] = type_counts.get(motion_type, 0) + 1
                
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                
                plt.bar(types, counts, color='lightgreen', alpha=0.7)
                plt.xlabel('Motion Type')
                plt.ylabel('Count')
                plt.title('Motion Type Distribution')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            stats_path = os.path.join(output_dir, 'batch_statistics.png')
            plt.savefig(stats_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Batch statistics saved to: {stats_path}")
            
        except Exception as e:
            print(f"Failed to generate batch statistics: {e}")


# 使用示例和命令行接口
def main():
    """
    命令行主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Bullet Time Camera Motion Evaluation Framework')
    parser.add_argument('--video', type=str, help='Single video file to evaluate')
    parser.add_argument('--batch', type=str, help='Text file with list of video paths for batch evaluation')
    parser.add_argument('--output', type=str, default='./bullet_time_evaluation_output', 
                       help='Output directory')
    parser.add_argument('--config', type=str, help='Configuration JSON file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help='Computing device')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # 创建评估框架
    framework = BulletTimeEvaluationFramework(device=args.device, config=config)
    
    if args.video:
        # 单个视频评估
        print(f"Evaluating single video: {args.video}")
        result = framework.evaluate_video(args.video, args.output)
        
        if result.get('fusion_result', {}).get('is_bullet_time', False):
            print("✓ Bullet time motion detected!")
        else:
            print("✗ No bullet time motion detected.")
            
    elif args.batch:
        # 批量评估
        print(f"Loading video list from: {args.batch}")
        with open(args.batch, 'r') as f:
            video_list = [line.strip() for line in f if line.strip()]
        
        print(f"Evaluating {len(video_list)} videos...")
        batch_results = framework.evaluate_batch(video_list, args.output)
        
        print(f"\nBatch evaluation completed:")
        print(f"Total videos: {batch_results['total_videos']}")
        print(f"Successful evaluations: {batch_results['successful_evaluations']}")
        print(f"Bullet time detected: {batch_results['bullet_time_detected']}")
        
    else:
        print("Please specify either --video or --batch argument")
        parser.print_help()


if __name__ == "__main__":
    main()
