"""
增强的运镜检测模块
集成了多种姿态估计方法来处理子弹时间等复杂运镜场景
"""

import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from math import ceil
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple, Optional, Union
import sys
import os.path as osp

# 导入原有模块
from .camera_motion import (
    CameraPredict as OriginalCameraPredict,
    load_foreground_masks,
    calculate_motion_magnitude,
    classify_motion_magnitude
)
from .pose_estimation_bullet_time import HumanPoseEstimator, BulletTimeCameraMotionEvaluator
from .utils import load_dimension_info, split_video_into_scenes


class EnhancedCameraPredict(OriginalCameraPredict):
    """
    增强的相机运镜预测器
    继承原有功能，增加了人体姿态估计和子弹时间检测
    """
    
    def __init__(self, device, submodules_list, enable_depth=True, enable_pose=True):
        super().__init__(device, submodules_list, enable_depth)
        
        self.enable_pose = enable_pose
        if self.enable_pose:
            try:
                self.pose_estimator = HumanPoseEstimator(device)
                self.bullet_time_evaluator = BulletTimeCameraMotionEvaluator(device)
                print("Pose estimation models loaded successfully")
            except Exception as e:
                print(f"Failed to load pose estimation models: {e}")
                self.pose_estimator = None
                self.bullet_time_evaluator = None
        else:
            self.pose_estimator = None
            self.bullet_time_evaluator = None
    
    def detect_bullet_time_motion(self, video, fps, end_frame, masks=None):
        """
        检测子弹时间运镜
        
        Args:
            video: 视频张量 [B, T, C, H, W]
            fps: 帧率
            end_frame: 结束帧
            masks: 前景mask（可选）
        
        Returns:
            bullet_time_info: 子弹时间检测结果
        """
        if self.pose_estimator is None:
            return None
            
        try:
            b, t, c, h, w = video.shape
            
            # 转换视频格式用于姿态估计
            pose_sequence = []
            
            for frame_idx in range(min(t, end_frame if end_frame != -1 else t)):
                frame = video[0, frame_idx].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # MediaPipe姿态检测
                rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self.pose_estimator.pose.process(rgb_frame)
                
                frame_info = {
                    'frame': frame_idx,
                    'yaw': None,
                    'pitch': None,
                    'roll': None,
                    'confidence': 0.0
                }
                
                if results.pose_landmarks:
                    # 提取关键点坐标
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([landmark.x, landmark.y, landmark.visibility])
                    landmarks = np.array(landmarks)
                    
                    # 估计yaw角度
                    yaw_angle = self.pose_estimator.estimate_yaw_from_keypoints(landmarks)
                    if yaw_angle is not None:
                        frame_info['yaw'] = yaw_angle
                        frame_info['confidence'] = 0.8
                    
                    # 如果有分割结果，也使用轮廓方法
                    if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                        silhouette_yaw = self.pose_estimator.estimate_yaw_from_silhouette(
                            results.segmentation_mask)
                        if silhouette_yaw is not None and frame_info['yaw'] is None:
                            frame_info['yaw'] = silhouette_yaw
                            frame_info['confidence'] = 0.6
                
                pose_sequence.append(frame_info)
            
            # 分析是否为子弹时间运镜
            return self._analyze_bullet_time_sequence(pose_sequence)
            
        except Exception as e:
            print(f"Error in bullet time detection: {e}")
            return None
    
    def _analyze_bullet_time_sequence(self, pose_sequence: List[Dict]) -> Optional[Dict]:
        """
        分析姿态序列，判断是否为子弹时间运镜
        """
        # 提取有效的yaw角度
        valid_yaws = [frame['yaw'] for frame in pose_sequence if frame['yaw'] is not None]
        
        if len(valid_yaws) < 10:  # 需要足够的数据点
            return None
        
        # 平滑角度序列
        smoothed_yaws = self._smooth_angles(valid_yaws)
        
        # 计算角度变化
        angle_changes = np.diff(smoothed_yaws)
        
        # 处理角度跳跃
        for i in range(len(angle_changes)):
            if angle_changes[i] > 180:
                angle_changes[i] -= 360
            elif angle_changes[i] < -180:
                angle_changes[i] += 360
        
        # 计算总旋转角度
        total_rotation = np.sum(angle_changes)
        
        # 判断旋转特征
        is_continuous_rotation = len(angle_changes) > 5 and np.std(angle_changes) < 20
        is_significant_rotation = abs(total_rotation) > 90  # 至少90度旋转
        
        # 计算旋转方向一致性
        rotation_direction = np.sign(angle_changes)
        direction_consistency = np.sum(rotation_direction == np.sign(total_rotation)) / len(rotation_direction)
        
        # 判断是否为子弹时间
        is_bullet_time = (is_continuous_rotation and 
                         is_significant_rotation and 
                         direction_consistency > 0.7)
        
        if is_bullet_time:
            # 判断旋转方向
            if total_rotation > 0:
                motion_type = "bullet_time_clockwise"
            else:
                motion_type = "bullet_time_counterclockwise"
            
            return {
                'motion_type': motion_type,
                'total_rotation': total_rotation,
                'direction_consistency': direction_consistency,
                'smoothness': 1.0 / (1.0 + np.std(angle_changes)),
                'confidence': min(0.9, direction_consistency + 0.2)
            }
        
        return None
    
    def _smooth_angles(self, angles: List[float], window_size: int = 5) -> np.ndarray:
        """
        平滑角度序列
        """
        if len(angles) < window_size:
            return np.array(angles)
        
        smoothed = []
        for i in range(len(angles)):
            start = max(0, i - window_size // 2)
            end = min(len(angles), i + window_size // 2 + 1)
            smoothed.append(np.mean(angles[start:end]))
        
        return np.array(smoothed)
    
    def detect_orbiting_motion(self, pred_track, pred_visibility, video_shape, masks=None):
        """
        检测环绕运镜（orbits）
        增强版本，结合特征点和姿态信息
        """
        # 原有的360度检测
        tracks = [pred_track[i].reshape(self.grid_size, self.grid_size, 2) 
                 for i in range(0, len(pred_track), 20)]
        
        orbit_results = self.get_edge_direction_360(tracks)
        
        # 统计orbits检测结果
        orbit_count = sum(1 for result in orbit_results if result == "orbits")
        orbit_ratio = orbit_count / max(1, len(orbit_results))
        
        if orbit_ratio > 0.3:  # 如果30%以上帧检测到orbits
            return "orbits"
        
        return None
    
    def enhanced_predict(self, video, fps, end_frame, masks=None):
        """
        增强的运镜预测方法
        结合原有方法和新的姿态估计
        """
        # 原有的预测结果
        original_results, magnitude_info = super().predict(video, fps, end_frame, masks)
        
        enhanced_results = original_results.copy()
        additional_info = {
            'magnitude_info': magnitude_info,
            'bullet_time_info': None,
            'pose_analysis': None
        }
        
        # 子弹时间检测
        if self.pose_estimator is not None:
            bullet_time_info = self.detect_bullet_time_motion(video, fps, end_frame, masks)
            if bullet_time_info:
                motion_type = bullet_time_info['motion_type']
                if motion_type not in enhanced_results:
                    enhanced_results.append(motion_type)
                additional_info['bullet_time_info'] = bullet_time_info
        
        # 增强的orbits检测
        pred_track, pred_visibility = self.infer(video, fps, end_frame)
        orbit_result = self.detect_orbiting_motion(
            pred_track, pred_visibility, (self.height, self.width), masks)
        
        if orbit_result and orbit_result not in enhanced_results:
            enhanced_results.append(orbit_result)
        
        # 运镜复杂度分析
        complexity_score = self._calculate_motion_complexity(enhanced_results, magnitude_info)
        additional_info['complexity_score'] = complexity_score
        
        return enhanced_results, additional_info
    
    def _calculate_motion_complexity(self, motion_types: List[str], magnitude_info: Dict) -> float:
        """
        计算运镜复杂度评分
        """
        # 基础分数基于运镜类型数量
        base_score = len(motion_types) * 0.2
        
        # 特殊运镜加分
        special_motions = ['bullet_time_clockwise', 'bullet_time_counterclockwise', 'orbits']
        special_count = sum(1 for motion in motion_types if motion in special_motions)
        special_score = special_count * 0.3
        
        # 运镜幅度加分
        magnitude_level = magnitude_info.get('magnitude_level', 'none')
        magnitude_scores = {
            'none': 0.0,
            'subtle': 0.1,
            'moderate': 0.2,
            'obvious': 0.3,
            'very_obvious': 0.4,
            'extremely_obvious': 0.5
        }
        magnitude_score = magnitude_scores.get(magnitude_level, 0.0)
        
        total_score = base_score + special_score + magnitude_score
        return min(1.0, total_score)


def enhanced_camera_motion(prompt_dict_ls, camera, mask_dir=None):
    """
    增强的运镜评测函数
    """
    sim = []
    video_results = []

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        
        for video_path in video_paths:
            end_frame = -1
            scene_list = split_video_into_scenes(video_path, 5.0)
            if len(scene_list) != 0:
                end_frame = int(scene_list[0][1].get_frames())
            
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader))) 
            frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
            video = video.permute(0, 3, 1, 2)[None].float().cuda()  # B T C H W
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # 加载前景mask
            masks = None
            if mask_dir:
                masks = load_foreground_masks(mask_dir, video_path)
                if masks is not None and end_frame != -1:
                    masks = masks[:end_frame]
            
            # 使用增强预测
            predict_results, additional_info = camera.enhanced_predict(video, fps, end_frame, masks)
            
            # 计算评分
            video_score = 1.0 if label in predict_results else 0.0
            
            # 特殊处理：如果期望是子弹时间相关的运镜
            if 'bullet_time' in label.lower() or 'orbit' in label.lower():
                bullet_time_info = additional_info.get('bullet_time_info')
                if bullet_time_info and bullet_time_info['confidence'] > 0.7:
                    video_score = max(video_score, 0.8)  # 给予较高分数
            
            # 扩展输出结构
            video_results.append({
                'video_path': video_path,
                'video_results': video_score,
                'motion_types': predict_results,
                'magnitude_info': additional_info['magnitude_info'],
                'bullet_time_info': additional_info.get('bullet_time_info'),
                'complexity_score': additional_info.get('complexity_score', 0.0),
                'expected_label': label
            })
            sim.append(video_score)
    
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_enhanced_camera_motion(json_dir, device, submodules_dict, **kwargs):
    """
    增强的运镜计算函数
    """
    # 获取配置参数
    mask_dir = kwargs.get('mask_dir', None)
    enable_depth = kwargs.get('enable_depth', True)
    enable_pose = kwargs.get('enable_pose', True)

    camera = EnhancedCameraPredict(device, submodules_dict, 
                                 enable_depth=enable_depth, 
                                 enable_pose=enable_pose)
    
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')

    all_results, video_results = enhanced_camera_motion(prompt_dict_ls, camera, mask_dir)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    
    return all_results, video_results


# 运镜分析工具函数
def analyze_camera_motion_sequence(video_results: List[Dict]) -> Dict:
    """
    分析运镜序列的统计信息
    """
    motion_types_count = {}
    complexity_scores = []
    magnitude_levels = []
    
    for result in video_results:
        # 统计运镜类型
        for motion_type in result['motion_types']:
            motion_types_count[motion_type] = motion_types_count.get(motion_type, 0) + 1
        
        # 收集复杂度分数
        complexity_scores.append(result.get('complexity_score', 0.0))
        
        # 收集运镜幅度
        magnitude_info = result.get('magnitude_info', {})
        magnitude_levels.append(magnitude_info.get('magnitude_level', 'none'))
    
    return {
        'motion_types_distribution': motion_types_count,
        'average_complexity': np.mean(complexity_scores),
        'complexity_std': np.std(complexity_scores),
        'magnitude_distribution': {level: magnitude_levels.count(level) 
                                 for level in set(magnitude_levels)},
        'total_videos': len(video_results),
        'bullet_time_videos': sum(1 for result in video_results 
                                if result.get('bullet_time_info') is not None)
    }


if __name__ == "__main__":
    # 测试增强运镜检测
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟submodules配置
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 创建增强预测器
    camera = EnhancedCameraPredict(device, submodules_dict)
    
    print("Enhanced camera motion detector initialized successfully!")
