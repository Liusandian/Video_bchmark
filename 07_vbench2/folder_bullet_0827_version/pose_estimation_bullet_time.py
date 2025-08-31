"""
子弹时间运镜中的人体姿态估计模块
当人脸不可见时，通过多种方法计算人体的yaw旋转角度
"""

import cv2
import numpy as np
import torch
import json
from typing import List, Dict, Tuple, Optional, Union
import mediapipe as mp
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import math


class HumanPoseEstimator:
    """
    人体姿态估计器，专门用于子弹时间运镜中的yaw角度计算
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 关键身体部位的关键点索引（MediaPipe格式）
        self.body_landmarks = {
            'shoulders': [11, 12],  # 左右肩膀
            'hips': [23, 24],       # 左右髋部
            'elbows': [13, 14],     # 左右肘部
            'knees': [25, 26],      # 左右膝盖
            'ankles': [27, 28],     # 左右脚踝
            'wrists': [15, 16],     # 左右手腕
        }
        
    def estimate_yaw_from_keypoints(self, keypoints: np.ndarray) -> Optional[float]:
        """
        基于身体关键点估计yaw角度
        
        Args:
            keypoints: MediaPipe姿态关键点 [33, 3] (x, y, visibility)
        
        Returns:
            yaw_angle: yaw角度（度数），None表示估计失败
        """
        if keypoints is None or len(keypoints) < 33:
            return None
            
        # 提取关键身体部位
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        
        # 检查关键点可见性
        if (left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5 or 
            left_hip[2] < 0.5 or right_hip[2] < 0.5):
            return self._estimate_yaw_fallback(keypoints)
        
        # 计算肩膀和髋部的中心线向量
        shoulder_vector = right_shoulder[:2] - left_shoulder[:2]
        hip_vector = right_hip[:2] - left_hip[:2]
        
        # 计算身体朝向角度
        body_angle = np.mean([
            np.arctan2(shoulder_vector[1], shoulder_vector[0]),
            np.arctan2(hip_vector[1], hip_vector[0])
        ])
        
        # 转换为yaw角度（相对于相机正面方向）
        yaw_angle = np.degrees(body_angle)
        
        # 规范化到[-180, 180]范围
        yaw_angle = self._normalize_angle(yaw_angle)
        
        return yaw_angle
    
    def _estimate_yaw_fallback(self, keypoints: np.ndarray) -> Optional[float]:
        """
        当主要关键点不可见时的备用估计方法
        """
        # 尝试使用其他可见的身体部位
        fallback_pairs = [
            ([13, 14], 'elbows'),      # 肘部
            ([25, 26], 'knees'),       # 膝盖
            ([15, 16], 'wrists'),      # 手腕
        ]
        
        for indices, name in fallback_pairs:
            left_point = keypoints[indices[0]]
            right_point = keypoints[indices[1]]
            
            if left_point[2] > 0.5 and right_point[2] > 0.5:
                vector = right_point[:2] - left_point[:2]
                angle = np.arctan2(vector[1], vector[0])
                return self._normalize_angle(np.degrees(angle))
        
        return None
    
    def estimate_yaw_from_silhouette(self, silhouette_mask: np.ndarray) -> Optional[float]:
        """
        基于人体轮廓估计yaw角度
        
        Args:
            silhouette_mask: 人体轮廓mask [H, W]
        
        Returns:
            yaw_angle: 估计的yaw角度
        """
        if silhouette_mask is None or np.sum(silhouette_mask) == 0:
            return None
            
        # 计算轮廓的主要方向
        contours, _ = cv2.findContours(silhouette_mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # 根据矩形的长宽比调整角度
        width, height = rect[1]
        if width > height:
            angle += 90
            
        return self._normalize_angle(angle)
    
    def estimate_yaw_from_optical_flow(self, flow_vectors: np.ndarray, 
                                     body_mask: np.ndarray) -> Optional[float]:
        """
        基于光流信息估计yaw角度
        
        Args:
            flow_vectors: 光流向量 [H, W, 2]
            body_mask: 人体区域mask [H, W]
        
        Returns:
            yaw_angle: 估计的yaw角度
        """
        if flow_vectors is None or body_mask is None:
            return None
            
        # 提取人体区域的光流
        body_flow = flow_vectors[body_mask > 0]
        
        if len(body_flow) < 10:  # 光流点太少
            return None
            
        # 计算主要运动方向
        mean_flow = np.mean(body_flow, axis=0)
        flow_angle = np.arctan2(mean_flow[1], mean_flow[0])
        
        # 根据相机运动推断人体旋转
        # 如果相机绕人体旋转，人体相对运动方向与yaw角度有关
        yaw_angle = self._normalize_angle(np.degrees(flow_angle) + 90)
        
        return yaw_angle
    
    def estimate_yaw_from_depth_profile(self, depth_map: np.ndarray, 
                                      body_mask: np.ndarray) -> Optional[float]:
        """
        基于深度轮廓估计yaw角度
        
        Args:
            depth_map: 深度图 [H, W]
            body_mask: 人体区域mask [H, W]
        
        Returns:
            yaw_angle: 估计的yaw角度
        """
        if depth_map is None or body_mask is None:
            return None
            
        # 提取人体区域的深度信息
        body_depth = depth_map[body_mask > 0]
        
        if len(body_depth) < 100:
            return None
            
        # 分析深度分布来推断朝向
        # 创建深度轮廓
        y_coords, x_coords = np.where(body_mask > 0)
        depths = depth_map[y_coords, x_coords]
        
        # 按深度排序，找到前景和背景点
        depth_sorted_indices = np.argsort(depths)
        foreground_points = np.column_stack([x_coords[depth_sorted_indices[:len(depths)//3]], 
                                           y_coords[depth_sorted_indices[:len(depths)//3]]])
        
        # 计算前景点的质心和主方向
        if len(foreground_points) > 5:
            centroid = np.mean(foreground_points, axis=0)
            
            # 使用PCA找到主方向
            centered_points = foreground_points - centroid
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # 主方向对应最大特征值的特征向量
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            angle = np.arctan2(main_direction[1], main_direction[0])
            
            return self._normalize_angle(np.degrees(angle))
        
        return None
    
    def estimate_3d_pose_rotation(self, keypoints_3d: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
        """
        基于3D关键点估计完整的旋转角度（pitch, yaw, roll）
        
        Args:
            keypoints_3d: 3D关键点坐标 [N, 3]
        
        Returns:
            rotation_angles: {'pitch': float, 'yaw': float, 'roll': float}
        """
        if keypoints_3d is None or len(keypoints_3d) < 4:
            return None
            
        # 定义身体坐标系的关键点
        try:
            # 肩膀中心
            shoulder_center = (keypoints_3d[11] + keypoints_3d[12]) / 2
            # 髋部中心  
            hip_center = (keypoints_3d[23] + keypoints_3d[24]) / 2
            # 头部（鼻子）
            head_point = keypoints_3d[0]
            
            # 计算身体的三个轴
            # Y轴：从髋部到肩膀（身体纵轴）
            y_axis = shoulder_center - hip_center
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            # X轴：从左肩到右肩
            x_axis = keypoints_3d[12] - keypoints_3d[11]  # 右肩 - 左肩
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Z轴：通过叉积计算
            z_axis = np.cross(x_axis, y_axis)
            z_axis = z_axis / np.linalg.norm(z_axis)
            
            # 构建旋转矩阵
            rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
            
            # 从旋转矩阵提取欧拉角
            pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)
            
            return {
                'pitch': pitch,
                'yaw': yaw, 
                'roll': roll
            }
            
        except Exception as e:
            print(f"3D pose estimation failed: {e}")
            return None
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """
        从旋转矩阵转换为欧拉角（ZYX顺序）
        """
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(R[2,1], R[2,2])  # roll
            y = math.atan2(-R[2,0], sy)      # pitch  
            z = math.atan2(R[1,0], R[0,0])   # yaw
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
            
        return np.degrees(x), np.degrees(y), np.degrees(z)
    
    def _normalize_angle(self, angle: float) -> float:
        """
        将角度规范化到[-180, 180]范围
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def process_video_for_bullet_time(self, video_path: str, 
                                    output_path: Optional[str] = None) -> List[Dict]:
        """
        处理子弹时间视频，提取每帧的yaw角度
        
        Args:
            video_path: 输入视频路径
            output_path: 输出JSON文件路径
        
        Returns:
            frame_data: 每帧的姿态信息列表
        """
        cap = cv2.VideoCapture(video_path)
        frame_data = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # MediaPipe姿态检测
            results = self.pose.process(rgb_frame)
            
            frame_info = {
                'frame': frame_idx,
                'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                'yaw': None,
                'pitch': None,
                'roll': None,
                'confidence': 0.0,
                'method_used': None
            }
            
            if results.pose_landmarks:
                # 提取关键点坐标
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.visibility])
                landmarks = np.array(landmarks)
                
                # 方法1: 基于2D关键点估计yaw
                yaw_2d = self.estimate_yaw_from_keypoints(landmarks)
                if yaw_2d is not None:
                    frame_info['yaw'] = yaw_2d
                    frame_info['method_used'] = 'keypoints_2d'
                    frame_info['confidence'] = 0.8
                
                # 方法2: 如果有3D信息，使用3D估计
                if hasattr(results, 'pose_world_landmarks') and results.pose_world_landmarks:
                    landmarks_3d = []
                    for landmark in results.pose_world_landmarks.landmark:
                        landmarks_3d.append([landmark.x, landmark.y, landmark.z])
                    landmarks_3d = np.array(landmarks_3d)
                    
                    rotation_3d = self.estimate_3d_pose_rotation(landmarks_3d)
                    if rotation_3d:
                        frame_info.update(rotation_3d)
                        frame_info['method_used'] = 'keypoints_3d'
                        frame_info['confidence'] = 0.9
                
                # 方法3: 基于分割mask估计（如果可用）
                if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
                    yaw_silhouette = self.estimate_yaw_from_silhouette(results.segmentation_mask)
                    if yaw_silhouette is not None and frame_info['yaw'] is None:
                        frame_info['yaw'] = yaw_silhouette
                        frame_info['method_used'] = 'silhouette'
                        frame_info['confidence'] = 0.6
            
            frame_data.append(frame_info)
            frame_idx += 1
            
        cap.release()
        
        # 后处理：平滑角度序列
        frame_data = self._smooth_angle_sequence(frame_data)
        
        # 保存结果
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(frame_data, f, indent=2)
                
        return frame_data
    
    def _smooth_angle_sequence(self, frame_data: List[Dict]) -> List[Dict]:
        """
        平滑角度序列，减少噪声
        """
        # 提取yaw角度序列
        yaw_angles = [frame.get('yaw') for frame in frame_data]
        
        # 插值缺失值
        valid_indices = [i for i, angle in enumerate(yaw_angles) if angle is not None]
        if len(valid_indices) < 2:
            return frame_data
            
        valid_angles = [yaw_angles[i] for i in valid_indices]
        
        # 使用线性插值填补缺失值
        for i in range(len(yaw_angles)):
            if yaw_angles[i] is None:
                # 找到最近的有效值进行插值
                left_idx = max([idx for idx in valid_indices if idx < i], default=None)
                right_idx = min([idx for idx in valid_indices if idx > i], default=None)
                
                if left_idx is not None and right_idx is not None:
                    # 线性插值
                    t = (i - left_idx) / (right_idx - left_idx)
                    angle = yaw_angles[left_idx] * (1 - t) + yaw_angles[right_idx] * t
                    yaw_angles[i] = angle
                    frame_data[i]['yaw'] = angle
                    frame_data[i]['method_used'] = 'interpolated'
                    frame_data[i]['confidence'] = 0.4
        
        # 应用移动平均滤波
        window_size = 5
        for i in range(len(yaw_angles)):
            if yaw_angles[i] is not None:
                start = max(0, i - window_size // 2)
                end = min(len(yaw_angles), i + window_size // 2 + 1)
                
                valid_neighbors = [yaw_angles[j] for j in range(start, end) if yaw_angles[j] is not None]
                if len(valid_neighbors) >= 3:
                    smoothed_angle = np.mean(valid_neighbors)
                    frame_data[i]['yaw'] = smoothed_angle
        
        return frame_data


class BulletTimeCameraMotionEvaluator:
    """
    子弹时间运镜评测器
    """
    
    def __init__(self, device='cuda'):
        self.pose_estimator = HumanPoseEstimator(device)
        
    def evaluate_bullet_time_rotation(self, video_path: str) -> Dict:
        """
        评测子弹时间运镜的旋转角度和平滑度
        
        Args:
            video_path: 视频路径
            
        Returns:
            evaluation_result: 评测结果
        """
        # 提取姿态序列
        pose_sequence = self.pose_estimator.process_video_for_bullet_time(video_path)
        
        # 计算旋转指标
        yaw_angles = [frame['yaw'] for frame in pose_sequence if frame['yaw'] is not None]
        
        if len(yaw_angles) < 10:
            return {
                'success': False,
                'error': 'Insufficient pose data for evaluation'
            }
        
        # 计算总旋转角度
        total_rotation = self._calculate_total_rotation(yaw_angles)
        
        # 计算角速度变化（平滑度）
        angular_velocity = np.diff(yaw_angles)
        smoothness_score = self._calculate_smoothness_score(angular_velocity)
        
        # 计算旋转方向一致性
        direction_consistency = self._calculate_direction_consistency(angular_velocity)
        
        # 检测是否完成完整的360度旋转
        full_rotation = abs(total_rotation) >= 350  # 允许10度误差
        
        return {
            'success': True,
            'total_rotation_degrees': total_rotation,
            'smoothness_score': smoothness_score,
            'direction_consistency': direction_consistency,
            'is_full_rotation': full_rotation,
            'frame_count': len(pose_sequence),
            'valid_pose_count': len(yaw_angles),
            'pose_detection_rate': len(yaw_angles) / len(pose_sequence),
            'angular_velocity_stats': {
                'mean': np.mean(angular_velocity),
                'std': np.std(angular_velocity),
                'max': np.max(angular_velocity),
                'min': np.min(angular_velocity)
            }
        }
    
    def _calculate_total_rotation(self, yaw_angles: List[float]) -> float:
        """计算总旋转角度，处理角度跳跃"""
        total = 0.0
        for i in range(1, len(yaw_angles)):
            diff = yaw_angles[i] - yaw_angles[i-1]
            
            # 处理角度跳跃（-180到180的边界）
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
                
            total += diff
            
        return total
    
    def _calculate_smoothness_score(self, angular_velocity: np.ndarray) -> float:
        """计算运镜平滑度评分"""
        if len(angular_velocity) == 0:
            return 0.0
            
        # 计算角速度的标准差（越小越平滑）
        velocity_std = np.std(angular_velocity)
        
        # 转换为0-1评分（标准差越小，评分越高）
        smoothness = np.exp(-velocity_std / 10.0)  # 经验参数
        
        return min(1.0, max(0.0, smoothness))
    
    def _calculate_direction_consistency(self, angular_velocity: np.ndarray) -> float:
        """计算旋转方向一致性"""
        if len(angular_velocity) == 0:
            return 0.0
            
        # 计算正向和负向运动的比例
        positive_count = np.sum(angular_velocity > 0)
        negative_count = np.sum(angular_velocity < 0)
        total_count = len(angular_velocity)
        
        if total_count == 0:
            return 0.0
            
        # 计算主导方向的比例
        dominant_ratio = max(positive_count, negative_count) / total_count
        
        return dominant_ratio


# 使用示例
if __name__ == "__main__":
    # 初始化评测器
    evaluator = BulletTimeCameraMotionEvaluator()
    
    # 评测子弹时间视频
    video_path = "path/to/bullet_time_video.mp4"
    result = evaluator.evaluate_bullet_time_rotation(video_path)
    
    print("子弹时间运镜评测结果:")
    print(f"总旋转角度: {result['total_rotation_degrees']:.1f}°")
    print(f"平滑度评分: {result['smoothness_score']:.3f}")
    print(f"方向一致性: {result['direction_consistency']:.3f}")
    print(f"是否完整旋转: {result['is_full_rotation']}")
    print(f"姿态检测率: {result['pose_detection_rate']:.3f}")
