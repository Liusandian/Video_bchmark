"""
基于深度信息的3D姿态估计模块
结合深度图和RGB信息来估计人体3D姿态和旋转角度
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
import math
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import sys
import os.path as osp

# 导入深度估计模型
sys.path.append(osp.join(osp.dirname(__file__), 'third_party', 'Depth-Anything-V2-main', 'Depth-Anything-V2-main'))
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except:
    print("Depth-Anything-V2 not available, 3D pose estimation will be limited")
    DepthAnythingV2 = None


class Depth3DPoseEstimator:
    """
    基于深度信息的3D姿态估计器
    """
    
    def __init__(self, device='cuda', depth_model_type='vitl'):
        self.device = device
        self.depth_model = None
        
        # 加载深度估计模型
        if DepthAnythingV2 is not None:
            try:
                self.depth_model = self._load_depth_model(depth_model_type)
                print(f"Depth model {depth_model_type} loaded successfully")
            except Exception as e:
                print(f"Failed to load depth model: {e}")
        
        # 人体关键点的3D模板（标准姿势）
        self.template_3d_keypoints = self._create_human_template()
        
        # 相机内参（假设值，实际使用时应该标定）
        self.camera_intrinsics = np.array([
            [800, 0, 320],    # fx, 0, cx
            [0, 800, 240],    # 0, fy, cy  
            [0, 0, 1]         # 0, 0, 1
        ], dtype=np.float32)
    
    def _load_depth_model(self, encoder='vitl'):
        """
        加载深度估计模型
        """
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        depth_model = DepthAnythingV2(**model_configs[encoder])

        # 尝试加载模型权重
        checkpoint_path = osp.join(
            osp.dirname(__file__), 'third_party', 'Depth-Anything-V2-main', 
            'Depth-Anything-V2-main', 'checkpoints', f'depth_anything_v2_{encoder}.pth'
        )

        if osp.exists(checkpoint_path):
            depth_model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        else:
            print(f"Warning: Depth model checkpoint not found at {checkpoint_path}")

        depth_model = depth_model.to(self.device).eval()
        return depth_model
    
    def _create_human_template(self):
        """
        创建标准人体3D关键点模板（MediaPipe格式）
        坐标系：Y轴向上，Z轴向前（朝向相机）
        """
        # 标准人体关键点的相对3D坐标（归一化）
        template = np.array([
            # 头部关键点
            [0.0, 1.0, 0.0],      # 0: 鼻子
            [0.05, 0.98, 0.02],   # 1: 左眼内角
            [-0.05, 0.98, 0.02],  # 2: 右眼内角
            [0.08, 0.98, 0.0],    # 3: 左眼
            [-0.08, 0.98, 0.0],   # 4: 右眼
            [0.12, 0.95, 0.0],    # 5: 左耳
            [-0.12, 0.95, 0.0],   # 6: 右耳
            [0.15, 0.92, 0.0],    # 7: 左嘴角
            [-0.15, 0.92, 0.0],   # 8: 右嘴角
            [0.0, 0.92, 0.05],    # 9: 上唇
            [0.0, 0.88, 0.05],    # 10: 下唇
            
            # 上身关键点
            [0.15, 0.75, 0.0],    # 11: 左肩
            [-0.15, 0.75, 0.0],   # 12: 右肩
            [0.25, 0.55, 0.0],    # 13: 左肘
            [-0.25, 0.55, 0.0],   # 14: 右肘
            [0.3, 0.35, 0.0],     # 15: 左手腕
            [-0.3, 0.35, 0.0],    # 16: 右手腕
            [0.35, 0.3, 0.0],     # 17: 左小指
            [-0.35, 0.3, 0.0],    # 18: 右小指
            [0.38, 0.32, 0.0],    # 19: 左食指
            [-0.38, 0.32, 0.0],   # 20: 右食指
            [0.35, 0.35, 0.0],    # 21: 左拇指
            [-0.35, 0.35, 0.0],   # 22: 右拇指
            
            # 下身关键点
            [0.1, 0.0, 0.0],      # 23: 左髋
            [-0.1, 0.0, 0.0],     # 24: 右髋
            [0.12, -0.4, 0.0],    # 25: 左膝
            [-0.12, -0.4, 0.0],   # 26: 右膝
            [0.1, -0.8, 0.0],     # 27: 左脚踝
            [-0.1, -0.8, 0.0],    # 28: 右脚踝
            [0.15, -0.85, 0.0],   # 29: 左脚跟
            [-0.15, -0.85, 0.0],  # 30: 右脚跟
            [0.05, -0.9, 0.0],    # 31: 左脚趾
            [-0.05, -0.9, 0.0],   # 32: 右脚趾
        ], dtype=np.float32)
        
        return template
    
    def compute_depth_map(self, image: np.ndarray, input_size=518) -> Optional[np.ndarray]:
        """
        计算图像的深度图
        """
        if self.depth_model is None:
            return None
            
        try:
            with torch.no_grad():
                depth = self.depth_model.infer_image(image, input_size)
            return depth
        except Exception as e:
            print(f"Depth computation failed: {e}")
            return None
    
    def estimate_3d_pose_from_depth(self, rgb_image: np.ndarray,
                                   keypoints_2d: np.ndarray,
                                   depth_map: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        从RGB图像和2D关键点估计3D姿态
        
        Args:
            rgb_image: RGB图像 [H, W, 3]
            keypoints_2d: 2D关键点 [N, 3] (x, y, visibility)
            depth_map: 深度图 [H, W]（可选，如果没有则自动计算）
        
        Returns:
            pose_3d_info: 3D姿态信息
        """
        if depth_map is None:
            depth_map = self.compute_depth_map(rgb_image)
            if depth_map is None:
                return None
        
        # 确保深度图和图像尺寸匹配
        if depth_map.shape[:2] != rgb_image.shape[:2]:
            depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # 提取有效的2D关键点
        valid_keypoints = keypoints_2d[keypoints_2d[:, 2] > 0.5]  # 可见性阈值
        
        if len(valid_keypoints) < 4:
            return None
        
        # 从2D关键点和深度图重建3D关键点
        keypoints_3d = self._reconstruct_3d_keypoints(valid_keypoints, depth_map)
        
        if keypoints_3d is None or len(keypoints_3d) < 4:
            return None
        
        # 估计人体姿态
        pose_estimation = self._estimate_pose_from_3d_keypoints(keypoints_3d)
        
        return pose_estimation
    
    def _reconstruct_3d_keypoints(self, keypoints_2d: np.ndarray, 
                                 depth_map: np.ndarray) -> Optional[np.ndarray]:
        """
        从2D关键点和深度图重建3D关键点
        """
        keypoints_3d = []
        
        h, w = depth_map.shape
        fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
        cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
        
        for kp in keypoints_2d:
            x, y, visibility = kp
            
            # 将归一化坐标转换为像素坐标
            if x <= 1.0 and y <= 1.0:  # 归一化坐标
                px = int(x * w)
                py = int(y * h)
            else:  # 像素坐标
                px, py = int(x), int(y)
            
            # 边界检查
            if 0 <= px < w and 0 <= py < h:
                # 获取深度值
                depth = depth_map[py, px]
                
                # 反投影到3D空间
                x_3d = (px - cx) * depth / fx
                y_3d = (py - cy) * depth / fy
                z_3d = depth
                
                keypoints_3d.append([x_3d, y_3d, z_3d])
        
        return np.array(keypoints_3d) if keypoints_3d else None
    
    def _estimate_pose_from_3d_keypoints(self, keypoints_3d: np.ndarray) -> Dict:
        """
        从3D关键点估计姿态
        """
        try:
            # 建立身体坐标系
            body_coordinate_system = self._establish_body_coordinate_system(keypoints_3d)
            
            if body_coordinate_system is None:
                return {'success': False, 'error': 'Failed to establish body coordinate system'}
            
            # 计算旋转矩阵
            rotation_matrix = body_coordinate_system['rotation_matrix']
            
            # 转换为欧拉角
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            # 提取yaw, pitch, roll
            pitch, yaw, roll = euler_angles
            
            # 计算置信度
            confidence = self._calculate_pose_confidence(keypoints_3d, body_coordinate_system)
            
            return {
                'success': True,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'rotation_matrix': rotation_matrix,
                'body_center': body_coordinate_system['center'],
                'confidence': confidence,
                'keypoints_3d': keypoints_3d
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _establish_body_coordinate_system(self, keypoints_3d: np.ndarray) -> Optional[Dict]:
        """
        建立身体坐标系
        """
        # 需要至少4个关键点来建立坐标系
        if len(keypoints_3d) < 4:
            return None
        
        # 假设关键点按MediaPipe格式排列
        # 尝试找到肩膀、髋部等关键身体部位
        
        # 简化版本：使用前几个点来估计方向
        # 实际应用中需要根据具体的关键点索引来选择
        
        # 计算身体中心
        center = np.mean(keypoints_3d, axis=0)
        
        # 估计主要方向
        # 使用PCA找到主要方向
        centered_points = keypoints_3d - center
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 排序特征向量
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, idx]
        
        # 构建旋转矩阵
        # 主要方向作为身体纵轴（Y轴）
        y_axis = sorted_eigenvectors[:, 0]
        
        # 确保Y轴向上
        if y_axis[1] < 0:
            y_axis = -y_axis
        
        # 选择一个垂直方向作为X轴
        if len(keypoints_3d) >= 2:
            # 使用两个点的连线作为参考
            ref_vector = keypoints_3d[1] - keypoints_3d[0]
            ref_vector = ref_vector / np.linalg.norm(ref_vector)
            
            # 计算X轴（与Y轴垂直）
            x_axis = ref_vector - np.dot(ref_vector, y_axis) * y_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
        else:
            # 默认X轴
            x_axis = np.array([1, 0, 0])
        
        # 计算Z轴（右手坐标系）
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 重新计算X轴确保正交
        x_axis = np.cross(y_axis, z_axis)
        
        # 构建旋转矩阵
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        
        return {
            'center': center,
            'rotation_matrix': rotation_matrix,
            'x_axis': x_axis,
            'y_axis': y_axis,
            'z_axis': z_axis
        }
    
    def _calculate_pose_confidence(self, keypoints_3d: np.ndarray, 
                                  body_coordinate_system: Dict) -> float:
        """
        计算姿态估计的置信度
        """
        # 基于关键点数量
        keypoint_score = min(1.0, len(keypoints_3d) / 10.0)
        
        # 基于关键点分布的合理性
        center = body_coordinate_system['center']
        distances = [np.linalg.norm(kp - center) for kp in keypoints_3d]
        
        # 检查距离分布是否合理（不应该有极端值）
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        if mean_distance > 0:
            distribution_score = 1.0 / (1.0 + std_distance / mean_distance)
        else:
            distribution_score = 0.5
        
        # 综合置信度
        confidence = (keypoint_score + distribution_score) / 2.0
        
        return min(1.0, max(0.0, confidence))
    
    def track_3d_pose_sequence(self, video_frames: List[np.ndarray],
                              keypoints_sequence: List[np.ndarray],
                              depth_maps: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """
        跟踪视频序列中的3D姿态
        """
        pose_sequence = []
        
        for i, (frame, keypoints_2d) in enumerate(zip(video_frames, keypoints_sequence)):
            depth_map = depth_maps[i] if depth_maps else None
            
            # 估计当前帧的3D姿态
            pose_3d = self.estimate_3d_pose_from_depth(frame, keypoints_2d, depth_map)
            
            if pose_3d and pose_3d['success']:
                pose_info = {
                    'frame_index': i,
                    'yaw': pose_3d['yaw'],
                    'pitch': pose_3d['pitch'],
                    'roll': pose_3d['roll'],
                    'confidence': pose_3d['confidence'],
                    'body_center': pose_3d['body_center'].tolist(),
                    'method': '3d_depth_based'
                }
            else:
                pose_info = {
                    'frame_index': i,
                    'yaw': None,
                    'pitch': None,
                    'roll': None,
                    'confidence': 0.0,
                    'body_center': None,
                    'method': 'failed'
                }
            
            pose_sequence.append(pose_info)
        
        # 后处理：平滑序列
        smoothed_sequence = self._smooth_pose_sequence(pose_sequence)
        
        return smoothed_sequence
    
    def _smooth_pose_sequence(self, pose_sequence: List[Dict]) -> List[Dict]:
        """
        平滑姿态序列
        """
        # 提取角度序列
        yaw_angles = [pose['yaw'] for pose in pose_sequence]
        pitch_angles = [pose['pitch'] for pose in pose_sequence]
        roll_angles = [pose['roll'] for pose in pose_sequence]
        
        # 对每个角度分别平滑
        smoothed_yaw = self._smooth_angle_sequence(yaw_angles)
        smoothed_pitch = self._smooth_angle_sequence(pitch_angles)
        smoothed_roll = self._smooth_angle_sequence(roll_angles)
        
        # 更新序列
        smoothed_sequence = []
        for i, pose in enumerate(pose_sequence):
            smoothed_pose = pose.copy()
            smoothed_pose['yaw'] = smoothed_yaw[i]
            smoothed_pose['pitch'] = smoothed_pitch[i]
            smoothed_pose['roll'] = smoothed_roll[i]
            smoothed_sequence.append(smoothed_pose)
        
        return smoothed_sequence
    
    def _smooth_angle_sequence(self, angles: List[Optional[float]], 
                              window_size: int = 5) -> List[Optional[float]]:
        """
        平滑角度序列
        """
        if not angles:
            return angles
        
        smoothed = []
        
        for i in range(len(angles)):
            if angles[i] is None:
                smoothed.append(None)
                continue
            
            # 收集窗口内的有效值
            start = max(0, i - window_size // 2)
            end = min(len(angles), i + window_size // 2 + 1)
            
            valid_values = [angles[j] for j in range(start, end) if angles[j] is not None]
            
            if valid_values:
                # 处理角度跳跃
                reference = angles[i]
                adjusted_values = []
                
                for val in valid_values:
                    diff = val - reference
                    if diff > 180:
                        val -= 360
                    elif diff < -180:
                        val += 360
                    adjusted_values.append(val)
                
                smoothed_value = np.mean(adjusted_values)
                
                # 确保角度在[-180, 180]范围内
                while smoothed_value > 180:
                    smoothed_value -= 360
                while smoothed_value < -180:
                    smoothed_value += 360
                
                smoothed.append(smoothed_value)
            else:
                smoothed.append(angles[i])
        
        return smoothed


class AdvancedBulletTimeAnalyzer:
    """
    高级子弹时间分析器
    结合深度信息进行更精确的分析
    """
    
    def __init__(self, device='cuda'):
        self.depth_pose_estimator = Depth3DPoseEstimator(device)
    
    def analyze_bullet_time_with_depth(self, video_frames: List[np.ndarray],
                                     keypoints_sequence: List[np.ndarray]) -> Dict:
        """
        基于深度信息分析子弹时间运镜
        """
        # 计算每帧的深度图
        depth_maps = []
        for frame in video_frames:
            depth_map = self.depth_pose_estimator.compute_depth_map(frame)
            depth_maps.append(depth_map)
        
        # 跟踪3D姿态序列
        pose_sequence = self.depth_pose_estimator.track_3d_pose_sequence(
            video_frames, keypoints_sequence, depth_maps)
        
        # 分析旋转模式
        rotation_analysis = self._analyze_rotation_pattern(pose_sequence)
        
        # 分析深度变化
        depth_analysis = self._analyze_depth_pattern(depth_maps, keypoints_sequence)
        
        # 综合分析
        bullet_time_result = self._comprehensive_bullet_time_analysis(
            rotation_analysis, depth_analysis, pose_sequence)
        
        return bullet_time_result
    
    def _analyze_rotation_pattern(self, pose_sequence: List[Dict]) -> Dict:
        """
        分析旋转模式
        """
        yaw_angles = [pose['yaw'] for pose in pose_sequence if pose['yaw'] is not None]
        
        if len(yaw_angles) < 5:
            return {'success': False, 'error': 'Insufficient pose data'}
        
        # 计算角度变化
        angle_changes = []
        for i in range(1, len(yaw_angles)):
            diff = yaw_angles[i] - yaw_angles[i-1]
            # 处理角度跳跃
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            angle_changes.append(diff)
        
        # 分析旋转特征
        total_rotation = sum(angle_changes)
        mean_angular_velocity = np.mean(angle_changes)
        angular_acceleration = np.std(angle_changes)
        
        # 判断旋转方向一致性
        direction_changes = sum(1 for i in range(1, len(angle_changes)) 
                              if np.sign(angle_changes[i]) != np.sign(angle_changes[i-1]))
        direction_consistency = 1.0 - (direction_changes / max(1, len(angle_changes) - 1))
        
        return {
            'success': True,
            'total_rotation': total_rotation,
            'mean_angular_velocity': mean_angular_velocity,
            'angular_acceleration': angular_acceleration,
            'direction_consistency': direction_consistency,
            'is_continuous_rotation': direction_consistency > 0.8 and abs(total_rotation) > 180
        }
    
    def _analyze_depth_pattern(self, depth_maps: List[Optional[np.ndarray]],
                             keypoints_sequence: List[np.ndarray]) -> Dict:
        """
        分析深度变化模式
        """
        if not depth_maps or len(depth_maps) < 2:
            return {'success': False, 'error': 'Insufficient depth data'}
        
        # 计算主体的平均深度变化
        subject_depths = []
        
        for depth_map, keypoints in zip(depth_maps, keypoints_sequence):
            if depth_map is None or len(keypoints) == 0:
                continue
            
            # 提取主体区域的深度
            h, w = depth_map.shape
            subject_pixels = []
            
            for kp in keypoints:
                if kp[2] > 0.5:  # 可见性阈值
                    x, y = kp[0], kp[1]
                    if x <= 1.0:  # 归一化坐标
                        px, py = int(x * w), int(y * h)
                    else:  # 像素坐标
                        px, py = int(x), int(y)
                    
                    if 0 <= px < w and 0 <= py < h:
                        subject_pixels.append(depth_map[py, px])
            
            if subject_pixels:
                subject_depths.append(np.mean(subject_pixels))
        
        if len(subject_depths) < 2:
            return {'success': False, 'error': 'No valid depth measurements'}
        
        # 分析深度变化
        depth_changes = np.diff(subject_depths)
        total_depth_change = subject_depths[-1] - subject_depths[0]
        depth_variance = np.var(subject_depths)
        
        return {
            'success': True,
            'total_depth_change': total_depth_change,
            'depth_variance': depth_variance,
            'mean_depth': np.mean(subject_depths),
            'depth_trend': 'approaching' if total_depth_change < 0 else 'receding'
        }
    
    def _comprehensive_bullet_time_analysis(self, rotation_analysis: Dict,
                                          depth_analysis: Dict,
                                          pose_sequence: List[Dict]) -> Dict:
        """
        综合分析子弹时间效果
        """
        result = {
            'is_bullet_time': False,
            'confidence': 0.0,
            'motion_type': 'unknown',
            'quality_metrics': {}
        }
        
        if not rotation_analysis['success']:
            result['error'] = 'Rotation analysis failed'
            return result
        
        # 判断是否为子弹时间
        is_continuous_rotation = rotation_analysis['is_continuous_rotation']
        direction_consistency = rotation_analysis['direction_consistency']
        total_rotation = abs(rotation_analysis['total_rotation'])
        
        # 子弹时间判断条件
        is_bullet_time = (
            is_continuous_rotation and
            direction_consistency > 0.7 and
            total_rotation > 90  # 至少90度旋转
        )
        
        if is_bullet_time:
            result['is_bullet_time'] = True
            
            # 确定旋转方向
            if rotation_analysis['total_rotation'] > 0:
                result['motion_type'] = 'bullet_time_clockwise'
            else:
                result['motion_type'] = 'bullet_time_counterclockwise'
            
            # 计算质量指标
            smoothness = 1.0 / (1.0 + rotation_analysis['angular_acceleration'])
            completeness = min(1.0, total_rotation / 360.0)
            
            result['quality_metrics'] = {
                'smoothness': smoothness,
                'direction_consistency': direction_consistency,
                'completeness': completeness,
                'total_rotation_degrees': rotation_analysis['total_rotation']
            }
            
            # 综合置信度
            confidence = (direction_consistency + smoothness + completeness) / 3.0
            result['confidence'] = confidence
            
            # 如果有深度信息，进一步分析
            if depth_analysis['success']:
                depth_consistency = 1.0 / (1.0 + depth_analysis['depth_variance'])
                result['quality_metrics']['depth_consistency'] = depth_consistency
                result['confidence'] = (result['confidence'] + depth_consistency) / 2.0
        
        return result


# 使用示例
if __name__ == "__main__":
    # 创建深度3D姿态估计器
    estimator = Depth3DPoseEstimator()
    
    # 创建高级子弹时间分析器
    analyzer = AdvancedBulletTimeAnalyzer()
    
    print("Depth-based 3D pose estimator initialized successfully!")
