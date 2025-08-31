"""
基于光流和特征点的旋转角度计算模块
专门用于处理子弹时间运镜中人脸不可见时的yaw角度估计
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize
import math


class OpticalFlowRotationEstimator:
    """
    基于光流的旋转角度估计器
    """
    
    def __init__(self):
        # 设置光流检测参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征点检测参数
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # RAFT光流模型（如果可用）
        self.raft_model = None
        try:
            self._load_raft_model()
        except:
            print("RAFT model not available, using Lucas-Kanade optical flow")
    
    def _load_raft_model(self):
        """
        加载RAFT光流模型（如果可用）
        """
        try:
            # 这里应该加载预训练的RAFT模型
            # 由于依赖问题，这里只是占位符
            pass
        except Exception as e:
            print(f"Failed to load RAFT model: {e}")
            self.raft_model = None
    
    def estimate_rotation_from_optical_flow(self, frame1: np.ndarray, 
                                          frame2: np.ndarray,
                                          mask: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        基于光流估计帧间旋转角度
        
        Args:
            frame1: 第一帧图像 [H, W, C]
            frame2: 第二帧图像 [H, W, C]
            mask: 可选的人体区域mask [H, W]
        
        Returns:
            rotation_info: 旋转信息字典
        """
        if frame1.shape != frame2.shape:
            return None
        
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 使用RAFT或LK光流
        if self.raft_model is not None:
            flow = self._compute_raft_flow(frame1, frame2)
        else:
            flow = self._compute_lk_flow(gray1, gray2, mask)
        
        if flow is None:
            return None
        
        # 分析光流模式
        rotation_analysis = self._analyze_flow_pattern(flow, mask)
        
        return rotation_analysis
    
    def _compute_lk_flow(self, gray1: np.ndarray, gray2: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        计算Lucas-Kanade光流
        """
        # 检测特征点
        if mask is not None:
            corners = cv2.goodFeaturesToTrack(gray1, mask=mask.astype(np.uint8), **self.feature_params)
        else:
            corners = cv2.goodFeaturesToTrack(gray1, **self.feature_params)
        
        if corners is None or len(corners) < 10:
            return None
        
        # 计算光流
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, corners, None, **self.lk_params)
        
        # 筛选有效的特征点
        good_new = next_corners[status == 1]
        good_old = corners[status == 1]
        
        if len(good_new) < 5:
            return None
        
        # 计算光流向量
        flow_vectors = good_new - good_old
        flow_points = good_old.reshape(-1, 2)
        
        return {
            'vectors': flow_vectors,
            'points': flow_points,
            'method': 'lucas_kanade'
        }
    
    def _compute_raft_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Optional[Dict]:
        """
        计算RAFT密集光流
        """
        # 这里应该使用RAFT模型计算密集光流
        # 由于模型依赖问题，暂时返回None
        return None
    
    def _analyze_flow_pattern(self, flow_data: Dict, 
                            mask: Optional[np.ndarray] = None) -> Dict:
        """
        分析光流模式，估计旋转角度
        """
        vectors = flow_data['vectors']
        points = flow_data['points']
        
        if len(vectors) < 5:
            return {'rotation_angle': None, 'confidence': 0.0}
        
        # 计算图像中心
        if mask is not None:
            # 如果有mask，使用mask的中心
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
            else:
                center_x, center_y = mask.shape[1] // 2, mask.shape[0] // 2
        else:
            center_x, center_y = points[:, 0].mean(), points[:, 1].mean()
        
        center = np.array([center_x, center_y])
        
        # 方法1: 基于切线运动分析
        rotation_angle_tangential = self._estimate_rotation_from_tangential_flow(
            points, vectors, center)
        
        # 方法2: 基于角度变化分析
        rotation_angle_angular = self._estimate_rotation_from_angular_change(
            points, vectors, center)
        
        # 方法3: 基于旋转中心估计
        rotation_center, rotation_angle_center = self._estimate_rotation_center_and_angle(
            points, vectors)
        
        # 综合多种方法的结果
        angles = [angle for angle in [rotation_angle_tangential, rotation_angle_angular, 
                                    rotation_angle_center] if angle is not None]
        
        if not angles:
            return {'rotation_angle': None, 'confidence': 0.0}
        
        # 计算加权平均（简单平均）
        final_angle = np.mean(angles)
        
        # 计算置信度（基于一致性）
        if len(angles) > 1:
            confidence = 1.0 / (1.0 + np.std(angles) / 10.0)
        else:
            confidence = 0.5
        
        return {
            'rotation_angle': final_angle,
            'confidence': confidence,
            'method_angles': {
                'tangential': rotation_angle_tangential,
                'angular': rotation_angle_angular,
                'center_based': rotation_angle_center
            },
            'rotation_center': rotation_center
        }
    
    def _estimate_rotation_from_tangential_flow(self, points: np.ndarray, 
                                              vectors: np.ndarray, 
                                              center: np.ndarray) -> Optional[float]:
        """
        基于切线运动估计旋转角度
        """
        if len(points) < 3:
            return None
        
        rotation_components = []
        
        for i, (point, vector) in enumerate(zip(points, vectors)):
            # 计算从中心到点的向量
            radius_vector = point - center
            radius_distance = np.linalg.norm(radius_vector)
            
            if radius_distance < 1e-6:  # 避免除零
                continue
            
            # 计算切线方向（垂直于半径向量）
            tangent_direction = np.array([-radius_vector[1], radius_vector[0]])
            tangent_direction = tangent_direction / np.linalg.norm(tangent_direction)
            
            # 计算光流在切线方向上的分量
            tangential_component = np.dot(vector, tangent_direction)
            
            # 转换为角度变化（弧度）
            angular_velocity = tangential_component / radius_distance
            rotation_components.append(angular_velocity)
        
        if not rotation_components:
            return None
        
        # 计算平均角速度
        mean_angular_velocity = np.mean(rotation_components)
        
        # 转换为度数
        rotation_angle_degrees = np.degrees(mean_angular_velocity)
        
        return rotation_angle_degrees
    
    def _estimate_rotation_from_angular_change(self, points: np.ndarray,
                                             vectors: np.ndarray,
                                             center: np.ndarray) -> Optional[float]:
        """
        基于角度变化估计旋转
        """
        if len(points) < 3:
            return None
        
        angle_changes = []
        
        for point, vector in zip(points, vectors):
            # 原始位置相对中心的角度
            original_angle = np.arctan2(point[1] - center[1], point[0] - center[0])
            
            # 新位置
            new_point = point + vector
            new_angle = np.arctan2(new_point[1] - center[1], new_point[0] - center[0])
            
            # 角度变化
            angle_change = new_angle - original_angle
            
            # 处理角度跳跃
            if angle_change > np.pi:
                angle_change -= 2 * np.pi
            elif angle_change < -np.pi:
                angle_change += 2 * np.pi
            
            angle_changes.append(angle_change)
        
        if not angle_changes:
            return None
        
        # 计算平均角度变化
        mean_angle_change = np.mean(angle_changes)
        
        # 转换为度数
        rotation_angle_degrees = np.degrees(mean_angle_change)
        
        return rotation_angle_degrees
    
    def _estimate_rotation_center_and_angle(self, points: np.ndarray,
                                          vectors: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        估计旋转中心和角度
        """
        if len(points) < 4:
            return None, None
        
        # 使用最小二乘法估计旋转中心
        def objective_function(center_candidate):
            center = np.array(center_candidate)
            errors = []
            
            for point, vector in zip(points, vectors):
                # 计算半径
                radius1 = np.linalg.norm(point - center)
                radius2 = np.linalg.norm(point + vector - center)
                
                # 旋转应该保持半径不变
                radius_error = abs(radius1 - radius2)
                errors.append(radius_error)
            
            return np.mean(errors)
        
        # 初始猜测：特征点的中心
        initial_center = np.mean(points, axis=0)
        
        try:
            # 优化旋转中心
            result = minimize(objective_function, initial_center, method='Nelder-Mead')
            estimated_center = result.x
            
            # 基于估计的中心计算旋转角度
            rotation_angle = self._estimate_rotation_from_angular_change(
                points, vectors, estimated_center)
            
            return estimated_center, rotation_angle
            
        except Exception as e:
            print(f"Rotation center estimation failed: {e}")
            return None, None


class FeatureTrackingRotationEstimator:
    """
    基于特征点追踪的旋转角度估计器
    """
    
    def __init__(self):
        # ORB特征检测器
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # SIFT特征检测器（如果可用）
        try:
            self.sift = cv2.SIFT_create()
        except:
            self.sift = None
        
        # 特征匹配器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def estimate_rotation_from_features(self, frame1: np.ndarray,
                                      frame2: np.ndarray,
                                      mask: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        基于特征点匹配估计旋转角度
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 检测和描述特征点
        kp1, des1 = self._detect_features(gray1, mask)
        kp2, des2 = self._detect_features(gray2, mask)
        
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None
        
        # 匹配特征点
        matches = self._match_features(des1, des2)
        
        if len(matches) < 10:
            return None
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 估计变换矩阵
        transformation_analysis = self._analyze_transformation(src_pts, dst_pts)
        
        return transformation_analysis
    
    def _detect_features(self, image: np.ndarray, 
                        mask: Optional[np.ndarray] = None) -> Tuple[List, Optional[np.ndarray]]:
        """
        检测特征点
        """
        if self.sift is not None:
            # 使用SIFT
            kp, des = self.sift.detectAndCompute(image, mask)
        else:
            # 使用ORB
            kp, des = self.orb.detectAndCompute(image, mask)
        
        return kp, des
    
    def _match_features(self, des1: np.ndarray, des2: np.ndarray) -> List:
        """
        匹配特征点
        """
        matches = self.matcher.match(des1, des2)
        
        # 按距离排序
        matches = sorted(matches, key=lambda x: x.distance)
        
        # 保留最好的匹配
        good_matches = matches[:min(100, len(matches) // 2)]
        
        return good_matches
    
    def _analyze_transformation(self, src_pts: np.ndarray, 
                              dst_pts: np.ndarray) -> Dict:
        """
        分析变换矩阵，提取旋转信息
        """
        try:
            # 估计仿射变换矩阵
            affine_matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC, 
                ransacReprojThreshold=5.0, confidence=0.99)
            
            if affine_matrix is None:
                return {'rotation_angle': None, 'confidence': 0.0}
            
            # 从仿射矩阵提取旋转角度
            rotation_angle = self._extract_rotation_from_affine(affine_matrix)
            
            # 计算置信度（基于内点比例）
            inlier_ratio = np.sum(inliers) / len(inliers) if inliers is not None else 0.0
            confidence = min(0.9, inlier_ratio)
            
            # 计算平移信息
            translation = affine_matrix[:2, 2]
            
            # 计算缩放信息
            scale_x = np.sqrt(affine_matrix[0, 0]**2 + affine_matrix[0, 1]**2)
            scale_y = np.sqrt(affine_matrix[1, 0]**2 + affine_matrix[1, 1]**2)
            
            return {
                'rotation_angle': rotation_angle,
                'confidence': confidence,
                'translation': translation,
                'scale': (scale_x, scale_y),
                'inlier_ratio': inlier_ratio,
                'affine_matrix': affine_matrix
            }
            
        except Exception as e:
            print(f"Transformation analysis failed: {e}")
            return {'rotation_angle': None, 'confidence': 0.0}
    
    def _extract_rotation_from_affine(self, affine_matrix: np.ndarray) -> float:
        """
        从仿射变换矩阵提取旋转角度
        """
        # 提取旋转部分
        rotation_matrix = affine_matrix[:2, :2]
        
        # 计算旋转角度
        rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        # 转换为度数
        rotation_angle_degrees = np.degrees(rotation_angle)
        
        return rotation_angle_degrees


class MultiMethodRotationEstimator:
    """
    多方法融合的旋转角度估计器
    """
    
    def __init__(self):
        self.optical_flow_estimator = OpticalFlowRotationEstimator()
        self.feature_tracking_estimator = FeatureTrackingRotationEstimator()
    
    def estimate_rotation_multi_method(self, frame1: np.ndarray,
                                     frame2: np.ndarray,
                                     mask: Optional[np.ndarray] = None) -> Dict:
        """
        使用多种方法估计旋转角度
        """
        results = {
            'optical_flow': None,
            'feature_tracking': None,
            'final_rotation': None,
            'confidence': 0.0,
            'method_used': None
        }
        
        # 方法1: 光流分析
        try:
            optical_flow_result = self.optical_flow_estimator.estimate_rotation_from_optical_flow(
                frame1, frame2, mask)
            if optical_flow_result:
                results['optical_flow'] = optical_flow_result
        except Exception as e:
            print(f"Optical flow estimation failed: {e}")
        
        # 方法2: 特征点追踪
        try:
            feature_result = self.feature_tracking_estimator.estimate_rotation_from_features(
                frame1, frame2, mask)
            if feature_result:
                results['feature_tracking'] = feature_result
        except Exception as e:
            print(f"Feature tracking estimation failed: {e}")
        
        # 融合结果
        final_result = self._fuse_results(results)
        results.update(final_result)
        
        return results
    
    def _fuse_results(self, results: Dict) -> Dict:
        """
        融合多种方法的结果
        """
        angles = []
        confidences = []
        methods = []
        
        # 收集有效结果
        if results['optical_flow'] and results['optical_flow']['rotation_angle'] is not None:
            angles.append(results['optical_flow']['rotation_angle'])
            confidences.append(results['optical_flow']['confidence'])
            methods.append('optical_flow')
        
        if results['feature_tracking'] and results['feature_tracking']['rotation_angle'] is not None:
            angles.append(results['feature_tracking']['rotation_angle'])
            confidences.append(results['feature_tracking']['confidence'])
            methods.append('feature_tracking')
        
        if not angles:
            return {
                'final_rotation': None,
                'confidence': 0.0,
                'method_used': 'none'
            }
        
        # 加权平均
        weights = np.array(confidences)
        weights = weights / np.sum(weights)  # 归一化
        
        weighted_angle = np.average(angles, weights=weights)
        final_confidence = np.max(confidences)  # 使用最高置信度
        
        # 选择主要方法
        best_method_idx = np.argmax(confidences)
        primary_method = methods[best_method_idx]
        
        return {
            'final_rotation': weighted_angle,
            'confidence': final_confidence,
            'method_used': primary_method,
            'all_angles': angles,
            'weights': weights.tolist()
        }
    
    def estimate_video_rotation_sequence(self, video_frames: List[np.ndarray],
                                       masks: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """
        估计视频序列的旋转角度
        """
        if len(video_frames) < 2:
            return []
        
        rotation_sequence = []
        cumulative_rotation = 0.0
        
        for i in range(len(video_frames) - 1):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            mask = masks[i] if masks else None
            
            # 估计帧间旋转
            rotation_result = self.estimate_rotation_multi_method(frame1, frame2, mask)
            
            # 累积旋转角度
            if rotation_result['final_rotation'] is not None:
                frame_rotation = rotation_result['final_rotation']
                cumulative_rotation += frame_rotation
            else:
                frame_rotation = 0.0
            
            rotation_info = {
                'frame_pair': (i, i + 1),
                'frame_rotation': frame_rotation,
                'cumulative_rotation': cumulative_rotation,
                'confidence': rotation_result['confidence'],
                'method_used': rotation_result['method_used'],
                'detailed_results': rotation_result
            }
            
            rotation_sequence.append(rotation_info)
        
        return rotation_sequence


# 使用示例
if __name__ == "__main__":
    # 创建多方法旋转估计器
    estimator = MultiMethodRotationEstimator()
    
    # 模拟视频帧
    # frame1 = cv2.imread("frame1.jpg")
    # frame2 = cv2.imread("frame2.jpg")
    # mask = cv2.imread("mask.jpg", 0)  # 人体区域mask
    
    # 估计旋转角度
    # result = estimator.estimate_rotation_multi_method(frame1, frame2, mask)
    
    print("Optical flow rotation estimator initialized successfully!")
