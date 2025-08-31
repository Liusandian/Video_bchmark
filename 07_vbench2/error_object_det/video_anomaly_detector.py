"""
视频背景异物检测器
专门用于检测视频中突然出现的异物，如海面突然出现的海岛、天空中飞出的黑块等
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
import argparse
from tqdm import tqdm
import logging


class VideoAnomalyDetector:
    """
    视频异物检测器
    
    支持多种检测方法:
    1. 帧差法检测 - 检测帧间变化
    2. 背景建模法 - 基于背景模型检测异物
    3. 光流法 - 分析运动模式
    4. 边缘变化检测 - 检测结构性变化
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化检测器
        
        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            'frame_diff_threshold': 30,          # 帧差阈值
            'background_learning_rate': 0.01,    # 背景学习率
            'min_contour_area': 500,             # 最小轮廓面积
            'max_contour_area': 50000,           # 最大轮廓面积
            'gaussian_blur_kernel': (5, 5),      # 高斯模糊核大小
            'morphology_kernel_size': 5,         # 形态学操作核大小
            'optical_flow_threshold': 2.0,       # 光流阈值
            'edge_threshold': 50,                # 边缘检测阈值
            'temporal_consistency_frames': 3,    # 时序一致性检查帧数
            'anomaly_score_threshold': 0.7,      # 异物评分阈值
            'region_analysis_enabled': True,     # 是否启用区域分析
            'debug_mode': False                  # 调试模式
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化检测器组件
        self.background_subtractor = None
        self.optical_flow_detector = None
        self.edge_detector = None
        
        # 检测结果存储
        self.detection_results = []
        self.frame_anomalies = []
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO if not self.config['debug_mode'] else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_detectors(self):
        """初始化各种检测器"""
        # 背景减除器
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # 光流检测器参数
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # 特征检测器
        self.feature_detector = cv2.goodFeaturesToTrack
        
        self.logger.info("检测器初始化完成")
    
    def detect_anomalies_frame_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        基于帧差的异物检测
        
        Args:
            frame1: 前一帧
            frame2: 当前帧
            
        Returns:
            diff_mask: 差异掩码
            anomalies: 检测到的异物列表
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊降噪
        gray1 = cv2.GaussianBlur(gray1, self.config['gaussian_blur_kernel'], 0)
        gray2 = cv2.GaussianBlur(gray2, self.config['gaussian_blur_kernel'], 0)
        
        # 计算帧差
        frame_diff = cv2.absdiff(gray1, gray2)
        
        # 阈值化
        threshold = self.config['frame_diff_threshold']
        _, diff_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.config['morphology_kernel_size'], 
                                          self.config['morphology_kernel_size']))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓，提取异物
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_contour_area'] <= area <= self.config['max_contour_area']:
                # 计算边界框
                x, y, w, h = cv2.boundingRect(contour)
                
                # 计算异物特征
                anomaly = {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2),
                    'aspect_ratio': w / h,
                    'contour': contour.tolist(),
                    'confidence': min(1.0, area / self.config['max_contour_area']),
                    'method': 'frame_diff'
                }
                
                anomalies.append(anomaly)
        
        return diff_mask, anomalies
    
    def detect_anomalies_background_modeling(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        基于背景建模的异物检测
        
        Args:
            frame: 当前帧
            
        Returns:
            fg_mask: 前景掩码
            anomalies: 检测到的异物列表
        """
        # 应用背景减除
        fg_mask = self.background_subtractor.apply(frame, learningRate=self.config['background_learning_rate'])
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 分析轮廓
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config['min_contour_area'] <= area <= self.config['max_contour_area']:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 分析异物特征
                anomaly = {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2),
                    'aspect_ratio': w / h,
                    'contour': contour.tolist(),
                    'confidence': self._calculate_background_confidence(frame, fg_mask, contour),
                    'method': 'background_modeling'
                }
                
                anomalies.append(anomaly)
        
        return fg_mask, anomalies
    
    def detect_anomalies_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> List[Dict]:
        """
        基于光流的异物检测
        
        Args:
            prev_gray: 前一帧灰度图
            curr_gray: 当前帧灰度图
            
        Returns:
            anomalies: 检测到的异物列表
        """
        # 检测特征点
        features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, qualityLevel=0.3, 
                                         minDistance=7, blockSize=7)
        
        if features is None or len(features) < 10:
            return []
        
        # 计算光流
        next_features, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, features, None, **self.lk_params)
        
        # 筛选有效特征点
        good_features = features[status == 1]
        good_next = next_features[status == 1]
        
        if len(good_features) < 5:
            return []
        
        # 计算运动向量
        motion_vectors = good_next - good_features
        motion_magnitudes = np.sqrt(motion_vectors[:, 0]**2 + motion_vectors[:, 1]**2)
        
        # 检测异常运动
        threshold = self.config['optical_flow_threshold']
        anomalous_points = good_features[motion_magnitudes > threshold]
        anomalous_motions = motion_vectors[motion_magnitudes > threshold]
        
        if len(anomalous_points) == 0:
            return []
        
        # 聚类异常点
        anomalies = self._cluster_anomalous_points(anomalous_points, anomalous_motions)
        
        return anomalies
    
    def detect_anomalies_edge_analysis(self, frame1: np.ndarray, frame2: np.ndarray) -> List[Dict]:
        """
        基于边缘变化的异物检测
        
        Args:
            frame1: 前一帧
            frame2: 当前帧
            
        Returns:
            anomalies: 检测到的异物列表
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges1 = cv2.Canny(gray1, self.config['edge_threshold'], 
                          self.config['edge_threshold'] * 2)
        edges2 = cv2.Canny(gray2, self.config['edge_threshold'], 
                          self.config['edge_threshold'] * 2)
        
        # 计算边缘差异
        edge_diff = cv2.absdiff(edges1, edges2)
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_diff = cv2.morphologyEx(edge_diff, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edge_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        anomalies = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 较小的阈值，因为是边缘
                x, y, w, h = cv2.boundingRect(contour)
                
                anomaly = {
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center': (x + w//2, y + h//2),
                    'aspect_ratio': w / h,
                    'contour': contour.tolist(),
                    'confidence': min(1.0, area / 1000),
                    'method': 'edge_analysis'
                }
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_background_confidence(self, frame: np.ndarray, mask: np.ndarray, 
                                       contour: np.ndarray) -> float:
        """
        计算背景建模方法的置信度
        """
        # 创建轮廓掩码
        contour_mask = np.zeros_like(mask)
        cv2.fillPoly(contour_mask, [contour], 255)
        
        # 计算轮廓内的前景像素比例
        masked_fg = cv2.bitwise_and(mask, contour_mask)
        fg_pixels = np.sum(masked_fg > 0)
        total_pixels = np.sum(contour_mask > 0)
        
        if total_pixels == 0:
            return 0.0
        
        fg_ratio = fg_pixels / total_pixels
        
        # 计算颜色一致性
        masked_frame = cv2.bitwise_and(frame, frame, mask=contour_mask)
        if np.sum(contour_mask) > 0:
            roi = masked_frame[contour_mask > 0]
            color_std = np.std(roi) if len(roi) > 0 else 0
            color_consistency = 1.0 / (1.0 + color_std / 50.0)
        else:
            color_consistency = 0.0
        
        # 综合置信度
        confidence = (fg_ratio + color_consistency) / 2.0
        return min(1.0, max(0.0, confidence))
    
    def _cluster_anomalous_points(self, points: np.ndarray, motions: np.ndarray) -> List[Dict]:
        """
        聚类异常运动点
        """
        if len(points) < 3:
            return []
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(eps=30, min_samples=3).fit(points)
        labels = clustering.labels_
        
        anomalies = []
        for label in set(labels):
            if label == -1:  # 噪声点
                continue
            
            cluster_points = points[labels == label]
            cluster_motions = motions[labels == label]
            
            if len(cluster_points) < 3:
                continue
            
            # 计算聚类边界框
            x_coords = cluster_points[:, 0].astype(int)
            y_coords = cluster_points[:, 1].astype(int)
            
            x, y = np.min(x_coords), np.min(y_coords)
            w, h = np.max(x_coords) - x, np.max(y_coords) - y
            
            # 计算运动一致性
            motion_consistency = 1.0 - (np.std(cluster_motions) / (np.mean(np.linalg.norm(cluster_motions, axis=1)) + 1e-6))
            
            anomaly = {
                'bbox': (x, y, w, h),
                'area': w * h,
                'center': (x + w//2, y + h//2),
                'aspect_ratio': w / max(h, 1),
                'point_count': len(cluster_points),
                'motion_consistency': motion_consistency,
                'confidence': min(1.0, len(cluster_points) / 10.0 * motion_consistency),
                'method': 'optical_flow'
            }
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def analyze_temporal_consistency(self, anomalies_sequence: List[List[Dict]], 
                                   window_size: int = 3) -> List[Dict]:
        """
        分析时序一致性，过滤瞬时噪声
        
        Args:
            anomalies_sequence: 多帧异物检测结果序列
            window_size: 时间窗口大小
            
        Returns:
            consistent_anomalies: 时序一致的异物
        """
        if len(anomalies_sequence) < window_size:
            return []
        
        consistent_anomalies = []
        
        # 对每个时间窗口进行分析
        for i in range(len(anomalies_sequence) - window_size + 1):
            window = anomalies_sequence[i:i + window_size]
            
            # 检查是否有持续出现的异物
            for anomaly in window[window_size // 2]:  # 以中间帧为基准
                consistency_count = 0
                
                for frame_anomalies in window:
                    # 检查是否有相似位置的异物
                    for other_anomaly in frame_anomalies:
                        if self._are_anomalies_similar(anomaly, other_anomaly):
                            consistency_count += 1
                            break
                
                # 如果在窗口内大部分帧都出现，认为是一致的
                if consistency_count >= window_size * 0.6:
                    anomaly['temporal_consistency'] = consistency_count / window_size
                    anomaly['frame_index'] = i + window_size // 2
                    consistent_anomalies.append(anomaly)
        
        return consistent_anomalies
    
    def _are_anomalies_similar(self, anomaly1: Dict, anomaly2: Dict, 
                             distance_threshold: float = 50.0) -> bool:
        """
        判断两个异物是否相似（位置接近）
        """
        center1 = anomaly1['center']
        center2 = anomaly2['center']
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance < distance_threshold
    
    def calculate_anomaly_score(self, anomaly: Dict, frame_shape: Tuple[int, int]) -> float:
        """
        计算异物的综合评分
        
        Args:
            anomaly: 异物信息
            frame_shape: 帧尺寸 (height, width)
            
        Returns:
            score: 异物评分 (0-1)
        """
        height, width = frame_shape
        total_area = height * width
        
        # 基础置信度
        base_confidence = anomaly.get('confidence', 0.5)
        
        # 尺寸评分（中等尺寸获得更高分）
        area = anomaly['area']
        area_ratio = area / total_area
        if area_ratio < 0.001:  # 太小
            size_score = area_ratio * 1000  # 0-1
        elif area_ratio > 0.1:  # 太大
            size_score = max(0, 1 - (area_ratio - 0.1) * 5)
        else:
            size_score = 1.0
        
        # 位置评分（远离边缘的异物更可疑）
        x, y, w, h = anomaly['bbox']
        center_x, center_y = x + w//2, y + h//2
        
        # 计算到最近边缘的距离
        edge_distance = min(center_x, center_y, width - center_x, height - center_y)
        edge_distance_ratio = edge_distance / min(width, height)
        position_score = min(1.0, edge_distance_ratio * 4)  # 距离边缘1/4以上获得满分
        
        # 形状评分（过于极端的长宽比降分）
        aspect_ratio = anomaly['aspect_ratio']
        if 0.3 <= aspect_ratio <= 3.0:
            shape_score = 1.0
        else:
            shape_score = max(0.3, 1.0 / max(aspect_ratio, 1/aspect_ratio))
        
        # 时序一致性评分
        temporal_score = anomaly.get('temporal_consistency', 0.5)
        
        # 综合评分
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # 置信度、尺寸、位置、形状、时序
        scores = [base_confidence, size_score, position_score, shape_score, temporal_score]
        
        final_score = sum(w * s for w, s in zip(weights, scores))
        
        return min(1.0, max(0.0, final_score))
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        分析整个视频，检测异物
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            analysis_result: 分析结果
        """
        self.logger.info(f"开始分析视频: {video_path}")
        
        # 初始化检测器
        self.initialize_detectors()
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧")
        
        # 存储检测结果
        all_anomalies = []
        frame_results = []
        
        prev_frame = None
        prev_gray = None
        
        # 逐帧处理
        for frame_idx in tqdm(range(frame_count), desc="处理视频帧"):
            ret, frame = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_anomalies = []
            
            if prev_frame is not None:
                # 方法1: 帧差检测
                try:
                    _, diff_anomalies = self.detect_anomalies_frame_diff(prev_frame, frame)
                    frame_anomalies.extend(diff_anomalies)
                except Exception as e:
                    self.logger.warning(f"帧差检测失败 (帧{frame_idx}): {e}")
                
                # 方法3: 光流检测
                try:
                    flow_anomalies = self.detect_anomalies_optical_flow(prev_gray, curr_gray)
                    frame_anomalies.extend(flow_anomalies)
                except Exception as e:
                    self.logger.warning(f"光流检测失败 (帧{frame_idx}): {e}")
                
                # 方法4: 边缘分析
                try:
                    edge_anomalies = self.detect_anomalies_edge_analysis(prev_frame, frame)
                    frame_anomalies.extend(edge_anomalies)
                except Exception as e:
                    self.logger.warning(f"边缘分析失败 (帧{frame_idx}): {e}")
            
            # 方法2: 背景建模（每帧都可以检测）
            try:
                _, bg_anomalies = self.detect_anomalies_background_modeling(frame)
                frame_anomalies.extend(bg_anomalies)
            except Exception as e:
                self.logger.warning(f"背景建模失败 (帧{frame_idx}): {e}")
            
            # 计算异物评分
            for anomaly in frame_anomalies:
                anomaly['score'] = self.calculate_anomaly_score(anomaly, (height, width))
                anomaly['frame_index'] = frame_idx
                anomaly['timestamp'] = frame_idx / fps
            
            # 过滤低分异物
            high_score_anomalies = [a for a in frame_anomalies 
                                  if a['score'] >= self.config['anomaly_score_threshold']]
            
            frame_results.append({
                'frame_index': frame_idx,
                'timestamp': frame_idx / fps,
                'anomaly_count': len(high_score_anomalies),
                'anomalies': high_score_anomalies
            })
            
            all_anomalies.extend(high_score_anomalies)
            
            prev_frame = frame.copy()
            prev_gray = curr_gray.copy()
        
        cap.release()
        
        # 时序一致性分析
        anomalies_sequence = [frame_result['anomalies'] for frame_result in frame_results]
        consistent_anomalies = self.analyze_temporal_consistency(
            anomalies_sequence, self.config['temporal_consistency_frames'])
        
        # 汇总结果
        analysis_result = {
            'video_path': video_path,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': frame_count / fps
            },
            'detection_summary': {
                'total_anomalies': len(all_anomalies),
                'consistent_anomalies': len(consistent_anomalies),
                'frames_with_anomalies': len([f for f in frame_results if f['anomaly_count'] > 0]),
                'anomaly_rate': len([f for f in frame_results if f['anomaly_count'] > 0]) / len(frame_results)
            },
            'frame_results': frame_results,
            'consistent_anomalies': consistent_anomalies,
            'is_anomalous_video': len(consistent_anomalies) > 0,
            'anomaly_confidence': np.mean([a['score'] for a in consistent_anomalies]) if consistent_anomalies else 0.0
        }
        
        self.logger.info(f"视频分析完成: 检测到{len(consistent_anomalies)}个一致性异物")
        
        return analysis_result
    
    def save_results(self, result: Dict, output_path: str):
        """
        保存检测结果
        
        Args:
            result: 检测结果
            output_path: 输出文件路径
        """
        # 处理numpy类型以便JSON序列化
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
        
        converted_result = convert_numpy(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {output_path}")
    
    def visualize_anomalies(self, video_path: str, result: Dict, 
                          output_dir: str, max_frames: int = 10):
        """
        可视化检测结果
        
        Args:
            video_path: 视频文件路径
            result: 检测结果
            output_dir: 输出目录
            max_frames: 最大可视化帧数
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 选择要可视化的帧（有异物的帧）
        frames_with_anomalies = [f for f in result['frame_results'] if f['anomaly_count'] > 0]
        
        if not frames_with_anomalies:
            self.logger.info("没有检测到异物，无需可视化")
            return
        
        # 按异物数量排序，选择前N帧
        frames_with_anomalies.sort(key=lambda x: x['anomaly_count'], reverse=True)
        selected_frames = frames_with_anomalies[:max_frames]
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        for frame_info in selected_frames:
            frame_idx = frame_info['frame_index']
            
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # 绘制异物边界框
            vis_frame = frame.copy()
            for anomaly in frame_info['anomalies']:
                x, y, w, h = anomaly['bbox']
                score = anomaly['score']
                method = anomaly['method']
                
                # 根据评分选择颜色
                color = (0, 255, 0) if score > 0.8 else (0, 255, 255) if score > 0.6 else (0, 0, 255)
                
                # 绘制边界框
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
                
                # 添加标签
                label = f"{method[:4]} {score:.2f}"
                cv2.putText(vis_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 1)
            
            # 保存可视化结果
            output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_anomalies.jpg")
            cv2.imwrite(output_path, vis_frame)
        
        cap.release()
        self.logger.info(f"可视化结果已保存到: {output_dir}")
    
    def generate_report(self, result: Dict, output_path: str):
        """
        生成检测报告
        
        Args:
            result: 检测结果
            output_path: 报告输出路径
        """
        video_info = result['video_info']
        summary = result['detection_summary']
        
        report = f"""
# 视频异物检测报告

## 视频信息
- 文件路径: {result['video_path']}
- 分辨率: {video_info['width']}x{video_info['height']}
- 帧率: {video_info['fps']:.2f} fps
- 总帧数: {video_info['frame_count']}
- 时长: {video_info['duration']:.2f} 秒

## 检测结果摘要
- **是否包含异物**: {'是' if result['is_anomalous_video'] else '否'}
- **异物置信度**: {result['anomaly_confidence']:.3f}
- **总异物数量**: {summary['total_anomalies']}
- **一致性异物数量**: {summary['consistent_anomalies']}
- **包含异物的帧数**: {summary['frames_with_anomalies']}
- **异物出现率**: {summary['anomaly_rate']:.2%}

## 详细检测信息
"""
        
        if result['consistent_anomalies']:
            report += "\n### 检测到的一致性异物:\n"
            for i, anomaly in enumerate(result['consistent_anomalies']):
                report += f"""
{i+1}. 异物信息:
   - 位置: ({anomaly['center'][0]}, {anomaly['center'][1]})
   - 大小: {anomaly['area']} 像素
   - 评分: {anomaly['score']:.3f}
   - 检测方法: {anomaly['method']}
   - 出现帧: {anomaly['frame_index']}
   - 时间戳: {anomaly['timestamp']:.2f} 秒
"""
        
        # 时间线分析
        report += "\n### 异物出现时间线:\n"
        anomaly_frames = [f['frame_index'] for f in result['frame_results'] if f['anomaly_count'] > 0]
        if anomaly_frames:
            report += f"异物出现帧: {', '.join(map(str, anomaly_frames[:20]))}"
            if len(anomaly_frames) > 20:
                report += f"... (共{len(anomaly_frames)}帧)"
        else:
            report += "无异物检测"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"检测报告已保存到: {output_path}")


def main():
    """
    命令行主函数
    """
    parser = argparse.ArgumentParser(description='视频背景异物检测器')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, default='./anomaly_detection_output', 
                       help='输出目录')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径(JSON格式)')
    parser.add_argument('--threshold', '-t', type=float, default=0.7, 
                       help='异物评分阈值')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='生成可视化结果')
    parser.add_argument('--report', '-r', action='store_true', 
                       help='生成检测报告')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if not config:
        config = {}
    
    # 更新配置
    config['anomaly_score_threshold'] = args.threshold
    config['debug_mode'] = args.debug
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建检测器
    detector = VideoAnomalyDetector(config)
    
    try:
        # 执行检测
        result = detector.analyze_video(args.input)
        
        # 保存结果
        result_path = output_dir / 'detection_result.json'
        detector.save_results(result, str(result_path))
        
        # 生成可视化
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            detector.visualize_anomalies(args.input, result, str(vis_dir))
        
        # 生成报告
        if args.report:
            report_path = output_dir / 'detection_report.md'
            detector.generate_report(result, str(report_path))
        
        # 输出简要结果
        print(f"\n检测完成!")
        print(f"视频路径: {args.input}")
        print(f"是否包含异物: {'是' if result['is_anomalous_video'] else '否'}")
        if result['is_anomalous_video']:
            print(f"异物置信度: {result['anomaly_confidence']:.3f}")
            print(f"检测到{result['detection_summary']['consistent_anomalies']}个一致性异物")
        print(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
