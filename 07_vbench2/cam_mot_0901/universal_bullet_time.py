"""
通用子弹时间运镜检测模块
支持人像、建筑物、风景等多种场景的全局姿态估计和子弹时间检测
"""

import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from tqdm import tqdm
import json
import os
import sys
from math import sqrt, atan2, degrees, radians
from vbench2.utils import load_dimension_info, split_video_into_scenes

class GlobalPoseEstimator:
    """全局姿态估计器 - 适用于人像、建筑物、风景等多种场景"""
    
    def __init__(self, device):
        self.device = device
        self.feature_detector = cv2.ORB_create(nfeatures=500)  # 特征点检测器
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # 特征匹配器
        
    def extract_features(self, frame):
        """提取帧的特征点"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """匹配两帧之间的特征点"""
        if desc1 is None or desc2 is None:
            return []
        matches = self.matcher.match(desc1, desc2)
        # 按距离排序，取最佳匹配
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:min(50, len(matches))]  # 取前50个最佳匹配
    
    def estimate_rotation_from_features(self, kp1, kp2, matches):
        """基于特征点匹配估计旋转角度"""
        if len(matches) < 8:  # 需要足够的匹配点
            return None, None, None
        
        # 提取匹配点坐标
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 使用RANSAC估计基础矩阵
        fundamental_matrix, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99
        )
        
        if fundamental_matrix is None:
            return None, None, None
        
        # 计算图像中心点
        h, w = 480, 640  # 假设标准分辨率，实际应该从图像获取
        center = np.array([[w/2, h/2]])
        
        # 计算特征点相对于中心的向量
        center_pts1 = pts1.reshape(-1, 2) - center
        center_pts2 = pts2.reshape(-1, 2) - center
        
        # 计算平均旋转角度
        angles1 = np.arctan2(center_pts1[:, 1], center_pts1[:, 0])
        angles2 = np.arctan2(center_pts2[:, 1], center_pts2[:, 0])
        
        # 计算角度差
        angle_diffs = angles2 - angles1
        angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
        angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
        
        # 使用中位数作为鲁棒估计
        yaw_change = np.median(angle_diffs) if len(angle_diffs) > 0 else 0
        
        # 估计pitch和roll（简化版本）
        # 基于特征点的垂直和水平分布变化
        vertical_shift = np.median(center_pts2[:, 1] - center_pts1[:, 1])
        horizontal_shift = np.median(center_pts2[:, 0] - center_pts1[:, 0])
        
        # 转换为角度估计（简化模型）
        pitch_change = np.arctan2(vertical_shift, h/2) if h > 0 else 0
        roll_change = np.arctan2(horizontal_shift, w/2) if w > 0 else 0
        
        return degrees(yaw_change), degrees(pitch_change), degrees(roll_change)
    
    def estimate_global_pose_sequence(self, video_path):
        """估计视频的全局姿态序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []
        
        poses = []
        prev_kp, prev_desc = None, None
        cumulative_yaw, cumulative_pitch, cumulative_roll = 0.0, 0.0, 0.0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in tqdm(range(frame_count), desc="全局姿态估计", disable=True):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 提取当前帧特征
            curr_kp, curr_desc = self.extract_features(frame)
            
            if prev_kp is not None and prev_desc is not None:
                # 匹配特征点并估计旋转
                matches = self.match_features(prev_desc, curr_desc)
                yaw_delta, pitch_delta, roll_delta = self.estimate_rotation_from_features(
                    prev_kp, curr_kp, matches
                )
                
                if yaw_delta is not None:
                    # 累积角度变化
                    cumulative_yaw += yaw_delta
                    cumulative_pitch += pitch_delta
                    cumulative_roll += roll_delta
                    
                    poses.append((cumulative_yaw, cumulative_pitch, cumulative_roll))
                else:
                    # 如果估计失败，使用前一帧的值
                    poses.append((cumulative_yaw, cumulative_pitch, cumulative_roll))
            else:
                # 第一帧，初始姿态为0
                poses.append((0.0, 0.0, 0.0))
            
            prev_kp, prev_desc = curr_kp, curr_desc
        
        cap.release()
        return poses


class UniversalBulletTimeDetector:
    """通用子弹时间运镜检测器"""
    
    def __init__(self, device, submodules_dict):
        self.device = device
        self.yaw_threshold = 75.0  # yaw角度阈值
        self.consistency_threshold = 0.7  # 运动一致性阈值
        self.min_valid_frames = 15  # 最少有效帧数
        
        # 初始化全局姿态估计器
        self.pose_estimator = GlobalPoseEstimator(device)
        
        # 初始化CoTracker（用于环绕运镜检测）
        self._init_cotracker(submodules_dict)
    
    def _init_cotracker(self, submodules_dict):
        """初始化CoTracker模型"""
        try:
            self.cotracker = torch.hub.load(
                submodules_dict.get("repo", "facebookresearch/co-tracker"), 
                submodules_dict.get("model", "cotracker_stride_4_wind_8")
            ).to(self.device)
            self.grid_size = 10
            print("✓ CoTracker模型初始化成功")
        except Exception as e:
            print(f"CoTracker初始化失败: {e}")
            self.cotracker = None
    
    def detect_orbit_motion_universal(self, video_path):
        """通用环绕运镜检测（基于CoTracker轨迹分析）"""
        if self.cotracker is None:
            return False, 0.0, {}
        
        try:
            # 读取视频
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader)))
            video = video.permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
            
            # 获取轨迹
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker(video, grid_size=self.grid_size)
            
            tracks = pred_tracks[0].detach().cpu().numpy()  # T N 2
            visibility = pred_visibility[0].detach().cpu().numpy().squeeze(-1)  # T N
            
            # 分析环形运动
            h, w = video.shape[3], video.shape[4]
            cx, cy = w * 0.5, h * 0.5
            
            circular_scores = []
            radial_consistencies = []
            
            for t in range(tracks.shape[0] - 1):
                curr_tracks = tracks[t]
                next_tracks = tracks[t + 1]
                vis_mask = (visibility[t] > 0.5) & (visibility[t + 1] > 0.5)
                
                if vis_mask.sum() < 8:  # 需要足够的可见点
                    continue
                
                valid_curr = curr_tracks[vis_mask]
                valid_next = next_tracks[vis_mask]
                
                # 计算相对于中心的角度变化
                angles_curr = np.arctan2(valid_curr[:, 1] - cy, valid_curr[:, 0] - cx)
                angles_next = np.arctan2(valid_next[:, 1] - cy, valid_next[:, 0] - cx)
                
                # 处理角度跨越
                angle_diffs = angles_next - angles_curr
                angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
                angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
                
                if len(angle_diffs) > 0:
                    mean_angle_change = np.mean(angle_diffs)
                    circular_scores.append(mean_angle_change)
                    
                    # 计算径向运动一致性
                    distances_curr = np.linalg.norm(valid_curr - np.array([cx, cy]), axis=1)
                    distances_next = np.linalg.norm(valid_next - np.array([cx, cy]), axis=1)
                    radial_changes = distances_next - distances_curr
                    radial_consistency = 1.0 / (1.0 + np.std(radial_changes))
                    radial_consistencies.append(radial_consistency)
            
            if not circular_scores:
                return False, 0.0, {'error': 'insufficient_motion_data'}
            
            # 分析环形运动特征
            circular_scores = np.array(circular_scores)
            
            # 计算运动方向一致性
            positive_count = (circular_scores > 0).sum()
            negative_count = (circular_scores < 0).sum()
            total_count = len(circular_scores)
            direction_consistency = max(positive_count, negative_count) / total_count
            
            # 计算总旋转角度
            total_rotation_rad = np.sum(np.abs(circular_scores))
            total_rotation_deg = total_rotation_rad * 180 / np.pi
            
            # 计算平均径向一致性
            avg_radial_consistency = np.mean(radial_consistencies) if radial_consistencies else 0.5
            
            # 环绕运镜判定
            is_orbit = (
                direction_consistency > self.consistency_threshold and 
                total_rotation_deg > 45 and  # 至少45度的环形运动
                avg_radial_consistency > 0.6  # 径向运动相对稳定
            )
            
            confidence = min(1.0, direction_consistency * (total_rotation_deg / 90) * avg_radial_consistency)
            
            details = {
                'direction_consistency': direction_consistency,
                'total_rotation_deg': total_rotation_deg,
                'avg_radial_consistency': avg_radial_consistency,
                'motion_frames': len(circular_scores),
                'total_frames': tracks.shape[0]
            }
            
            return is_orbit, confidence, details
            
        except Exception as e:
            print(f"环绕运镜检测失败: {e}")
            return False, 0.0, {'error': str(e)}
    
    def analyze_global_yaw_rotation(self, poses):
        """分析全局yaw角度旋转"""
        if len(poses) < self.min_valid_frames:
            return False, 0.0, {'error': 'insufficient_frames'}
        
        # 提取yaw角度序列
        yaw_angles = np.array([pose[0] for pose in poses])
        
        # 角度平滑处理（去除噪声）
        from scipy.signal import savgol_filter
        try:
            if len(yaw_angles) >= 5:
                yaw_smoothed = savgol_filter(yaw_angles, 
                                           min(5, len(yaw_angles)//2*2+1), 2)
            else:
                yaw_smoothed = yaw_angles
        except:
            yaw_smoothed = yaw_angles
        
        # 计算总旋转角度
        total_rotation = abs(yaw_smoothed[-1] - yaw_smoothed[0])
        
        # 计算运动一致性
        diffs = np.diff(yaw_smoothed)
        if len(diffs) == 0:
            return False, 0.0, {'error': 'no_motion_detected'}
        
        # 分析运动方向一致性
        positive_diffs = (diffs > 1.0).sum()  # 阈值1度，避免噪声
        negative_diffs = (diffs < -1.0).sum()
        significant_motion_frames = positive_diffs + negative_diffs
        
        if significant_motion_frames == 0:
            return False, 0.0, {'error': 'no_significant_motion'}
        
        direction_consistency = max(positive_diffs, negative_diffs) / significant_motion_frames
        
        # 计算平均旋转速度
        avg_rotation_speed = total_rotation / len(yaw_angles)
        
        # 计算运动平滑度（避免抖动）
        motion_smoothness = 1.0 / (1.0 + np.std(diffs))
        
        # 综合判定
        is_valid_yaw_rotation = (
            total_rotation >= self.yaw_threshold and
            direction_consistency >= self.consistency_threshold and
            motion_smoothness > 0.3  # 运动相对平滑
        )
        
        # 计算置信度
        rotation_score = min(1.0, total_rotation / 180.0)
        consistency_score = direction_consistency
        smoothness_score = motion_smoothness
        
        confidence = (rotation_score * 0.5 + consistency_score * 0.3 + smoothness_score * 0.2)
        
        details = {
            'total_rotation': total_rotation,
            'direction_consistency': direction_consistency,
            'avg_rotation_speed': avg_rotation_speed,
            'motion_smoothness': motion_smoothness,
            'significant_motion_frames': significant_motion_frames,
            'total_frames': len(poses),
            'yaw_sequence': yaw_angles.tolist()
        }
        
        return is_valid_yaw_rotation, confidence, details
    
    def detect_universal_bullet_time(self, video_path):
        """通用子弹时间检测（支持人像、建筑物、风景等场景）"""
        print(f"开始检测视频: {os.path.basename(video_path)}")
        
        # 1. 全局姿态估计
        print("  步骤1: 全局姿态估计...")
        poses = self.pose_estimator.estimate_global_pose_sequence(video_path)
        
        if not poses:
            return {
                'is_bullet_time': False,
                'confidence': 0.0,
                'error': 'pose_estimation_failed',
                'global_pose': {'detected': False},
                'orbit_motion': {'detected': False}
            }
        
        # 2. 分析全局yaw旋转
        print("  步骤2: 分析yaw角度旋转...")
        has_yaw_rotation, yaw_confidence, yaw_details = self.analyze_global_yaw_rotation(poses)
        
        # 3. 检测环绕运镜
        print("  步骤3: 检测环绕运镜...")
        has_orbit, orbit_confidence, orbit_details = self.detect_orbit_motion_universal(video_path)
        
        # 4. 综合判断子弹时间
        print("  步骤4: 综合判断...")
        
        # 子弹时间需要同时满足：全局yaw旋转 + 环绕运镜
        is_bullet_time = has_yaw_rotation and has_orbit
        
        # 计算综合置信度
        if is_bullet_time:
            # 加权平均：yaw旋转占主导地位
            combined_confidence = yaw_confidence * 0.7 + orbit_confidence * 0.3
        else:
            combined_confidence = 0.0
        
        # 构建详细结果
        result = {
            'is_bullet_time': is_bullet_time,
            'confidence': combined_confidence,
            'scene_type': self._classify_scene_type(video_path),
            'global_pose': {
                'detected': has_yaw_rotation,
                'confidence': yaw_confidence,
                'details': yaw_details
            },
            'orbit_motion': {
                'detected': has_orbit,
                'confidence': orbit_confidence,
                'details': orbit_details
            },
            'summary': {
                'total_yaw_rotation': yaw_details.get('total_rotation', 0),
                'motion_consistency': yaw_details.get('direction_consistency', 0),
                'orbit_rotation': orbit_details.get('total_rotation_deg', 0),
                'detection_quality': min(yaw_confidence, orbit_confidence)
            }
        }
        
        print(f"  检测完成: {'✓ 子弹时间' if is_bullet_time else '✗ 非子弹时间'} (置信度: {combined_confidence:.3f})")
        return result
    
    def _classify_scene_type(self, video_path):
        """简单的场景类型分类（基于视频特征）"""
        # 这里可以集成更复杂的场景分类模型
        # 目前返回通用类型
        return "universal"


def compute_universal_bullet_time(json_dir, device, submodules_dict, **kwargs):
    """VBench2通用子弹时间检测评估函数"""
    detector = UniversalBulletTimeDetector(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='bullet_time', lang='en')
    
    video_results = []
    scores = []
    scene_stats = {'universal': 0, 'portrait': 0, 'architecture': 0, 'landscape': 0}
    
    for prompt_dict in tqdm(prompt_dict_ls, desc="通用子弹时间检测评估"):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            try:
                result = detector.detect_universal_bullet_time(video_path)
                score = 1.0 if result['is_bullet_time'] else 0.0
                
                # 统计场景类型
                scene_type = result.get('scene_type', 'universal')
                scene_stats[scene_type] = scene_stats.get(scene_type, 0) + 1
                
                video_results.append({
                    'video_path': video_path, 
                    'video_results': score,
                    'scene_type': scene_type,
                    'details': result
                })
                scores.append(score)
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                video_results.append({
                    'video_path': video_path, 
                    'video_results': 0.0,
                    'error': str(e)
                })
                scores.append(0.0)
    
    avg_score = np.mean(scores) if scores else 0.0
    
    # 添加统计信息
    evaluation_summary = {
        'average_score': avg_score,
        'total_videos': len(video_results),
        'detected_bullet_time': sum(scores),
        'scene_statistics': scene_stats
    }
    
    return avg_score, video_results, evaluation_summary


# 集成到原有的camera_motion系统
class EnhancedCameraPredict:
    """增强的相机预测器，集成通用子弹时间检测"""
    
    def __init__(self, device, submodules_dict):
        # 初始化原有的CoTracker模型
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        
        try:
            self.model = torch.hub.load(
                submodules_dict["repo"], 
                submodules_dict["model"]
            ).to(self.device)
        except:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(
                submodules_dict["repo"], 
                submodules_dict["model"]
            ).to(self.device)
        
        # 初始化通用子弹时间检测器
        self.bullet_time_detector = UniversalBulletTimeDetector(device, submodules_dict)
    
    def predict_with_universal_bullet_time(self, video_path, fps, end_frame):
        """集成通用子弹时间检测的预测方法"""
        # 1. 原有的相机运动分类
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader))) 
        video = video.permute(0, 3, 1, 2)[None].float().to(self.device)
        
        # 原有轨迹分析
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size)
        if end_frame != -1:
            pred_tracks = pred_tracks[:, :end_frame]
            pred_visibility = pred_visibility[:, :end_frame]
        
        pred_track = pred_tracks[0].long().detach().cpu().numpy()
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks = [pred_track[i].reshape(self.grid_size, self.grid_size, 2) 
                 for i in range(0, len(pred_track), 20)]
        
        # 简化的运动分类（从原代码提取）
        standard_results = self._simple_camera_classify(track1, track2, tracks)
        
        # 2. 通用子弹时间检测
        bullet_result = self.bullet_time_detector.detect_universal_bullet_time(video_path)
        
        # 3. 构建增强结果
        enhanced_results = standard_results.copy()
        if bullet_result['is_bullet_time']:
            if "bullet_time" not in enhanced_results:
                enhanced_results.append("bullet_time")
        
        return {
            'standard_motions': standard_results,
            'enhanced_motions': enhanced_results,
            'bullet_time': bullet_result,
            'scene_type': bullet_result.get('scene_type', 'universal')
        }
    
    def _simple_camera_classify(self, track1, track2, tracks):
        """简化的相机运动分类"""
        # 这里可以实现原有的分类逻辑
        # 为简化，返回基本分类
        return ["unknown"]


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submodules_dict = {
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8'
    }
    
    detector = UniversalBulletTimeDetector(device, submodules_dict)
    
    # 测试视频路径
    test_videos = [
        "path/to/portrait_bullet_time.mp4",
        "path/to/architecture_bullet_time.mp4", 
        "path/to/landscape_bullet_time.mp4"
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n测试视频: {video_path}")
            result = detector.detect_universal_bullet_time(video_path)
            print(f"检测结果: {result['is_bullet_time']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"场景类型: {result['scene_type']}")
        else:
            print(f"视频文件不存在: {video_path}")
