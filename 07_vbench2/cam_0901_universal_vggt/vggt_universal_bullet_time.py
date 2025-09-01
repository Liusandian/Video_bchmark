"""
基于VGGT的通用子弹时间运镜检测模块
结合VGGT全局姿态估计和CoTracker轨迹分析实现多场景子弹时间检测
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
import torchvision.transforms as transforms
from PIL import Image
from vbench2.utils import load_dimension_info, split_video_into_scenes

# 可选的人脸检测依赖
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False


class VGGTGlobalPoseEstimator:
    """基于VGGT的全局姿态估计器 - 扩展支持全局场景分析"""
    
    def __init__(self, device, vggt_path, model_weights=None):
        self.device = device
        self.vggt_model = None
        self.face_detector = None
        self._init_vggt_model(vggt_path, model_weights)
        self._init_face_detector()
        
        # 全局分析参数
        self.grid_regions = 9  # 3x3网格分析
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def _init_vggt_model(self, vggt_path, model_weights):
        """初始化VGGT模型"""
        try:
            # 添加VGGT路径到系统路径
            if vggt_path not in sys.path:
                sys.path.append(vggt_path)
            
            # 导入VGGT模型 - 根据实际VGGT结构调整
            try:
                from model import VGGT
                self.vggt_model = VGGT().to(self.device)
            except ImportError:
                try:
                    from models.vggt import VGGT
                    self.vggt_model = VGGT().to(self.device)
                except ImportError:
                    # 尝试其他可能的导入路径
                    from vggt import VGGT
                    self.vggt_model = VGGT().to(self.device)
            
            # 加载预训练权重
            if model_weights and os.path.exists(model_weights):
                checkpoint = torch.load(model_weights, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.vggt_model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    self.vggt_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.vggt_model.load_state_dict(checkpoint)
                print("✓ VGGT模型权重加载成功")
            else:
                print("⚠ 使用VGGT默认权重")
            
            self.vggt_model.eval()
            print("✓ VGGT全局姿态估计模型初始化成功")
            
        except Exception as e:
            print(f"VGGT模型初始化失败: {e}")
            print("将使用备用的特征点匹配方法")
            self.vggt_model = None
    
    def _init_face_detector(self):
        """初始化人脸检测器"""
        if _HAS_DLIB:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                print("✓ Dlib人脸检测器初始化成功")
            except Exception as e:
                print(f"Dlib初始化失败: {e}")
                self.face_detector = None
    
    def preprocess_for_vggt(self, image_region):
        """为VGGT模型预处理图像区域"""
        # 标准化预处理
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image_region).unsqueeze(0).to(self.device)
    
    def extract_face_pose_with_vggt(self, frame):
        """使用VGGT提取人脸姿态（如果存在人脸）"""
        if self.vggt_model is None or self.face_detector is None:
            return None
        
        try:
            # 检测人脸
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if len(faces) == 0:
                return None
            
            # 选择最大的人脸
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
            
            # 扩展人脸区域
            margin = 20
            h, w = frame.shape[:2]
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size == 0:
                return None
            
            # 转换为RGB并预处理
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess_for_vggt(face_rgb)
            
            # VGGT推理
            with torch.no_grad():
                outputs = self.vggt_model(input_tensor)
                
                # 根据VGGT的输出格式调整
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    yaw, pitch, roll = outputs
                elif isinstance(outputs, torch.Tensor):
                    if outputs.shape[-1] == 3:
                        yaw, pitch, roll = outputs[0, 0], outputs[0, 1], outputs[0, 2]
                    else:
                        # 假设输出是单个角度或需要其他处理
                        yaw = outputs[0] if len(outputs.shape) > 0 else outputs
                        pitch = roll = torch.tensor(0.0).to(self.device)
                else:
                    return None
                
                # 转换为numpy
                yaw = float(yaw.cpu().numpy()) if hasattr(yaw, 'cpu') else float(yaw)
                pitch = float(pitch.cpu().numpy()) if hasattr(pitch, 'cpu') else float(pitch)
                roll = float(roll.cpu().numpy()) if hasattr(roll, 'cpu') else float(roll)
                
                return {'yaw': yaw, 'pitch': pitch, 'roll': roll, 'confidence': 1.0, 'source': 'face'}
                
        except Exception as e:
            print(f"VGGT人脸姿态估计失败: {e}")
            return None
    
    def extract_global_pose_with_vggt(self, frame):
        """使用VGGT进行全局姿态估计（多区域分析）"""
        if self.vggt_model is None:
            return None
        
        h, w = frame.shape[:2]
        region_poses = []
        
        # 将图像分为多个区域进行分析
        regions = [
            (0, 0, w//2, h//2),           # 左上
            (w//2, 0, w, h//2),           # 右上
            (0, h//2, w//2, h),           # 左下
            (w//2, h//2, w, h),           # 右下
            (w//4, h//4, 3*w//4, 3*h//4), # 中心区域
        ]
        
        try:
            for i, (x1, y1, x2, y2) in enumerate(regions):
                region = frame[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                # 转换为RGB并预处理
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
                input_tensor = self.preprocess_for_vggt(region_rgb)
                
                # VGGT推理
                with torch.no_grad():
                    outputs = self.vggt_model(input_tensor)
                    
                    # 处理输出
                    if isinstance(outputs, tuple) and len(outputs) == 3:
                        yaw, pitch, roll = outputs
                    elif isinstance(outputs, torch.Tensor):
                        if outputs.shape[-1] >= 3:
                            yaw, pitch, roll = outputs[0, 0], outputs[0, 1], outputs[0, 2]
                        else:
                            yaw = outputs[0] if len(outputs.shape) > 0 else outputs
                            pitch = roll = torch.tensor(0.0).to(self.device)
                    else:
                        continue
                    
                    # 转换为numpy
                    yaw = float(yaw.cpu().numpy()) if hasattr(yaw, 'cpu') else float(yaw)
                    pitch = float(pitch.cpu().numpy()) if hasattr(pitch, 'cpu') else float(pitch)
                    roll = float(roll.cpu().numpy()) if hasattr(roll, 'cpu') else float(roll)
                    
                    region_poses.append({
                        'region': i,
                        'yaw': yaw,
                        'pitch': pitch, 
                        'roll': roll,
                        'weight': 1.0
                    })
            
            if not region_poses:
                return None
            
            # 计算加权平均姿态
            total_weight = sum(pose['weight'] for pose in region_poses)
            if total_weight == 0:
                return None
            
            avg_yaw = sum(pose['yaw'] * pose['weight'] for pose in region_poses) / total_weight
            avg_pitch = sum(pose['pitch'] * pose['weight'] for pose in region_poses) / total_weight
            avg_roll = sum(pose['roll'] * pose['weight'] for pose in region_poses) / total_weight
            
            # 计算一致性（作为置信度）
            yaw_std = np.std([pose['yaw'] for pose in region_poses])
            consistency = max(0.0, 1.0 - yaw_std / 45.0)  # 45度为参考标准差
            
            return {
                'yaw': avg_yaw,
                'pitch': avg_pitch,
                'roll': avg_roll,
                'confidence': consistency,
                'source': 'global',
                'region_count': len(region_poses)
            }
            
        except Exception as e:
            print(f"VGGT全局姿态估计失败: {e}")
            return None
    
    def fallback_pose_estimation(self, frame1, frame2):
        """备用姿态估计方法（基于特征点匹配）"""
        try:
            # 提取特征点
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            kp1, desc1 = self.feature_detector.detectAndCompute(gray1, None)
            kp2, desc2 = self.feature_detector.detectAndCompute(gray2, None)
            
            if desc1 is None or desc2 is None:
                return None
            
            # 匹配特征点
            matches = self.matcher.match(desc1, desc2)
            if len(matches) < 8:
                return None
            
            matches = sorted(matches, key=lambda x: x.distance)[:50]
            
            # 计算角度变化
            h, w = frame1.shape[:2]
            cx, cy = w * 0.5, h * 0.5
            
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
            
            # 计算相对于中心的角度
            angles1 = np.arctan2(pts1[:, 1] - cy, pts1[:, 0] - cx)
            angles2 = np.arctan2(pts2[:, 1] - cy, pts2[:, 0] - cx)
            
            angle_diffs = angles2 - angles1
            angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
            angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
            
            yaw_change = np.median(angle_diffs) * 180 / np.pi
            
            # 简化的pitch和roll估计
            vertical_shift = np.median(pts2[:, 1] - pts1[:, 1])
            horizontal_shift = np.median(pts2[:, 0] - pts1[:, 0])
            
            pitch_change = np.arctan2(vertical_shift, h/2) * 180 / np.pi
            roll_change = np.arctan2(horizontal_shift, w/2) * 180 / np.pi
            
            return {
                'yaw': yaw_change,
                'pitch': pitch_change,
                'roll': roll_change,
                'confidence': min(1.0, len(matches) / 50.0),
                'source': 'fallback'
            }
            
        except Exception as e:
            print(f"备用姿态估计失败: {e}")
            return None
    
    def estimate_pose_sequence(self, video_path):
        """估计视频的姿态序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return []
        
        poses = []
        prev_frame = None
        cumulative_yaw = 0.0
        cumulative_pitch = 0.0
        cumulative_roll = 0.0
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in tqdm(range(frame_count), desc="VGGT姿态估计", disable=True):
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_result = None
            
            # 优先尝试人脸姿态估计
            face_pose = self.extract_face_pose_with_vggt(frame)
            if face_pose and face_pose['confidence'] > 0.5:
                pose_result = face_pose
            else:
                # 尝试全局姿态估计
                global_pose = self.extract_global_pose_with_vggt(frame)
                if global_pose and global_pose['confidence'] > 0.3:
                    pose_result = global_pose
                elif prev_frame is not None:
                    # 使用备用方法
                    fallback_pose = self.fallback_pose_estimation(prev_frame, frame)
                    if fallback_pose:
                        pose_result = fallback_pose
            
            if pose_result:
                # 累积角度变化
                if frame_idx > 0:  # 从第二帧开始累积
                    cumulative_yaw += pose_result['yaw']
                    cumulative_pitch += pose_result['pitch']
                    cumulative_roll += pose_result['roll']
                
                poses.append({
                    'frame': frame_idx,
                    'yaw': cumulative_yaw,
                    'pitch': cumulative_pitch,
                    'roll': cumulative_roll,
                    'delta_yaw': pose_result['yaw'],
                    'delta_pitch': pose_result['pitch'],
                    'delta_roll': pose_result['roll'],
                    'confidence': pose_result['confidence'],
                    'source': pose_result['source']
                })
            else:
                # 如果估计失败，使用前一帧的累积值
                poses.append({
                    'frame': frame_idx,
                    'yaw': cumulative_yaw,
                    'pitch': cumulative_pitch,
                    'roll': cumulative_roll,
                    'delta_yaw': 0.0,
                    'delta_pitch': 0.0,
                    'delta_roll': 0.0,
                    'confidence': 0.0,
                    'source': 'interpolated'
                })
            
            prev_frame = frame.copy()
        
        cap.release()
        return poses


class VGGTUniversalBulletTimeDetector:
    """基于VGGT的通用子弹时间检测器"""
    
    def __init__(self, device, submodules_dict):
        self.device = device
        self.yaw_threshold = 75.0  # yaw角度阈值
        self.consistency_threshold = 0.7  # 运动一致性阈值
        self.min_valid_frames = 15  # 最少有效帧数
        self.min_confidence = 0.3  # 最低置信度阈值
        
        # 初始化VGGT全局姿态估计器
        vggt_path = submodules_dict.get('vggt_path', 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main')
        vggt_weights = submodules_dict.get('vggt_weights', None)
        self.pose_estimator = VGGTGlobalPoseEstimator(device, vggt_path, vggt_weights)
        
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
    
    def analyze_vggt_pose_sequence(self, poses):
        """分析VGGT姿态序列"""
        if len(poses) < self.min_valid_frames:
            return False, 0.0, {'error': 'insufficient_frames', 'frame_count': len(poses)}
        
        # 过滤有效姿态
        valid_poses = [p for p in poses if p['confidence'] >= self.min_confidence]
        if len(valid_poses) < self.min_valid_frames:
            return False, 0.0, {'error': 'insufficient_valid_poses', 'valid_count': len(valid_poses)}
        
        # 提取yaw角度序列
        yaw_sequence = [p['yaw'] for p in valid_poses]
        delta_yaw_sequence = [p['delta_yaw'] for p in valid_poses]
        confidence_sequence = [p['confidence'] for p in valid_poses]
        
        # 计算总旋转角度
        total_yaw_rotation = abs(yaw_sequence[-1] - yaw_sequence[0])
        
        # 分析运动一致性
        if len(delta_yaw_sequence) <= 1:
            return False, 0.0, {'error': 'insufficient_motion_data'}
        
        # 过滤显著运动（避免噪声）
        significant_deltas = [d for d in delta_yaw_sequence if abs(d) > 1.0]
        if len(significant_deltas) == 0:
            return False, 0.0, {'error': 'no_significant_motion'}
        
        # 计算方向一致性
        positive_motion = sum(1 for d in significant_deltas if d > 0)
        negative_motion = sum(1 for d in significant_deltas if d < 0)
        direction_consistency = max(positive_motion, negative_motion) / len(significant_deltas)
        
        # 计算平均置信度
        avg_confidence = np.mean(confidence_sequence)
        
        # 计算运动平滑度
        if len(delta_yaw_sequence) > 2:
            motion_smoothness = 1.0 / (1.0 + np.std(delta_yaw_sequence))
        else:
            motion_smoothness = 0.5
        
        # 分析姿态来源分布
        source_stats = {}
        for pose in valid_poses:
            source = pose['source']
            source_stats[source] = source_stats.get(source, 0) + 1
        
        # 综合判定
        is_valid_rotation = (
            total_yaw_rotation >= self.yaw_threshold and
            direction_consistency >= self.consistency_threshold and
            avg_confidence >= self.min_confidence and
            motion_smoothness > 0.2
        )
        
        # 计算综合置信度
        rotation_score = min(1.0, total_yaw_rotation / 180.0)
        consistency_score = direction_consistency
        confidence_score = avg_confidence
        smoothness_score = motion_smoothness
        
        combined_confidence = (
            rotation_score * 0.4 +
            consistency_score * 0.3 +
            confidence_score * 0.2 +
            smoothness_score * 0.1
        )
        
        details = {
            'total_yaw_rotation': total_yaw_rotation,
            'direction_consistency': direction_consistency,
            'avg_confidence': avg_confidence,
            'motion_smoothness': motion_smoothness,
            'significant_motion_count': len(significant_deltas),
            'valid_pose_count': len(valid_poses),
            'total_frame_count': len(poses),
            'source_distribution': source_stats,
            'yaw_sequence': yaw_sequence[:100],  # 限制输出长度
            'rotation_score': rotation_score,
            'consistency_score': consistency_score,
            'confidence_score': confidence_score,
            'smoothness_score': smoothness_score
        }
        
        return is_valid_rotation, combined_confidence, details
    
    def detect_orbit_motion(self, video_path):
        """检测环绕运镜（基于CoTracker）"""
        if self.cotracker is None:
            return False, 0.0, {'error': 'cotracker_not_available'}
        
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
            
            for t in range(tracks.shape[0] - 1):
                curr_tracks = tracks[t]
                next_tracks = tracks[t + 1]
                vis_mask = (visibility[t] > 0.5) & (visibility[t + 1] > 0.5)
                
                if vis_mask.sum() < 8:
                    continue
                
                valid_curr = curr_tracks[vis_mask]
                valid_next = next_tracks[vis_mask]
                
                # 计算角度变化
                angles_curr = np.arctan2(valid_curr[:, 1] - cy, valid_curr[:, 0] - cx)
                angles_next = np.arctan2(valid_next[:, 1] - cy, valid_next[:, 0] - cx)
                
                angle_diffs = angles_next - angles_curr
                angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
                angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
                
                if len(angle_diffs) > 0:
                    circular_scores.append(np.mean(angle_diffs))
            
            if not circular_scores:
                return False, 0.0, {'error': 'no_motion_data'}
            
            circular_scores = np.array(circular_scores)
            
            # 分析环形运动特征
            positive_count = (circular_scores > 0).sum()
            negative_count = (circular_scores < 0).sum()
            total_count = len(circular_scores)
            
            direction_consistency = max(positive_count, negative_count) / total_count
            total_rotation_rad = np.sum(np.abs(circular_scores))
            total_rotation_deg = total_rotation_rad * 180 / np.pi
            
            is_orbit = (
                direction_consistency > self.consistency_threshold and 
                total_rotation_deg > 30  # 至少30度的环形运动
            )
            
            confidence = min(1.0, direction_consistency * (total_rotation_deg / 90))
            
            details = {
                'direction_consistency': direction_consistency,
                'total_rotation_deg': total_rotation_deg,
                'motion_frames': len(circular_scores),
                'total_frames': tracks.shape[0]
            }
            
            return is_orbit, confidence, details
            
        except Exception as e:
            print(f"环绕运镜检测失败: {e}")
            return False, 0.0, {'error': str(e)}
    
    def detect_vggt_bullet_time(self, video_path):
        """基于VGGT的通用子弹时间检测"""
        print(f"开始VGGT子弹时间检测: {os.path.basename(video_path)}")
        
        # 1. VGGT姿态估计
        print("  步骤1: VGGT姿态序列估计...")
        poses = self.pose_estimator.estimate_pose_sequence(video_path)
        
        if not poses:
            return {
                'is_bullet_time': False,
                'confidence': 0.0,
                'error': 'vggt_pose_estimation_failed',
                'vggt_pose': {'detected': False},
                'orbit_motion': {'detected': False}
            }
        
        # 2. 分析VGGT姿态序列
        print("  步骤2: 分析VGGT yaw角度序列...")
        has_yaw_rotation, yaw_confidence, yaw_details = self.analyze_vggt_pose_sequence(poses)
        
        # 3. CoTracker环绕运动检测
        print("  步骤3: CoTracker环绕运动检测...")
        has_orbit, orbit_confidence, orbit_details = self.detect_orbit_motion(video_path)
        
        # 4. 综合判断
        print("  步骤4: 综合判断子弹时间...")
        is_bullet_time = has_yaw_rotation and has_orbit
        
        # 计算综合置信度
        if is_bullet_time:
            combined_confidence = yaw_confidence * 0.75 + orbit_confidence * 0.25  # VGGT权重更高
        else:
            combined_confidence = 0.0
        
        # 构建详细结果
        result = {
            'is_bullet_time': is_bullet_time,
            'confidence': combined_confidence,
            'method': 'vggt_universal',
            'vggt_pose': {
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
                'total_yaw_rotation': yaw_details.get('total_yaw_rotation', 0),
                'motion_consistency': yaw_details.get('direction_consistency', 0),
                'avg_vggt_confidence': yaw_details.get('avg_confidence', 0),
                'orbit_rotation': orbit_details.get('total_rotation_deg', 0),
                'detection_quality': min(yaw_confidence, orbit_confidence),
                'pose_source_distribution': yaw_details.get('source_distribution', {})
            }
        }
        
        status = "✓ 子弹时间" if is_bullet_time else "✗ 非子弹时间"
        print(f"  检测完成: {status} (置信度: {combined_confidence:.3f})")
        return result


def compute_vggt_bullet_time(json_dir, device, submodules_dict, **kwargs):
    """VBench2 VGGT子弹时间检测评估函数"""
    detector = VGGTUniversalBulletTimeDetector(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='bullet_time', lang='en')
    
    video_results = []
    scores = []
    
    for prompt_dict in tqdm(prompt_dict_ls, desc="VGGT子弹时间检测评估"):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            try:
                result = detector.detect_vggt_bullet_time(video_path)
                score = 1.0 if result['is_bullet_time'] else 0.0
                
                video_results.append({
                    'video_path': video_path, 
                    'video_results': score,
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
    return avg_score, video_results


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    submodules_dict = {
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8',
        'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
        'vggt_weights': 'path/to/vggt/weights.pth'  # 可选
    }
    
    detector = VGGTUniversalBulletTimeDetector(device, submodules_dict)
    
    # 测试视频
    test_videos = [
        "path/to/portrait_bullet_time.mp4",
        "path/to/architecture_bullet_time.mp4", 
        "path/to/landscape_bullet_time.mp4"
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n测试视频: {video_path}")
            result = detector.detect_vggt_bullet_time(video_path)
            print(f"检测结果: {result['is_bullet_time']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"VGGT yaw旋转: {result['summary']['total_yaw_rotation']:.1f}°")
            print(f"姿态来源分布: {result['summary']['pose_source_distribution']}")
        else:
            print(f"视频文件不存在: {video_path}")
