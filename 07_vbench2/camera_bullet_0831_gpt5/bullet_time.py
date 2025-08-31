"""
子弹时间运镜检测模块
基于VGGT头部姿态估计和CoTracker轨迹分析的综合检测方法
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
import torchvision.transforms as transforms
from PIL import Image
from vbench2.utils import load_dimension_info, split_video_into_scenes

# 可选的人脸检测依赖
try:
    import dlib
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False

class VGGTHeadPoseEstimator:
    """基于VGGT的头部姿态估计器"""
    
    def __init__(self, device, vggt_path, model_weights=None):
        self.device = device
        self.model = None
        self.face_detector = None
        self._init_vggt_model(vggt_path, model_weights)
        self._init_face_detector()
    
    def _init_vggt_model(self, vggt_path, model_weights):
        """初始化VGGT模型"""
        try:
            # 添加VGGT路径到系统路径
            if vggt_path not in sys.path:
                sys.path.append(vggt_path)
            
            # 导入VGGT模型
            from models.vggt import VGGT
            
            # 创建模型实例
            self.model = VGGT(num_classes=3).to(self.device)  # yaw, pitch, roll
            
            # 加载预训练权重
            if model_weights and os.path.exists(model_weights):
                checkpoint = torch.load(model_weights, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                print("✓ VGGT模型权重加载成功")
            
            self.model.eval()
            print("✓ VGGT头部姿态估计模型初始化成功")
            
        except Exception as e:
            print(f"VGGT模型初始化失败: {e}")
            self.model = None
    
    def _init_face_detector(self):
        """初始化人脸检测器"""
        if _HAS_DLIB:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                print("✓ Dlib人脸检测器初始化成功")
            except Exception as e:
                print(f"Dlib初始化失败: {e}")
                self.face_detector = None
        else:
            print("警告: dlib未安装，将使用整图进行姿态估计")
    
    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(face_img).unsqueeze(0).to(self.device)
    
    def detect_largest_face(self, frame):
        """检测最大的人脸区域"""
        if self.face_detector is None:
            return frame  # 返回整个帧
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        
        if len(faces) == 0:
            return None
        
        # 选择最大的人脸
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        x1, y1, x2, y2 = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
        
        # 扩展边界框
        margin = 20
        h, w = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        face_img = frame[y1:y2, x1:x2]
        return face_img if face_img.size > 0 else None
    
    def estimate_pose(self, frame):
        """估计单帧的头部姿态"""
        if self.model is None:
            return None
        
        # 检测人脸
        face_img = self.detect_largest_face(frame)
        if face_img is None:
            return None
        
        try:
            # 预处理
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess_face(face_rgb)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                if isinstance(outputs, tuple):
                    yaw, pitch, roll = outputs
                else:
                    # 假设输出是[batch, 3]的格式
                    yaw, pitch, roll = outputs[0, 0], outputs[0, 1], outputs[0, 2]
                
                # 转换为角度
                yaw = float(yaw.cpu().numpy()) if hasattr(yaw, 'cpu') else float(yaw)
                pitch = float(pitch.cpu().numpy()) if hasattr(pitch, 'cpu') else float(pitch)
                roll = float(roll.cpu().numpy()) if hasattr(roll, 'cpu') else float(roll)
                
                return (yaw, pitch, roll)
                
        except Exception as e:
            print(f"姿态估计失败: {e}")
            return None
    
    def extract_video_poses(self, video_path):
        """提取视频中每帧的头部姿态"""
        poses = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return poses
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(total_frames), desc="提取头部姿态", disable=True):
            ret, frame = cap.read()
            if not ret:
                break
            
            pose = self.estimate_pose(frame)
            poses.append(pose)
        
        cap.release()
        return poses


class BulletTimeDetector:
    """子弹时间运镜检测器"""
    
    def __init__(self, device, submodules_dict):
        self.device = device
        self.yaw_threshold = 75.0  # yaw角度阈值
        self.consistency_threshold = 0.7  # 运动一致性阈值
        self.min_valid_frames = 10  # 最少有效帧数
        
        # 初始化VGGT头部姿态估计器
        vggt_path = submodules_dict.get('vggt_path', 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main')
        model_weights = submodules_dict.get('vggt_weights', None)
        self.pose_estimator = VGGTHeadPoseEstimator(device, vggt_path, model_weights)
        
        # 初始化CoTracker模型（用于环绕运镜检测）
        self._init_cotracker(submodules_dict)
    
    def _init_cotracker(self, submodules_dict):
        """初始化CoTracker模型"""
        try:
            self.cotracker = torch.hub.load(
                submodules_dict.get("repo", "facebookresearch/co-tracker"), 
                submodules_dict.get("model", "cotracker_stride_4_wind_8")
            ).to(self.device)
            print("✓ CoTracker模型初始化成功")
        except Exception as e:
            print(f"CoTracker初始化失败: {e}")
            self.cotracker = None
    
    def detect_orbit_motion(self, video_path):
        """检测环绕运镜（基于CoTracker轨迹分析）"""
        if self.cotracker is None:
            return False, 0.0
        
        try:
            # 读取视频
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader)))
            video = video.permute(0, 3, 1, 2)[None].float().to(self.device)  # B T C H W
            
            # 获取轨迹
            with torch.no_grad():
                pred_tracks, pred_visibility = self.cotracker(video, grid_size=10)
            
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
                
                if vis_mask.sum() < 5:
                    continue
                
                valid_curr = curr_tracks[vis_mask]
                valid_next = next_tracks[vis_mask]
                
                # 计算角度变化
                angles_curr = np.arctan2(valid_curr[:, 1] - cy, valid_curr[:, 0] - cx)
                angles_next = np.arctan2(valid_next[:, 1] - cy, valid_next[:, 0] - cx)
                
                # 处理角度跨越
                angle_diffs = angles_next - angles_curr
                angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
                angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
                
                if len(angle_diffs) > 0:
                    mean_angle_change = np.mean(angle_diffs)
                    circular_scores.append(mean_angle_change)
            
            if not circular_scores:
                return False, 0.0
            
            # 计算环形运动一致性
            circular_scores = np.array(circular_scores)
            positive_count = (circular_scores > 0).sum()
            negative_count = (circular_scores < 0).sum()
            total_count = len(circular_scores)
            
            consistency = max(positive_count, negative_count) / total_count
            total_rotation = abs(np.sum(circular_scores)) * 180 / np.pi
            
            is_orbit = consistency > self.consistency_threshold and total_rotation > 30
            confidence = min(1.0, consistency * (total_rotation / 90))
            
            return is_orbit, confidence
            
        except Exception as e:
            print(f"环绕运镜检测失败: {e}")
            return False, 0.0
    
    def analyze_yaw_rotation(self, poses):
        """分析yaw角度旋转"""
        valid_poses = [pose for pose in poses if pose is not None]
        
        if len(valid_poses) < self.min_valid_frames:
            return False, 0.0, {}
        
        yaw_angles = [pose[0] for pose in valid_poses]
        
        # 角度序列平滑处理
        yaw_angles = np.array(yaw_angles)
        
        # 处理角度跳跃（-180到180的跳跃）
        yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw_angles)))
        
        # 计算总旋转角度
        total_rotation = abs(yaw_unwrapped[-1] - yaw_unwrapped[0])
        
        # 计算运动一致性
        diffs = np.diff(yaw_unwrapped)
        if len(diffs) == 0:
            return False, 0.0, {}
        
        # 计算主要旋转方向的一致性
        positive_diffs = (diffs > 0).sum()
        negative_diffs = (diffs < 0).sum()
        direction_consistency = max(positive_diffs, negative_diffs) / len(diffs)
        
        # 计算平均旋转速度
        avg_rotation_speed = total_rotation / len(yaw_angles)
        
        # 判断是否为有效的yaw旋转
        is_valid_yaw = (
            total_rotation >= self.yaw_threshold and 
            direction_consistency >= self.consistency_threshold
        )
        
        # 计算置信度
        confidence = min(1.0, (total_rotation / 180.0) * direction_consistency)
        
        details = {
            'total_rotation': total_rotation,
            'direction_consistency': direction_consistency,
            'avg_rotation_speed': avg_rotation_speed,
            'valid_frames': len(valid_poses),
            'total_frames': len(poses),
            'yaw_sequence': yaw_angles.tolist()
        }
        
        return is_valid_yaw, confidence, details
    
    def detect_bullet_time(self, video_path):
        """综合检测子弹时间运镜"""
        # 1. 提取头部姿态
        poses = self.pose_estimator.extract_video_poses(video_path)
        
        # 2. 分析yaw旋转
        has_yaw_rotation, yaw_confidence, yaw_details = self.analyze_yaw_rotation(poses)
        
        # 3. 检测环绕运镜
        has_orbit, orbit_confidence = self.detect_orbit_motion(video_path)
        
        # 4. 综合判断
        is_bullet_time = has_yaw_rotation and has_orbit
        
        # 5. 计算综合置信度
        if is_bullet_time:
            combined_confidence = (yaw_confidence * 0.7 + orbit_confidence * 0.3)
        else:
            combined_confidence = 0.0
        
        result = {
            'is_bullet_time': is_bullet_time,
            'confidence': combined_confidence,
            'yaw_rotation': {
                'detected': has_yaw_rotation,
                'confidence': yaw_confidence,
                'details': yaw_details
            },
            'orbit_motion': {
                'detected': has_orbit,
                'confidence': orbit_confidence
            }
        }
        
        return result


def compute_bullet_time(json_dir, device, submodules_dict, **kwargs):
    """VBench2子弹时间检测评估函数"""
    detector = BulletTimeDetector(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='bullet_time', lang='en')
    
    video_results = []
    scores = []
    
    for prompt_dict in tqdm(prompt_dict_ls, desc="评估子弹时间运镜"):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            try:
                result = detector.detect_bullet_time(video_path)
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
        'vggt_path': 'VBench-2.0/vbench2/third_party/vggt-main/vggt-main',
        'vggt_weights': 'path/to/vggt/weights.pth',
        'repo': 'facebookresearch/co-tracker',
        'model': 'cotracker_stride_4_wind_8'
    }
    
    detector = BulletTimeDetector(device, submodules_dict)
    
    # 测试单个视频
    test_video = "path/to/test/video.mp4"
    if os.path.exists(test_video):
        result = detector.detect_bullet_time(test_video)
        print(f"子弹时间检测结果: {result}")
