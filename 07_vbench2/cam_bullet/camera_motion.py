import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from math import ceil
from tqdm import tqdm
from .third_party.cotracker.utils.visualizer import Visualizer
import json
import os
from vbench2.utils import load_dimension_info, split_video_into_scenes
from tqdm import tqdm
import torch.nn.functional as F

# 可选的人像分割依赖（若不可用则自动降级为简易中心掩码）
try:
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# 可选的人脸姿态估计依赖（用于子弹时间检测）
try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False

try:
    import dlib
    _HAS_DLIB = True
except Exception:
    _HAS_DLIB = False


def transform(vector):
    x = np.mean([item[0] for item in vector])
    y = np.mean([item[1] for item in vector])
    return [x, y]

def transform_class(vector, min_reso, factor=0.005): # 768*0.05
    scale = min_reso * factor
    x, y = vector
    direction = []
    if x > scale:
        direction.append("right")
    elif x < -scale:
        direction.append("left")
    if y > scale:
        direction.append("down")
    elif y < -scale:
        direction.append("up")
    return direction if direction else ["static"]

def transform_class360(vector, min_reso, factor=0.008): # 768*0.05
    scale = min_reso * factor
    up, down, y = vector
    if abs(y)<scale:
        if up * down<0 and up>scale:
            return "orbits"  #orbits_counterclockwise
        elif up*down<0 and up<-scale:
            return "orbits"   #orbits_clockwise
        else:
            return None

class CameraPredict:
    def __init__(self, device, submodules_list):
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        try:
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        except:
            # workaround for CERTIFICATE_VERIFY_FAILED (see: https://github.com/pytorch/pytorch/issues/33288#issuecomment-954160699)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)

        # 分割与Dolly检测相关参数
        self.seg_processor = None
        self.seg_model = None
        self._init_segmentation_model()
        self.dolly_proj_threshold = 0.01  # 相对尺度阈值（相对于min(H,W)）
        self.dolly_consistency_threshold = 0.65
        
        # 子弹时间检测相关参数
        self.face_mesh = None
        self.face_detector = None
        self._init_face_pose_estimation()
        self.bullet_time_angle_threshold = 15.0  # 角度变化阈值（度）
        self.bullet_time_consistency_threshold = 0.7  # 子弹时间一致性阈值

    def transform360(self, vector):
        up=[]
        down=[]
        for item in vector:
            if item[2]>self.scale/2:
                down.append(item[0])
            else:
                up.append(item[0])
        y = np.mean([item[1] for item in vector])
        if len(up)>0:
            mean_up=sum(up)/len(up)
        else:
            mean_up=0
        if len(down)>0:
            mean_down=sum(down)/len(down)
        else:
           mean_down=0
        return [mean_up, mean_down, y]

    def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos"):
        b,_,_,h,w=video.shape
        self.scale=min(h,w)
        self.height=h
        self.width=w
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        if save_video:
            vis = Visualizer(save_dir=save_dir, pad_value=120, fps=fps, linewidth=3)
            vis.visualize(video, pred_tracks, pred_visibility, filename="temp1")
            raise
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        return pred_tracks[0].long().detach().cpu().numpy()

    def infer_with_visibility(self, video, fps=16, end_frame=-1):
        """返回轨迹与可见性（numpy）。不影响原有infer用法。"""
        b,_,_,h,w=video.shape
        self.scale=min(h,w)
        self.height=h
        self.width=w
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        tracks = pred_tracks[0].detach().cpu().numpy()  # T N 2（浮点）
        vis = pred_visibility[0].detach().cpu().numpy().squeeze(-1)  # T N
        return tracks, vis

    def _init_segmentation_model(self):
        if not _HAS_TRANSFORMERS:
            return
        try:
            self.seg_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512").to(self.device)
            self.seg_model.eval()
        except Exception:
            self.seg_processor = None
            self.seg_model = None
    
    def _init_face_pose_estimation(self):
        """初始化人脸姿态估计模型"""
        if _HAS_MEDIAPIPE:
            try:
                # 使用MediaPipe Face Mesh进行人脸姿态估计
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                print("✓ MediaPipe人脸姿态估计初始化成功")
            except Exception as e:
                print(f"MediaPipe初始化失败: {e}")
                self.face_mesh = None
        
        # 备用方案：如果有dlib可以使用更简单的人脸检测
        if _HAS_DLIB and self.face_mesh is None:
            try:
                self.face_detector = dlib.get_frontal_face_detector()
                print("✓ Dlib人脸检测器初始化成功")
            except Exception as e:
                print(f"Dlib初始化失败: {e}")
                self.face_detector = None

    def _default_center_mask(self, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        ch, cw = h // 2, w // 2
        mask[max(0, ch - h//4):min(h, ch + h//4), max(0, cw - w//4):min(w, cw + w//4)] = 1
        return mask

    def segment_human(self, frame):
        """对单帧进行人像分割，返回0/1掩码。"""
        h, w = frame.shape[:2]
        if self.seg_model is None or self.seg_processor is None:
            return self._default_center_mask(h, w)
        try:
            inputs = self.seg_processor(images=frame, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            logits = outputs.logits
            up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1).squeeze(0).detach().cpu().numpy()
            # ADE20K中人的label一般为12（经验值），如需精细可映射label集
            mask = (pred == 12).astype(np.uint8)
            if mask.sum() < 50:
                return self._default_center_mask(h, w)
            return mask
        except Exception:
            return self._default_center_mask(h, w)

    def segment_video_masks(self, video):
        """为视频每帧生成前景人像掩码列表。video: B T C H W (float, 0-?)"""
        b, t, c, h, w = video.shape
        frames = video[0].permute(0,2,3,1).detach().cpu().numpy()  # T H W C
        # 归一转uint8（若已是0-255则clip即可）
        if frames.max() <= 1.5:
            frames = (frames * 255.0).clip(0,255).astype(np.uint8)
        else:
            frames = frames.clip(0,255).astype(np.uint8)
        masks = []
        for i in range(t):
            masks.append(self.segment_human(frames[i]))
        return masks
    
    def estimate_face_pose(self, frame):
        """
        估计单帧图像中的人脸姿态角度
        返回: (yaw, pitch, roll) 角度（度），如果检测失败返回None
        """
        if self.face_mesh is None:
            return None
        
        try:
            # 转换为RGB格式（MediaPipe需要RGB）
            if frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # 提取关键点（鼻尖、左右眼角、嘴角等）
            # MediaPipe Face Mesh的关键点索引
            nose_tip = face_landmarks.landmark[1]  # 鼻尖
            chin = face_landmarks.landmark[175]    # 下巴
            left_eye = face_landmarks.landmark[33] # 左眼角
            right_eye = face_landmarks.landmark[263] # 右眼角
            left_mouth = face_landmarks.landmark[61]  # 左嘴角
            right_mouth = face_landmarks.landmark[291] # 右嘴角
            
            # 转换为像素坐标
            nose_tip_px = (int(nose_tip.x * w), int(nose_tip.y * h))
            chin_px = (int(chin.x * w), int(chin.y * h))
            left_eye_px = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_px = (int(right_eye.x * w), int(right_eye.y * h))
            left_mouth_px = (int(left_mouth.x * w), int(left_mouth.y * h))
            right_mouth_px = (int(right_mouth.x * w), int(right_mouth.y * h))
            
            # 计算姿态角度
            # Yaw (左右转头): 基于左右眼/嘴角的水平偏移
            eye_center_x = (left_eye_px[0] + right_eye_px[0]) / 2
            mouth_center_x = (left_mouth_px[0] + right_mouth_px[0]) / 2
            face_center_x = (eye_center_x + mouth_center_x) / 2
            yaw_offset = (face_center_x - w/2) / (w/2)  # 归一化到[-1,1]
            yaw = yaw_offset * 45  # 转换为角度（最大±45度）
            
            # Pitch (上下点头): 基于鼻尖和下巴的垂直关系
            nose_chin_y = nose_tip_px[1] - chin_px[1]
            expected_nose_chin_y = h * 0.15  # 期望的鼻尖-下巴距离
            pitch_ratio = nose_chin_y / expected_nose_chin_y if expected_nose_chin_y > 0 else 0
            pitch = (1 - pitch_ratio) * 30  # 转换为角度
            
            # Roll (侧倾): 基于左右眼的水平线倾斜
            eye_slope = (right_eye_px[1] - left_eye_px[1]) / max(1, abs(right_eye_px[0] - left_eye_px[0]))
            roll = np.arctan(eye_slope) * 180 / np.pi  # 转换为角度
            
            return (float(yaw), float(pitch), float(roll))
            
        except Exception as e:
            # print(f"人脸姿态估计失败: {e}")
            return None
    
    def extract_video_face_poses(self, video):
        """
        提取视频中每帧的人脸姿态角度
        video: B T C H W (float, 0-?)
        返回: 姿态角度列表 [(yaw, pitch, roll), ...]
        """
        b, t, c, h, w = video.shape
        frames = video[0].permute(0,2,3,1).detach().cpu().numpy()  # T H W C
        
        # 归一转uint8
        if frames.max() <= 1.5:
            frames = (frames * 255.0).clip(0,255).astype(np.uint8)
        else:
            frames = frames.clip(0,255).astype(np.uint8)
        
        poses = []
        for i in range(t):
            pose = self.estimate_face_pose(frames[i])
            poses.append(pose)
        
        return poses
    
    def detect_bullet_time_motion(self, tracks, visibility, face_poses, h, w):
        """
        检测子弹时间运镜效果
        tracks: T N 2 (numpy, 像素坐标)
        visibility: T N (numpy, [0-1])
        face_poses: [(yaw, pitch, roll), ...] 人脸姿态角度序列
        返回: (bullet_time_type, confidence, details)
        """
        if tracks.shape[0] < 10 or len(face_poses) < 10:  # 子弹时间需要足够的帧数
            return "no_bullet_time", 0.0, {}
        
        # 过滤有效的人脸姿态数据
        valid_poses = [pose for pose in face_poses if pose is not None]
        if len(valid_poses) < 5:
            return "no_bullet_time", 0.0, {}
        
        # 分析人脸姿态角度变化
        yaw_angles = [pose[0] for pose in valid_poses]
        pitch_angles = [pose[1] for pose in valid_poses]
        roll_angles = [pose[2] for pose in valid_poses]
        
        # 计算角度变化范围和趋势
        yaw_range = max(yaw_angles) - min(yaw_angles)
        pitch_range = max(pitch_angles) - min(pitch_angles)
        roll_range = max(roll_angles) - min(roll_angles)
        
        # 计算角度变化的总体趋势（线性拟合）
        frame_indices = list(range(len(valid_poses)))
        
        try:
            yaw_trend = np.polyfit(frame_indices, yaw_angles, 1)[0] if len(yaw_angles) > 1 else 0
            pitch_trend = np.polyfit(frame_indices, pitch_angles, 1)[0] if len(pitch_angles) > 1 else 0
            roll_trend = np.polyfit(frame_indices, roll_angles, 1)[0] if len(roll_angles) > 1 else 0
        except:
            yaw_trend = pitch_trend = roll_trend = 0
        
        # 分析CoTracker特征点的环形运动模式
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
            
            # 计算点到中心的角度变化（用于检测环形运动）
            angles_curr = np.arctan2(valid_curr[:, 1] - cy, valid_curr[:, 0] - cx)
            angles_next = np.arctan2(valid_next[:, 1] - cy, valid_next[:, 0] - cx)
            
            # 计算角度差，处理跨越边界的情况
            angle_diffs = angles_next - angles_curr
            angle_diffs = np.where(angle_diffs > np.pi, angle_diffs - 2*np.pi, angle_diffs)
            angle_diffs = np.where(angle_diffs < -np.pi, angle_diffs + 2*np.pi, angle_diffs)
            
            # 计算平均角度变化
            if len(angle_diffs) > 0:
                mean_angle_change = np.mean(angle_diffs)
                circular_scores.append(mean_angle_change)
        
        # 分析结果
        details = {
            'yaw_range': float(yaw_range),
            'pitch_range': float(pitch_range),
            'roll_range': float(roll_range),
            'yaw_trend': float(yaw_trend),
            'pitch_trend': float(pitch_trend),
            'roll_trend': float(roll_trend),
            'circular_scores': circular_scores,
            'num_valid_poses': len(valid_poses),
            'face_pose_ratio': len(valid_poses) / len(face_poses)
        }
        
        if not circular_scores:
            return "no_bullet_time", 0.0, details
        
        # 计算环形运动的一致性
        circular_scores = np.array(circular_scores)
        mean_circular_motion = np.mean(circular_scores)
        circular_consistency = 1.0 - np.std(circular_scores) / (abs(mean_circular_motion) + 0.1)
        circular_consistency = max(0.0, min(1.0, circular_consistency))
        
        details.update({
            'mean_circular_motion': float(mean_circular_motion),
            'circular_consistency': float(circular_consistency),
            'total_angle_change': float(abs(mean_circular_motion) * len(circular_scores))
        })
        
        # 判断子弹时间类型
        # 主要基于yaw角度变化（左右转头）和环形运动的一致性
        primary_angle_change = abs(yaw_range)
        secondary_angle_change = max(abs(pitch_range), abs(roll_range))
        
        # 总角度变化（考虑环形运动）
        total_rotation = abs(mean_circular_motion) * len(circular_scores) * 180 / np.pi
        
        # 角度变化阈值
        angle_90 = 90 - self.bullet_time_angle_threshold
        angle_180 = 180 - self.bullet_time_angle_threshold
        angle_270 = 270 - self.bullet_time_angle_threshold
        angle_360 = 360 - self.bullet_time_angle_threshold
        
        # 综合置信度：角度变化幅度 + 环形运动一致性 + 人脸检测比例
        angle_confidence = min(1.0, primary_angle_change / 45.0)  # yaw角度置信度
        motion_confidence = circular_consistency  # 运动一致性
        pose_confidence = details['face_pose_ratio']  # 人脸检测成功率
        
        overall_confidence = (angle_confidence * 0.4 + motion_confidence * 0.4 + pose_confidence * 0.2)
        
        if overall_confidence < self.bullet_time_consistency_threshold:
            return "no_bullet_time", float(overall_confidence), details
        
        # 根据角度变化判断子弹时间类型
        if primary_angle_change >= angle_90 or total_rotation >= angle_90:
            if primary_angle_change >= angle_360 or total_rotation >= angle_360:
                bullet_type = "bullet_time_360"
            elif primary_angle_change >= angle_270 or total_rotation >= angle_270:
                bullet_type = "bullet_time_270"
            elif primary_angle_change >= angle_180 or total_rotation >= angle_180:
                bullet_type = "bullet_time_180"
            else:
                bullet_type = "bullet_time_90"
            
            return bullet_type, float(overall_confidence), details
        else:
            return "no_bullet_time", float(overall_confidence), details

    def detect_dolly_motion(self, tracks, visibility, h, w, masks=None):
        """
        使用前景点的径向投影和平均距离变化来判断dolly_in/out和zoom_in/out。
        tracks: T N 2 (numpy, 像素坐标)
        visibility: T N (numpy, [0-1])
        masks: 前景掩码列表，用于筛选前景区域特征点
        返回: (label, confidence, details)
        """
        if tracks.shape[0] < 2:
            return "no_dolly", 0.0, {}
        
        cx, cy = w * 0.5, h * 0.5
        scale = float(min(h, w))
        
        # 收集前景区域的特征点
        foreground_tracks = []
        foreground_visibility = []
        
        for t in range(tracks.shape[0]):
            curr_tracks = tracks[t]  # N 2
            curr_vis = visibility[t]  # N
            
            if masks is not None and t < len(masks):
                # 基于掩码筛选前景特征点
                mask = masks[t]
                fg_indices = []
                for i, (track, vis) in enumerate(zip(curr_tracks, curr_vis)):
                    if vis > 0.5:
                        x, y = int(track[0]), int(track[1])
                        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
                            fg_indices.append(i)
                
                if len(fg_indices) >= 3:  # 至少需要3个前景点
                    fg_tracks = curr_tracks[fg_indices]
                    fg_vis = curr_vis[fg_indices]
                else:
                    # 如果前景点太少，使用中心区域的点
                    center_mask = ((curr_tracks[:, 0] - cx)**2 + (curr_tracks[:, 1] - cy)**2) < (scale * 0.3)**2
                    valid_mask = (curr_vis > 0.5) & center_mask
                    if valid_mask.sum() >= 3:
                        fg_tracks = curr_tracks[valid_mask]
                        fg_vis = curr_vis[valid_mask]
                    else:
                        fg_tracks = curr_tracks[curr_vis > 0.5][:10]  # 取前10个可见点
                        fg_vis = curr_vis[curr_vis > 0.5][:10]
            else:
                # 没有掩码时，使用所有可见的特征点
                valid_mask = curr_vis > 0.5
                fg_tracks = curr_tracks[valid_mask]
                fg_vis = curr_vis[valid_mask]
            
            foreground_tracks.append(fg_tracks)
            foreground_visibility.append(fg_vis)
        
        # 计算平均距离变化趋势
        avg_distances = []
        proj_series = []
        
        for t in range(len(foreground_tracks)):
            if len(foreground_tracks[t]) >= 3:
                # 计算到中心的平均距离
                distances = np.linalg.norm(foreground_tracks[t] - np.array([cx, cy]), axis=1)
                avg_distances.append(distances.mean() / scale)  # 归一化到[0,1]
        
        # 分析相邻帧间的径向运动
        for t in range(len(foreground_tracks)-1):
            curr_tracks = foreground_tracks[t]
            next_tracks = foreground_tracks[t+1]
            
            if len(curr_tracks) < 3 or len(next_tracks) < 3:
                continue
                
            # 找到最近邻对应（简化版本：假设CoTracker已经处理了对应关系）
            min_len = min(len(curr_tracks), len(next_tracks))
            c = curr_tracks[:min_len]
            n = next_tracks[:min_len]
            
            # 计算径向投影
            radial = c - np.array([cx, cy], dtype=c.dtype)
            norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-6
            radial_dir = radial / norms
            motion = n - c
            radial_proj = (motion * radial_dir).sum(axis=1) / scale
            
            if radial_proj.size > 0:
                proj_series.append(radial_proj.mean())
        
        # 分析结果
        details = {
            'avg_distances': avg_distances,
            'proj_series': proj_series,
            'num_foreground_points': [len(tracks) for tracks in foreground_tracks]
        }
        
        if len(avg_distances) < 2 or len(proj_series) < 1:
            return "no_dolly", 0.0, details
        
        # 距离变化趋势分析
        avg_distances = np.array(avg_distances)
        distance_trend = np.polyfit(range(len(avg_distances)), avg_distances, 1)[0]  # 线性趋势
        distance_change_ratio = (avg_distances[-1] - avg_distances[0]) / (avg_distances[0] + 1e-6)
        
        # 径向投影分析
        proj_series = np.array(proj_series)
        mean_proj = proj_series.mean()
        
        # 运动一致性
        if mean_proj >= 0:
            proj_consistency = (proj_series >= 0).mean()
        else:
            proj_consistency = (proj_series < 0).mean()
        
        # 距离变化一致性
        distance_consistency = 1.0 - abs(distance_trend) * 2  # 距离变化越线性，一致性越高
        distance_consistency = max(0.0, min(1.0, distance_consistency))
        
        # 综合一致性
        overall_consistency = (proj_consistency + distance_consistency) / 2
        
        details.update({
            'distance_trend': float(distance_trend),
            'distance_change_ratio': float(distance_change_ratio),
            'mean_proj': float(mean_proj),
            'proj_consistency': float(proj_consistency),
            'distance_consistency': float(distance_consistency),
            'overall_consistency': float(overall_consistency)
        })
        
        if overall_consistency < self.dolly_consistency_threshold:
            return "no_dolly", float(overall_consistency), details
        
        # 判断运镜类型
        th_proj = self.dolly_proj_threshold
        th_dist = 0.05  # 距离变化阈值
        
        # 优先考虑距离变化趋势（zoom的主要特征）
        if abs(distance_change_ratio) > th_dist:
            if distance_change_ratio > 0:  # 距离增加
                if distance_trend > 0:  # 趋势向上
                    motion_type = "zoom_out"
                    confidence = min(1.0, abs(distance_change_ratio) * 10) * overall_consistency
                else:
                    motion_type = "dolly_out"  # 可能是dolly out导致的距离增加
                    confidence = min(1.0, abs(mean_proj) * 5) * overall_consistency
            else:  # 距离减少
                if distance_trend < 0:  # 趋势向下
                    motion_type = "zoom_in"
                    confidence = min(1.0, abs(distance_change_ratio) * 10) * overall_consistency
                else:
                    motion_type = "dolly_in"  # 可能是dolly in导致的距离减少
                    confidence = min(1.0, abs(mean_proj) * 5) * overall_consistency
        else:
            # 距离变化不明显时，主要基于径向投影判断dolly
            if mean_proj > th_proj:
                motion_type = "dolly_out"
                confidence = min(1.0, abs(mean_proj) * 5) * overall_consistency
            elif mean_proj < -th_proj:
                motion_type = "dolly_in"
                confidence = min(1.0, abs(mean_proj) * 5) * overall_consistency
            else:
                motion_type = "no_dolly"
                confidence = float(overall_consistency)
        
        return motion_type, float(confidence), details

    def predict_with_segmentation(self, video, fps=16, end_frame=-1):
        """基于人像分割的前景点筛选 + zoom/dolly/子弹时间检测，并返回标准相机运动结果。"""
        # 1) 生成人像掩码
        masks = self.segment_video_masks(video)
        h = masks[0].shape[0]
        w = masks[0].shape[1]
        
        # 2) 获取轨迹与可见性
        tracks, vis = self.infer_with_visibility(video, fps=fps, end_frame=end_frame)  # T N 2 / T N
        
        # 3) 提取人脸姿态角度序列
        face_poses = self.extract_video_face_poses(video)
        
        # 4) 基于前景区域的特征点进行zoom/dolly检测
        motion_type, motion_conf, motion_details = self.detect_dolly_motion(tracks, vis, h, w, masks)
        
        # 5) 子弹时间检测
        bullet_type, bullet_conf, bullet_details = self.detect_bullet_time_motion(tracks, vis, face_poses, h, w)
        
        # 6) 原有的标准分类
        pred_track = tracks.astype(np.int64)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks_grid=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        standard = self.camera_classify(track1, track2, tracks_grid)
        
        # 7) 增强标准运动结果
        enhanced_standard = standard.copy()
        
        # 加入检测到的zoom运镜
        if motion_type in ["zoom_in", "zoom_out"] and motion_conf > 0.6:
            if motion_type not in enhanced_standard:
                enhanced_standard.append(motion_type)
        
        # 加入检测到的子弹时间运镜
        if bullet_type != "no_bullet_time" and bullet_conf > 0.7:
            if bullet_type not in enhanced_standard:
                enhanced_standard.append(bullet_type)
        
        # 8) 确定主要运镜类型（优先级：子弹时间 > zoom/dolly > 标准运动）
        primary_motion = "static"
        primary_confidence = 0.0
        
        if bullet_type != "no_bullet_time" and bullet_conf > 0.7:
            primary_motion = bullet_type
            primary_confidence = bullet_conf
        elif motion_type != "no_dolly" and motion_conf > 0.6:
            primary_motion = motion_type
            primary_confidence = motion_conf
        elif enhanced_standard and enhanced_standard[0] != "None":
            primary_motion = enhanced_standard[0]
            primary_confidence = 0.8  # 标准检测的默认置信度
        
        return {
            "standard_motions": enhanced_standard,
            "primary_motion": primary_motion,
            "primary_confidence": primary_confidence,
            "motion_type": motion_type,
            "motion_confidence": motion_conf,
            "motion_details": motion_details,
            "bullet_time_type": bullet_type,
            "bullet_time_confidence": bullet_conf,
            "bullet_time_details": bullet_details,
            "face_pose_count": len([p for p in face_poses if p is not None]),
            "total_frames": len(face_poses),
            "foreground_mask_coverage": [mask.sum() / (mask.shape[0] * mask.shape[1]) for mask in masks]
        }
    
    def get_edge_point(self, track):
        middle = self.grid_size // 2
        number = self.number_points / 2.0
        start = ceil(middle-number)
        end = ceil(middle+number)
        idx=0
        top = [list(track[idx, i, :]) for i in range(start, end)]
        down = [list(track[self.grid_size-idx-1, i, :]) for i in range(start, end)]
        left = [list(track[i, idx, :]) for i in range(start, end)]
        right = [list(track[i, self.grid_size-idx-1, :]) for i in range(start, end)]
        return top, down, left, right
    
    def get_edge_point_360(self, track):
        middle = self.grid_size // 2
        number = 2
        lists=[0,1,self.grid_size-2,self.grid_size-1]
        idx=2
        res=[]
        for i in lists:
            if track[i, idx, 0]<0 or track[i, idx, 1]<0:
                res.append(None)
            else:
                res.append(list(track[i, idx, :]))
        return res
    
    def get_edge_direction_360(self, tracks):
        alls=[]
        for track1, track2 in zip(tracks[:-1], tracks[1:]):
            edge_points1 = self.get_edge_point_360(track1)
            edge_points2 = self.get_edge_point_360(track2)
            vector_results = []
            for points1, points2 in zip(edge_points1, edge_points2):
                if self.check_valid(points1) and self.check_valid(points2):
                    vector_results.append([points2[0]-points1[0], points2[1]-points1[1], points1[1]])
            if len(vector_results)==0:
                continue
            vector_results_360 = self.transform360(vector_results)
            class_results360 = transform_class360(vector_results_360, min_reso=self.scale)
            alls.append(class_results360)
        return alls
    
    def check_valid(self, point):
        if point is not None:
            if point[0]>0 and point[0]<self.width and point[1]>0 and point[1]<self.height:
                return True
            else:
                return False
        else:
            return False
        
    def get_edge_direction(self, track1, track2):
        edge_points1 = self.get_edge_point(track1)
        edge_points2 = self.get_edge_point(track2)
        vector_results = []
        for points1, points2 in zip(edge_points1, edge_points2):
            vectors = [[end[0]-start[0], end[1]-start[1], start[1]] for start, end in zip(points1, points2)]
            vector_results.append(vectors)
        vector_results_pan = list(map(transform, vector_results)) 
        class_results = [transform_class(vector, min_reso=self.scale) for vector in vector_results_pan]
        return class_results

    def classify_top_down(self, top, down):
        results = []
        classes = [f"{item_t}_{item_d}" for item_t in top for item_d in down]
        results_mapping = {
            "left_left": "pan_right",
            "right_right": "pan_left",
            "down_down": "tilt_up",
            "up_up": "tilt_down",
            "up_down": "zoom_in",
            "down_up": "zoom_out",
            "static_static": "static"
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else ["None"]
    
    def classify_left_right(self, left, right):
        results = []
        classes = [f"{item_l}_{item_r}" for item_l in left for item_r in right]
        results_mapping = {
            "left_left": "pan_right",
            "right_right": "pan_left",
            "down_down": "tilt_up",
            "up_up": "tilt_down",
            "left_right": "zoom_in",
            "right_left": "zoom_out",
            "static_static": "static"
        }
        results = [results_mapping.get(cls) for cls in classes if cls in results_mapping]
        return results if results else ["None"]


    def camera_classify(self, track1, track2, tracks):
        top, down, left, right = self.get_edge_direction(track1, track2)
        r360_results = self.get_edge_direction_360(tracks)
        top_results = self.classify_top_down(top, down)
        left_results = self.classify_left_right(left, right)
        results = list(set(top_results + left_results + r360_results))
        if "tilt_up" in results and "zoom_in" in results:
            results.append("oblique")
        if "static" in results and len(results)>1:
            results.remove("static")
        if "None" in results and len(results)>1:
            results.remove("None")  
        return results
    
    def predict(self, video, fps, end_frame):
        pred_track = self.infer(video, fps, end_frame)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        results = self.camera_classify(track1, track2, tracks)

        return results
    
def camera_motion(prompt_dict_ls, camera):
    sim = []
    video_results = []

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
    
            end_frame=-1
            scene_list = split_video_into_scenes(video_path, 5.0)
            if len(scene_list)!=0:
                end_frame = int(scene_list[0][1].get_frames())
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader))) 
            frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
            video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            predict_results = camera.predict(video, fps, end_frame)
            video_score = 1.0 if label in predict_results else 0.0
            video_results.append({'video_path': video_path, 'video_results': video_score})
            sim.append(video_score)
    
    avg_score = np.mean(sim)
    return avg_score, video_results

def compute_camera_motion(json_dir, device, submodules_dict, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    all_results, video_results = camera_motion(prompt_dict_ls, camera)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results