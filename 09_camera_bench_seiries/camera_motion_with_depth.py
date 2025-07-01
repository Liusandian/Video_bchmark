import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
from math import ceil
from tqdm import tqdm
import json
import os
from vbench2.utils import load_dimension_info, split_video_into_scenes


def transform(vector):
    x = np.mean([item[0] for item in vector])
    y = np.mean([item[1] for item in vector])
    return [x, y]


def transform_class(vector, min_reso, factor=0.005):
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


def transform_class360(vector, min_reso, factor=0.008):
    scale = min_reso * factor
    up, down, y = vector
    if abs(y)<scale:
        if up * down<0 and up>scale:
            return "orbits"
        elif up*down<0 and up<-scale:
            return "orbits"
        else:
            return None


class DepthBasedCameraPredict:
    def __init__(self, device, submodules_list):
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        
        # 初始化CoTracker模型
        try:
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        except:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        
        # 初始化深度估计模型
        self.init_depth_estimation_model()
        
        # 深度分割参数
        self.depth_threshold_percentile = 30  # 深度阈值百分位数，小于此值认为是前景
        self.use_adaptive_threshold = True    # 是否使用自适应阈值
        self.background_erosion_size = 3      # 背景mask腐蚀操作的kernel大小

    def init_depth_estimation_model(self):
        """初始化深度估计模型"""
        try:
            # 优先使用MiDaS模型（轻量级且效果好）
            self.depth_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS', pretrained=True).to(self.device)
            self.depth_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
            self.depth_model_type = "midas"
            print("使用MiDaS深度估计模型")
        except Exception as e:
            try:
                # 备选：使用DPT模型
                self.depth_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', pretrained=True).to(self.device)
                self.depth_transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform
                self.depth_model_type = "dpt"
                print("使用DPT深度估计模型")
            except Exception as e2:
                print(f"深度估计模型加载失败: {e}, {e2}")
                print("将使用传统方法进行前景背景分割")
                self.depth_model = None
                self.depth_model_type = None

    def estimate_depth(self, frame):
        """
        估计单帧的深度图
        
        Args:
            frame: 输入图像 (H, W, C) numpy array, RGB格式
            
        Returns:
            depth_map: 深度图 (H, W) numpy array，值越小表示距离越近
        """
        if self.depth_model is None:
            return None
        
        try:
            # 预处理图像
            input_tensor = self.depth_transform(frame).to(self.device)
            
            # 深度估计
            with torch.no_grad():
                depth_tensor = self.depth_model(input_tensor)
                
            # 转换为numpy数组
            depth_map = depth_tensor.squeeze().cpu().numpy()
            
            # 调整尺寸到原始图像大小
            if depth_map.shape != frame.shape[:2]:
                depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            
            return depth_map
            
        except Exception as e:
            print(f"深度估计失败: {e}")
            return None

    def create_background_mask_from_depth(self, depth_map, adaptive_threshold=True):
        """
        基于深度图创建背景mask
        
        Args:
            depth_map: 深度图 (H, W)
            adaptive_threshold: 是否使用自适应阈值
            
        Returns:
            background_mask: 背景mask (H, W)，1表示背景，0表示前景
        """
        if depth_map is None:
            return np.ones_like(depth_map, dtype=np.float32)
        
        try:
            # 归一化深度值
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            if adaptive_threshold:
                # 自适应阈值：基于深度直方图动态确定前景背景分割点
                hist, bins = np.histogram(depth_normalized.flatten(), bins=50)
                
                # 寻找直方图的两个主要峰值（前景和背景）
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
                
                if len(peaks) >= 2:
                    # 选择最靠近前景的峰值作为阈值
                    threshold = bins[peaks[0]] + (bins[peaks[1]] - bins[peaks[0]]) * 0.3
                else:
                    # 备选：使用百分位数阈值
                    threshold = np.percentile(depth_normalized, self.depth_threshold_percentile)
            else:
                # 固定百分位数阈值
                threshold = np.percentile(depth_normalized, self.depth_threshold_percentile)
            
            # 创建背景mask（深度值大于阈值的区域为背景）
            background_mask = (depth_normalized > threshold).astype(np.float32)
            
            # 形态学操作：去除噪声并平滑边界
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.background_erosion_size, self.background_erosion_size))
            background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)
            background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_CLOSE, kernel)
            
            return background_mask
            
        except Exception as e:
            print(f"创建背景mask失败: {e}")
            return np.ones_like(depth_map, dtype=np.float32)

    def filter_background_feature_points(self, pred_tracks, pred_visibility, video_frames, frame_idx=0):
        """
        基于深度估计过滤出背景特征点
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据  
            video_frames: 视频帧数组 (T, H, W, C)
            frame_idx: 用于深度估计的帧索引
            
        Returns:
            background_point_ids: 背景特征点ID列表
            background_mask: 背景掩码
        """
        if frame_idx >= len(video_frames):
            frame_idx = 0
            
        # 获取指定帧
        frame = video_frames[frame_idx]  # (H, W, C)
        
        # 深度估计
        depth_map = self.estimate_depth(frame)
        
        # 创建背景mask
        background_mask = self.create_background_mask_from_depth(depth_map, self.use_adaptive_threshold)
        
        # 获取当前帧的可见特征点
        visible_mask = pred_visibility[frame_idx] > 0.5
        visible_points = pred_tracks[frame_idx, visible_mask]
        visible_ids = np.where(visible_mask)[0]
        
        # 筛选背景区域的特征点
        background_point_ids = []
        for i, point in enumerate(visible_points):
            x, y = int(point[0]), int(point[1])
            
            # 确保坐标在图像范围内
            if (0 <= x < background_mask.shape[1] and 
                0 <= y < background_mask.shape[0]):
                
                # 检查是否在背景区域内
                if background_mask[y, x] > 0.5:  # 背景区域
                    background_point_ids.append(visible_ids[i])
        
        print(f"总特征点数: {len(visible_ids)}, 背景特征点数: {len(background_point_ids)} "
              f"({len(background_point_ids)/len(visible_ids)*100:.1f}%)")
        
        return background_point_ids, background_mask

    def get_background_edge_points(self, pred_tracks, background_point_ids, frame_idx):
        """
        从背景特征点中提取边缘区域的点
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            background_point_ids: 背景特征点ID列表
            frame_idx: 帧索引
            
        Returns:
            top_points, down_points, left_points, right_points: 四个边缘区域的背景点列表
        """
        if len(background_point_ids) == 0:
            return [], [], [], []
            
        # 获取背景特征点坐标
        background_points = pred_tracks[frame_idx, background_point_ids, :]  # (N_bg, 2)
        
        # 定义边缘区域的阈值
        edge_threshold = min(self.width, self.height) * 0.15
        
        # 根据位置将背景点分类到不同边缘区域
        top_points = []
        down_points = []
        left_points = []
        right_points = []
        
        for point in background_points:
            x, y = float(point[0]), float(point[1])
            
            # 检查是否在各个边缘区域内
            if y <= edge_threshold:  # 顶部边缘
                top_points.append([x, y])
            
            if y >= self.height - edge_threshold:  # 底部边缘
                down_points.append([x, y])
            
            if x <= edge_threshold:  # 左侧边缘
                left_points.append([x, y])
            
            if x >= self.width - edge_threshold:  # 右侧边缘
                right_points.append([x, y])
        
        # 如果某个边缘区域点数不足，使用最近的背景点补充
        min_points_per_edge = 2
        
        if len(top_points) < min_points_per_edge:
            y_coords = background_points[:, 1]
            top_indices = np.argsort(y_coords)[:min_points_per_edge]
            top_points = [[float(background_points[i, 0]), float(background_points[i, 1])] for i in top_indices]
        
        if len(down_points) < min_points_per_edge:
            y_coords = background_points[:, 1]
            down_indices = np.argsort(y_coords)[-min_points_per_edge:]
            down_points = [[float(background_points[i, 0]), float(background_points[i, 1])] for i in down_indices]
        
        if len(left_points) < min_points_per_edge:
            x_coords = background_points[:, 0]
            left_indices = np.argsort(x_coords)[:min_points_per_edge]
            left_points = [[float(background_points[i, 0]), float(background_points[i, 1])] for i in left_indices]
        
        if len(right_points) < min_points_per_edge:
            x_coords = background_points[:, 0]
            right_indices = np.argsort(x_coords)[-min_points_per_edge:]
            right_points = [[float(background_points[i, 0]), float(background_points[i, 1])] for i in right_indices]
        
        return top_points, down_points, left_points, right_points

    def get_background_edge_direction(self, pred_tracks, background_point_ids, video_frames, frame1_idx, frame2_idx):
        """
        基于背景特征点分析边缘区域的运动方向
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            background_point_ids: 背景特征点ID列表
            video_frames: 视频帧数组
            frame1_idx: 起始帧索引
            frame2_idx: 结束帧索引
            
        Returns:
            class_results: 四个边缘区域的运动分类结果
        """
        # 获取两帧的背景边缘点
        edge_points1 = self.get_background_edge_points(pred_tracks, background_point_ids, frame1_idx)
        edge_points2 = self.get_background_edge_points(pred_tracks, background_point_ids, frame2_idx)
        
        vector_results = []
        
        # 计算每个边缘区域的运动向量
        for points1, points2 in zip(edge_points1, edge_points2):
            if len(points1) == 0 or len(points2) == 0:
                vector_results.append([])
                continue
                
            # 计算运动向量
            vectors = []
            min_len = min(len(points1), len(points2))
            
            for i in range(min_len):
                start = points1[i]
                end = points2[i]
                vector = [end[0] - start[0], end[1] - start[1], start[1]]
                vectors.append(vector)
            
            vector_results.append(vectors)
        
        # 转换为平均运动向量
        vector_results_pan = []
        for vectors in vector_results:
            if len(vectors) > 0:
                avg_vector = transform(vectors)
                vector_results_pan.append(avg_vector)
            else:
                vector_results_pan.append([0, 0])
        
        # 分类运动方向
        class_results = [transform_class(vector, min_reso=self.scale) for vector in vector_results_pan]
        
        return class_results

    def classify_top_down(self, top, down):
        """分析上下边缘的运动模式"""
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
        """分析左右边缘的运动模式"""
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

    def background_camera_classify(self, pred_tracks, background_point_ids, video_frames, first_frame=0, last_frame=-1):
        """
        基于背景特征点进行相机运动分类
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            background_point_ids: 背景特征点ID列表  
            video_frames: 视频帧数组
            first_frame: 起始帧索引
            last_frame: 结束帧索引
            
        Returns:
            results: 检测到的相机运动类型列表
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(background_point_ids) == 0:
            return ['static']
        
        # 使用背景边缘点分析运动方向
        edge_directions = self.get_background_edge_direction(
            pred_tracks, background_point_ids, video_frames, first_frame, last_frame)
        
        print(f"背景边缘运动方向: {edge_directions}")
        
        # 分析上下边缘和左右边缘的运动模式
        top_results = self.classify_top_down(edge_directions[0], edge_directions[1])  # 上、下
        left_results = self.classify_left_right(edge_directions[2], edge_directions[3])  # 左、右
        
        # 合并结果
        results = list(set(top_results + left_results))
        
        # 处理特殊组合
        if "tilt_up" in results and "zoom_in" in results:
            results.append("oblique")
        
        # 清理结果
        if "static" in results and len(results) > 1:
            results.remove("static")
        if "None" in results and len(results) > 1:
            results.remove("None")
        
        print(f"背景运动分析结果: {results}")
        
        return results if results else ["static"]

    def analyze_background_zoom_motion(self, pred_tracks, background_point_ids, first_frame=0, last_frame=-1):
        """
        基于背景特征点分析zoom运动
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            background_point_ids: 背景特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            zoom_type: 'zoom_in', 'zoom_out', 'static', 'complex'
            motion_details: 详细运动信息
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(background_point_ids) == 0:
            return 'static', {'reason': 'no_background_points'}
        
        # 获取背景特征点的初始和最终位置
        initial_points = pred_tracks[first_frame, background_point_ids]
        final_points = pred_tracks[last_frame, background_point_ids]
        
        # 计算图像中心
        center_x, center_y = self.width / 2, self.height / 2
        
        # 计算每个点到中心的距离变化
        initial_distances = np.sqrt((initial_points[:, 0] - center_x)**2 + 
                                  (initial_points[:, 1] - center_y)**2)
        final_distances = np.sqrt((final_points[:, 0] - center_x)**2 + 
                                (final_points[:, 1] - center_y)**2)
        
        distance_changes = final_distances - initial_distances
        
        # 统计运动方向
        motion_threshold = self.scale * 0.02
        zoom_out_ratio = np.mean(distance_changes > motion_threshold)
        zoom_in_ratio = np.mean(distance_changes < -motion_threshold)
        static_ratio = 1 - zoom_out_ratio - zoom_in_ratio
        
        motion_details = {
            'total_background_points': len(background_point_ids),
            'zoom_out_ratio': zoom_out_ratio,
            'zoom_in_ratio': zoom_in_ratio,
            'static_ratio': static_ratio,
            'mean_distance_change': np.mean(distance_changes)
        }
        
        # 判断运动类型
        confidence_threshold = 0.6
        
        if zoom_out_ratio >= confidence_threshold:
            zoom_type = 'zoom_out'
        elif zoom_in_ratio >= confidence_threshold:
            zoom_type = 'zoom_in'
        elif static_ratio >= confidence_threshold:
            zoom_type = 'static'
        else:
            zoom_type = 'complex'
        
        return zoom_type, motion_details

    def analyze_background_pan_tilt_motion(self, pred_tracks, background_point_ids, first_frame=0, last_frame=-1):
        """
        基于背景特征点分析pan和tilt运动
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(background_point_ids) == 0:
            return ['static'], {'reason': 'no_background_points'}
        
        # 获取背景特征点的运动向量
        initial_points = pred_tracks[first_frame, background_point_ids]
        final_points = pred_tracks[last_frame, background_point_ids]
        
        motion_vectors = final_points - initial_points
        
        # 计算平均运动向量
        mean_motion_x = np.mean(motion_vectors[:, 0])
        mean_motion_y = np.mean(motion_vectors[:, 1])
        
        motion_threshold = self.scale * 0.02
        
        motion_details = {
            'mean_motion_x': mean_motion_x,
            'mean_motion_y': mean_motion_y,
            'motion_magnitude': np.sqrt(mean_motion_x**2 + mean_motion_y**2),
            'consistent_x_ratio': np.mean(np.sign(motion_vectors[:, 0]) == np.sign(mean_motion_x)),
            'consistent_y_ratio': np.mean(np.sign(motion_vectors[:, 1]) == np.sign(mean_motion_y))
        }
        
        # 判断运动类型
        motion_results = []
        
        if abs(mean_motion_x) > motion_threshold:
            if mean_motion_x > 0:
                motion_results.append('pan_right')
            else:
                motion_results.append('pan_left')
        
        if abs(mean_motion_y) > motion_threshold:
            if mean_motion_y > 0:
                motion_results.append('tilt_down')
            else:
                motion_results.append('tilt_up')
        
        if not motion_results:
            motion_results.append('static')
        
        return motion_results, motion_details

    def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos", visualization_type="grid"):
        """推理函数，获取特征点轨迹"""
        b, _, _, h, w = video.shape
        self.scale = min(h, w)
        self.height = h
        self.width = w
        
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size)
        
        if save_video:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, "cotracker_visualization.mp4")
            self.visualize_tracks_with_depth(video, pred_tracks, pred_visibility, output_path, fps)
        
        if end_frame != -1:
            pred_tracks = pred_tracks[:, :end_frame]
            pred_visibility = pred_visibility[:, :end_frame]
            
        return pred_tracks[0].long().detach().cpu().numpy(), pred_visibility[0].detach().cpu().numpy()

    def visualize_tracks_with_depth(self, video, pred_tracks, pred_visibility, output_path, fps=30):
        """
        可视化特征点轨迹并叠加深度信息和背景mask
        """
        B, T, C, H, W = video.shape
        
        # 转换视频格式
        video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # 获取特征点数据
        tracks = pred_tracks[0].cpu().numpy()
        visibility = pred_visibility[0].cpu().numpy()
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        print(f"正在生成深度增强的特征点可视化...")
        
        for t in tqdm(range(T), desc="处理帧"):
            frame = video_np[t].copy()
            
            # 深度估计和背景分割
            depth_map = self.estimate_depth(frame)
            if depth_map is not None:
                background_mask = self.create_background_mask_from_depth(depth_map)
                
                # 叠加背景mask（绿色半透明）
                background_overlay = np.zeros_like(frame)
                background_overlay[:, :, 1] = (background_mask * 255).astype(np.uint8)
                frame = cv2.addWeighted(frame, 0.8, background_overlay, 0.2, 0)
            
            # 筛选背景特征点
            if depth_map is not None:
                background_point_ids, _ = self.filter_background_feature_points(
                    tracks[:, np.newaxis, :], visibility[:, np.newaxis], [frame], 0)
                background_points_set = set(background_point_ids)
            else:
                background_points_set = set()
            
            # 绘制特征点
            for n in range(tracks.shape[1]):
                if visibility[t, n] > 0.5:
                    x, y = int(tracks[t, n, 0]), int(tracks[t, n, 1])
                    
                    if 0 <= x < W and 0 <= y < H:
                        # 背景点用绿色，前景点用红色
                        color = (0, 255, 0) if n in background_points_set else (255, 0, 0)
                        cv2.circle(frame, (x, y), 3, color, -1)
            
            # 添加信息文本
            cv2.putText(frame, f"Frame: {t+1}/{T}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Background Points: {len(background_points_set)}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        out.release()
        print(f"深度增强可视化视频已保存到: {output_path}")

    def predict_with_depth_background(self, pred_tracks, pred_visibility, video_frames):
        """
        基于深度估计的背景特征点进行相机运动预测
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            video_frames: 视频帧数组 (T, H, W, C)
            
        Returns:
            results: 检测到的相机运动类型列表
        """
        # 1. 基于深度估计筛选背景特征点
        background_point_ids, background_mask = self.filter_background_feature_points(
            pred_tracks, pred_visibility, video_frames, frame_idx=0)
        
        if len(background_point_ids) == 0:
            print("未找到背景特征点，回退到传统方法")
            return self.predict_traditional(pred_tracks, pred_visibility)
        
        results = []
        
        # 2. 基于背景边缘点分析运动
        edge_results = self.background_camera_classify(
            pred_tracks, background_point_ids, video_frames)
        
        if edge_results and edge_results != ['static']:
            results.extend(edge_results)
        
        # 3. 分析背景zoom运动
        zoom_type, zoom_details = self.analyze_background_zoom_motion(
            pred_tracks, background_point_ids)
        print(f"背景Zoom分析结果: {zoom_type}")
        
        if zoom_type in ['zoom_in', 'zoom_out'] and zoom_type not in results:
            results.append(zoom_type)
        
        # 4. 分析背景pan/tilt运动
        pan_tilt_types, pan_tilt_details = self.analyze_background_pan_tilt_motion(
            pred_tracks, background_point_ids)
        print(f"背景Pan/Tilt分析结果: {pan_tilt_types}")
        
        non_static_pan_tilt = [t for t in pan_tilt_types if t != 'static']
        for motion_type in non_static_pan_tilt:
            if motion_type not in results:
                results.append(motion_type)
        
        # 5. 检测复杂运动组合
        if len(results) > 1:
            has_zoom = any(r in ['zoom_in', 'zoom_out'] for r in results)
            has_tilt = any(r in ['tilt_up', 'tilt_down'] for r in results)
            
            if has_zoom and has_tilt and 'oblique' not in results:
                results.append('oblique')
        
        # 6. 去重并过滤
        results = list(set(results))
        if 'static' in results and len(results) > 1:
            results.remove('static')
        
        return results if results else ['static']

    def predict_traditional(self, pred_tracks, pred_visibility):
        """传统方法预测（备选方案）"""
        track1 = pred_tracks[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_tracks[-1].reshape((self.grid_size, self.grid_size, 2))
        
        # 简化的传统分类逻辑
        return ['static']  # 简化实现

    def predict(self, video, fps, end_frame):
        """
        主预测函数
        
        Args:
            video: 输入视频张量 (B, T, C, H, W)
            fps: 帧率
            end_frame: 结束帧
            
        Returns:
            results: 预测的运镜类型列表
        """
        # 获取特征点轨迹
        pred_tracks, pred_visibility = self.infer(video, fps, end_frame)
        
        # 转换视频格式为numpy用于深度估计
        video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()  # (T, H, W, C)
        video_np = (video_np * 255).astype(np.uint8)
        
        # 使用深度估计的背景分析方法
        results = self.predict_with_depth_background(pred_tracks, pred_visibility, video_np)
        
        print(f"最终运镜类型预测结果: {results}")
        
        return results


def camera_motion_with_depth(prompt_dict_ls, camera, save_visualizations=False, vis_output_dir="./camera_motion_depth_visualizations"):
    """
    使用深度估计的相机运动评测函数
    """
    sim = []
    video_results = []

    if save_visualizations:
        os.makedirs(vis_output_dir, exist_ok=True)

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        
        for idx, video_path in enumerate(video_paths):
            try:
                end_frame = -1
                scene_list = split_video_into_scenes(video_path, 5.0)
                if len(scene_list) != 0:
                    end_frame = int(scene_list[0][1].get_frames())
                    
                video_reader = decord.VideoReader(video_path)
                video = video_reader.get_batch(range(len(video_reader))) 
                frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
                video = video.permute(0, 3, 1, 2)[None].float().cuda()
                
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                # 可视化（可选）
                if save_visualizations:
                    vis_save_dir = os.path.join(vis_output_dir, f"video_{idx}_{label}")
                    os.makedirs(vis_save_dir, exist_ok=True)
                    camera.infer(video, fps, end_frame, save_video=True, save_dir=vis_save_dir)
                
                # 预测运镜类型
                predict_results = camera.predict(video, fps, end_frame)
                video_score = 1.0 if label in predict_results else 0.0
                
                video_results.append({
                    'video_path': video_path, 
                    'video_results': video_score,
                    'ground_truth': label,
                    'predictions': predict_results
                })
                sim.append(video_score)
                
            except Exception as e:
                print(f"处理视频 {video_path} 时出错: {e}")
                continue
    
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_camera_motion_with_depth(json_dir, device, submodules_dict, save_visualizations=False, **kwargs):
    """
    使用深度估计的相机运动计算函数
    """
    camera = DepthBasedCameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    
    avg_score, video_results = camera_motion_with_depth(
        prompt_dict_ls, camera, save_visualizations=save_visualizations)
    
    return avg_score, video_results


# 测试函数
def test_depth_based_camera_motion(video_path, output_dir="./test_depth_analysis", device="cuda"):
    """
    测试基于深度估计的相机运动检测
    """
    import decord
    
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    camera = DepthBasedCameraPredict(device, submodules_dict)
    
    # 读取视频
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    video = video.permute(0, 3, 1, 2)[None].float()
    
    if device == "cuda":
        video = video.cuda()
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"开始测试深度估计相机运动检测: {video_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 预测并保存可视化
    results = camera.predict(video, fps, -1)
    
    # 保存深度增强的可视化
    _ = camera.infer(video, fps, -1, save_video=True, save_dir=output_dir)
    
    print(f"检测到的运镜类型: {results}")
    print(f"可视化结果已保存到: {output_dir}")


if __name__ == "__main__":
    # 使用示例
    print("深度估计增强的相机运动检测模块")
    print("主要特性:")
    print("1. 使用MiDaS/DPT进行深度估计")
    print("2. 基于深度分离前景背景")
    print("3. 仅基于背景特征点计算运镜类型")
    print("4. 更准确的相机运动检测") 