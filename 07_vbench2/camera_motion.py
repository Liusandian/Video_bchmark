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

    def visualize_and_save_tracks(self, video, pred_tracks, pred_visibility, output_path, fps=30):
        """
        使用OpenCV将CoTracker的特征点可视化并保存为视频
        
        Args:
            video: 输入视频张量 (B, T, C, H, W)
            pred_tracks: 预测的特征点轨迹 (B, T, N, 2)
            pred_visibility: 特征点的可见性 (B, T, N, 1)
            output_path: 输出视频路径
            fps: 输出视频的帧率
        """
        # 获取视频尺寸
        B, T, C, H, W = video.shape
        
        # 转换视频格式为numpy数组 (T, H, W, C)
        video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # 获取特征点数据
        tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
        visibility = pred_visibility[0].cpu().numpy()  # (T, N, 1)
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        # 生成不同颜色用于不同的特征点
        num_points = tracks.shape[1]
        colors = []
        for i in range(num_points):
            color = (
                int(255 * (i % 3) / 3),
                int(255 * ((i // 3) % 3) / 3),
                int(255 * ((i // 9) % 3) / 3)
            )
            colors.append(color)
        
        print(f"正在可视化特征点轨迹，共 {T} 帧，{num_points} 个特征点...")
        
        for t in tqdm(range(T), desc="处理帧"):
            # 获取当前帧
            frame = video_np[t].copy()
            
            # 绘制当前帧的所有可见特征点
            for n in range(num_points):
                if visibility[t, n, 0] > 0.5:  # 只绘制可见的点
                    x, y = int(tracks[t, n, 0]), int(tracks[t, n, 1])
                    
                    # 确保坐标在图像范围内
                    if 0 <= x < W and 0 <= y < H:
                        # 绘制特征点（圆圈）
                        cv2.circle(frame, (x, y), 3, colors[n], -1)
                        
                        # 绘制特征点轨迹（如果不是第一帧）
                        if t > 0:
                            # 寻找前一个可见帧的位置
                            prev_t = t - 1
                            while prev_t >= 0 and visibility[prev_t, n, 0] <= 0.5:
                                prev_t -= 1
                            
                            if prev_t >= 0:
                                prev_x, prev_y = int(tracks[prev_t, n, 0]), int(tracks[prev_t, n, 1])
                                if 0 <= prev_x < W and 0 <= prev_y < H:
                                    cv2.line(frame, (prev_x, prev_y), (x, y), colors[n], 2)
            
            # 在帧上添加信息文本
            cv2.putText(frame, f"Frame: {t+1}/{T}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Visible Points: {int(np.sum(visibility[t, :, 0] > 0.5))}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 写入帧到输出视频
            out.write(frame)
        
        # 释放资源
        out.release()
        print(f"特征点可视化视频已保存到: {output_path}")

    def visualize_grid_tracks(self, video, pred_tracks, pred_visibility, output_path, fps=30):
        """
        专门为网格形式的特征点进行可视化
        
        Args:
            video: 输入视频张量 (B, T, C, H, W)
            pred_tracks: 预测的特征点轨迹 (B, T, N, 2)
            pred_visibility: 特征点的可见性 (B, T, N, 1)
            output_path: 输出视频路径
            fps: 输出视频的帧率
        """
        # 获取视频尺寸
        B, T, C, H, W = video.shape
        
        # 转换视频格式为numpy数组
        video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        # 获取特征点数据
        tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
        visibility = pred_visibility[0].cpu().numpy()  # (T, N, 1)
        
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        print(f"正在可视化网格特征点，共 {T} 帧，网格大小 {self.grid_size}x{self.grid_size}...")
        
        for t in tqdm(range(T), desc="处理帧"):
            frame = video_np[t].copy()
            
            # 重塑为网格形式
            grid_tracks = tracks[t].reshape(self.grid_size, self.grid_size, 2)
            grid_visibility = visibility[t].reshape(self.grid_size, self.grid_size, 1)
            
            # 绘制网格点和连线
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if grid_visibility[i, j, 0] > 0.5:
                        x, y = int(grid_tracks[i, j, 0]), int(grid_tracks[i, j, 1])
                        
                        if 0 <= x < W and 0 <= y < H:
                            # 绘制网格点
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                            
                            # 绘制网格连线（水平和垂直）
                            if j < self.grid_size - 1 and grid_visibility[i, j+1, 0] > 0.5:
                                next_x, next_y = int(grid_tracks[i, j+1, 0]), int(grid_tracks[i, j+1, 1])
                                if 0 <= next_x < W and 0 <= next_y < H:
                                    cv2.line(frame, (x, y), (next_x, next_y), (255, 0, 0), 1)
                            
                            if i < self.grid_size - 1 and grid_visibility[i+1, j, 0] > 0.5:
                                next_x, next_y = int(grid_tracks[i+1, j, 0]), int(grid_tracks[i+1, j, 1])
                                if 0 <= next_x < W and 0 <= next_y < H:
                                    cv2.line(frame, (x, y), (next_x, next_y), (255, 0, 0), 1)
            
            # 添加信息文本
            cv2.putText(frame, f"Frame: {t+1}/{T}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Grid: {self.grid_size}x{self.grid_size}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"网格特征点可视化视频已保存到: {output_path}")

    def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos", visualization_type="grid"):
        b,_,_,h,w=video.shape
        self.scale=min(h,w)
        self.height=h
        self.width=w
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        
        if save_video:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, "cotracker_visualization.mp4")
            
            if visualization_type == "grid":
                self.visualize_grid_tracks(video, pred_tracks, pred_visibility, output_path, fps)
            else:
                self.visualize_and_save_tracks(video, pred_tracks, pred_visibility, output_path, fps)
            
            return pred_tracks[0].long().detach().cpu().numpy(), pred_visibility[0].detach().cpu().numpy()
            
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        return pred_tracks[0].long().detach().cpu().numpy(), pred_visibility[0].detach().cpu().numpy()
    
    def find_stable_feature_points(self, pred_visibility, threshold=0.8):
        """
        找到在threshold比例以上的帧中都可见的稳定特征点
        
        Args:
            pred_visibility: (T, N) 可见性矩阵
            threshold: 可见性阈值，0.8表示在80%以上的帧中可见
            
        Returns:
            stable_point_ids: 稳定特征点的ID列表
        """
        T, N = pred_visibility.shape
        
        # 计算每个特征点的可见性比例 (大于0.5认为可见)
        visibility_ratio = np.mean(pred_visibility > 0.5, axis=0)  # (N,)
        
        # 找到可见性比例大于阈值的特征点
        stable_point_ids = np.where(visibility_ratio >= threshold)[0]
        
        print(f"总特征点数: {N}, 稳定特征点数: {len(stable_point_ids)} ({len(stable_point_ids)/N*100:.1f}%)")
        print(f"稳定特征点可见性统计: min={visibility_ratio[stable_point_ids].min():.2f}, "
              f"max={visibility_ratio[stable_point_ids].max():.2f}, "
              f"mean={visibility_ratio[stable_point_ids].mean():.2f}")
        
        return stable_point_ids
    
    def analyze_zoom_motion(self, pred_tracks, stable_point_ids, first_frame=0, last_frame=-1):
        """
        基于稳定特征点分析zoom运动
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            zoom_type: 'zoom_in', 'zoom_out', 'static', 'complex'
            motion_details: 详细运动信息
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(stable_point_ids) == 0:
            return 'static', {'reason': 'no_stable_points'}
        
        # 获取稳定特征点的初始和最终位置
        initial_points = pred_tracks[first_frame, stable_point_ids]  # (N_stable, 2)
        final_points = pred_tracks[last_frame, stable_point_ids]     # (N_stable, 2)
        
        # 计算图像中心
        center_x, center_y = self.width / 2, self.height / 2
        
        # 计算每个点到中心的初始距离和最终距离
        initial_distances = np.sqrt((initial_points[:, 0] - center_x)**2 + 
                                  (initial_points[:, 1] - center_y)**2)
        final_distances = np.sqrt((final_points[:, 0] - center_x)**2 + 
                                (final_points[:, 1] - center_y)**2)
        
        # 计算距离变化
        distance_changes = final_distances - initial_distances
        
        # 分析边缘点和中心点的运动模式
        # 将点按距离中心的远近分为边缘点和中心点
        center_distance_threshold = min(self.width, self.height) * 0.3
        edge_point_mask = initial_distances > center_distance_threshold
        
        edge_distance_changes = distance_changes[edge_point_mask] if np.any(edge_point_mask) else np.array([])
        center_distance_changes = distance_changes[~edge_point_mask] if np.any(~edge_point_mask) else np.array([])
        
        # 统计运动方向
        zoom_out_ratio = np.mean(distance_changes > self.scale * 0.02) if len(distance_changes) > 0 else 0
        zoom_in_ratio = np.mean(distance_changes < -self.scale * 0.02) if len(distance_changes) > 0 else 0
        static_ratio = 1 - zoom_out_ratio - zoom_in_ratio
        
        # 边缘点的运动分析（更重要）
        edge_zoom_out_ratio = np.mean(edge_distance_changes > self.scale * 0.02) if len(edge_distance_changes) > 0 else 0
        edge_zoom_in_ratio = np.mean(edge_distance_changes < -self.scale * 0.02) if len(edge_distance_changes) > 0 else 0
        
        motion_details = {
            'total_stable_points': len(stable_point_ids),
            'edge_points': np.sum(edge_point_mask),
            'center_points': np.sum(~edge_point_mask),
            'zoom_out_ratio': zoom_out_ratio,
            'zoom_in_ratio': zoom_in_ratio,
            'static_ratio': static_ratio,
            'edge_zoom_out_ratio': edge_zoom_out_ratio,
            'edge_zoom_in_ratio': edge_zoom_in_ratio,
            'mean_distance_change': np.mean(distance_changes),
            'edge_mean_distance_change': np.mean(edge_distance_changes) if len(edge_distance_changes) > 0 else 0
        }
        
        # 判断运动类型
        confidence_threshold = 0.6  # 至少60%的点要有一致的运动
        
        # 优先基于边缘点判断，因为边缘点对zoom运动更敏感
        if len(edge_distance_changes) > 0:
            if edge_zoom_out_ratio >= confidence_threshold:
                zoom_type = 'zoom_out'
            elif edge_zoom_in_ratio >= confidence_threshold:
                zoom_type = 'zoom_in'
            elif edge_zoom_out_ratio + edge_zoom_in_ratio < 0.3:  # 大部分边缘点静止
                zoom_type = 'static'
            else:
                zoom_type = 'complex'
        else:
            # 如果没有边缘点，则基于全部点判断
            if zoom_out_ratio >= confidence_threshold:
                zoom_type = 'zoom_out'
            elif zoom_in_ratio >= confidence_threshold:
                zoom_type = 'zoom_in'
            elif static_ratio >= confidence_threshold:
                zoom_type = 'static'
            else:
                zoom_type = 'complex'
        
        return zoom_type, motion_details
    
    def analyze_pan_tilt_motion(self, pred_tracks, stable_point_ids, first_frame=0, last_frame=-1):
        """
        分析pan和tilt运动
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            motion_type: pan/tilt运动类型
            motion_details: 详细运动信息
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(stable_point_ids) == 0:
            return 'static', {'reason': 'no_stable_points'}
        
        # 获取稳定特征点的运动向量
        initial_points = pred_tracks[first_frame, stable_point_ids]  # (N_stable, 2)
        final_points = pred_tracks[last_frame, stable_point_ids]     # (N_stable, 2)
        
        # 计算运动向量
        motion_vectors = final_points - initial_points  # (N_stable, 2)
        
        # 计算平均运动向量
        mean_motion_x = np.mean(motion_vectors[:, 0])
        mean_motion_y = np.mean(motion_vectors[:, 1])
        
        # 设置运动阈值
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
    
    def detect_subject(self, pred_tracks, pred_visibility, frame_idx=0):
        """
        检测画面中的主体区域
        基于特征点密度和中心位置来识别主体
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            frame_idx: 分析的帧索引
            
        Returns:
            subject_points: 主体区域的特征点ID列表
            subject_center: 主体中心坐标
            subject_bbox: 主体边界框
        """
        # 获取当前帧的可见特征点
        visible_mask = pred_visibility[frame_idx] > 0.5
        visible_points = pred_tracks[frame_idx, visible_mask]
        visible_ids = np.where(visible_mask)[0]
        
        if len(visible_points) == 0:
            return [], None, None
        
        # 计算图像中心区域（假设主体通常在中心）
        center_x, center_y = self.width / 2, self.height / 2
        center_region_ratio = 0.6  # 中心区域占比
        
        center_width = self.width * center_region_ratio
        center_height = self.height * center_region_ratio
        
        # 找到中心区域的特征点
        center_mask = (
            (visible_points[:, 0] >= center_x - center_width/2) &
            (visible_points[:, 0] <= center_x + center_width/2) &
            (visible_points[:, 1] >= center_y - center_height/2) &
            (visible_points[:, 1] <= center_y + center_height/2)
        )
        
        center_points = visible_points[center_mask]
        center_point_ids = visible_ids[center_mask]
        
        if len(center_points) == 0:
            # 如果中心区域没有点，使用密度最高的区域
            from scipy import spatial
            if len(visible_points) > 3:
                # 使用KD树找到密度最高的区域
                tree = spatial.cKDTree(visible_points)
                distances, indices = tree.query(visible_points, k=min(5, len(visible_points)))
                density_scores = 1.0 / (np.mean(distances, axis=1) + 1e-6)
                
                # 选择密度最高的点作为主体中心
                center_idx = np.argmax(density_scores)
                subject_center = visible_points[center_idx]
                
                # 选择周围的点作为主体
                radius = min(self.width, self.height) * 0.2
                distances_to_center = np.sqrt(np.sum((visible_points - subject_center)**2, axis=1))
                subject_mask = distances_to_center <= radius
                
                subject_points = visible_ids[subject_mask]
                subject_center = np.mean(visible_points[subject_mask], axis=0)
            else:
                subject_points = visible_ids
                subject_center = np.mean(visible_points, axis=0)
        else:
            subject_points = center_point_ids
            subject_center = np.mean(center_points, axis=0)
        
        # 计算主体边界框
        if len(subject_points) > 0:
            subject_coords = pred_tracks[frame_idx, subject_points]
            min_x, min_y = np.min(subject_coords, axis=0)
            max_x, max_y = np.max(subject_coords, axis=0)
            subject_bbox = [min_x, min_y, max_x, max_y]
        else:
            subject_bbox = None
        
        return subject_points, subject_center, subject_bbox
    
    def analyze_perspective_change(self, pred_tracks, stable_point_ids, first_frame=0, last_frame=-1):
        """
        分析透视变化，这是dolly运动的关键特征
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            perspective_change: 透视变化指标
            depth_layers: 深度层次分析结果
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(stable_point_ids) == 0:
            return None, None
        
        # 获取起始和结束帧的特征点位置
        initial_points = pred_tracks[first_frame, stable_point_ids]
        final_points = pred_tracks[last_frame, stable_point_ids]
        
        # 计算图像中心
        center_x, center_y = self.width / 2, self.height / 2
        center = np.array([center_x, center_y])
        
        # 计算每个点到中心的距离和角度
        initial_distances = np.sqrt(np.sum((initial_points - center)**2, axis=1))
        final_distances = np.sqrt(np.sum((final_points - center)**2, axis=1))
        
        # 计算运动向量
        motion_vectors = final_points - initial_points
        motion_magnitudes = np.sqrt(np.sum(motion_vectors**2, axis=1))
        
        # 分析径向运动模式（dolly运动的特征）
        initial_directions = (initial_points - center) / (initial_distances[:, np.newaxis] + 1e-6)
        radial_motion = np.sum(motion_vectors * initial_directions, axis=1)
        
        # 按距离中心的远近分层（近景、中景、远景）
        distance_percentiles = np.percentile(initial_distances, [33, 67])
        near_mask = initial_distances <= distance_percentiles[0]
        mid_mask = (initial_distances > distance_percentiles[0]) & (initial_distances <= distance_percentiles[1])
        far_mask = initial_distances > distance_percentiles[1]
        
        # 分析各层的运动模式
        layers = {
            'near': {'mask': near_mask, 'motion': radial_motion[near_mask] if np.any(near_mask) else np.array([])},
            'mid': {'mask': mid_mask, 'motion': radial_motion[mid_mask] if np.any(mid_mask) else np.array([])},
            'far': {'mask': far_mask, 'motion': radial_motion[far_mask] if np.any(far_mask) else np.array([])}
        }
        
        # 计算透视变化指标
        perspective_metrics = {}
        
        for layer_name, layer_data in layers.items():
            if len(layer_data['motion']) > 0:
                perspective_metrics[f'{layer_name}_mean_radial'] = np.mean(layer_data['motion'])
                perspective_metrics[f'{layer_name}_std_radial'] = np.std(layer_data['motion'])
            else:
                perspective_metrics[f'{layer_name}_mean_radial'] = 0
                perspective_metrics[f'{layer_name}_std_radial'] = 0
        
        # 检测dolly运动的典型模式
        dolly_indicators = self._detect_dolly_pattern(perspective_metrics, motion_magnitudes)
        
        return dolly_indicators, layers
    
    def _detect_dolly_pattern(self, perspective_metrics, motion_magnitudes):
        """
        检测dolly运动的典型模式
        """
        # dolly运动的特征：
        # 1. 近景点向外/内运动幅度大
        # 2. 远景点运动幅度小
        # 3. 运动方向具有一致性
        
        near_radial = perspective_metrics.get('near_mean_radial', 0)
        mid_radial = perspective_metrics.get('mid_mean_radial', 0)
        far_radial = perspective_metrics.get('far_mean_radial', 0)
        
        # 运动阈值
        motion_threshold = min(self.width, self.height) * 0.01
        
        indicators = {
            'has_dolly_pattern': False,
            'dolly_direction': 'static',
            'confidence': 0.0,
            'near_motion': near_radial,
            'mid_motion': mid_radial,
            'far_motion': far_radial
        }
        
        # 检测dolly in模式（近景向外，远景向内或静止）
        if near_radial > motion_threshold and near_radial > abs(far_radial):
            indicators['has_dolly_pattern'] = True
            indicators['dolly_direction'] = 'dolly_in'
            indicators['confidence'] = min(1.0, abs(near_radial) / motion_threshold)
        
        # 检测dolly out模式（近景向内，远景向外或静止）
        elif near_radial < -motion_threshold and abs(near_radial) > abs(far_radial):
            indicators['has_dolly_pattern'] = True
            indicators['dolly_direction'] = 'dolly_out'
            indicators['confidence'] = min(1.0, abs(near_radial) / motion_threshold)
        
        return indicators
    
    def analyze_subject_motion(self, pred_tracks, pred_visibility, first_frame=0, last_frame=-1):
        """
        分析主体的运动模式
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            subject_motion: 主体运动分析结果
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
        
        # 检测起始帧和结束帧的主体
        subject_points_start, subject_center_start, subject_bbox_start = self.detect_subject(
            pred_tracks, pred_visibility, first_frame)
        subject_points_end, subject_center_end, subject_bbox_end = self.detect_subject(
            pred_tracks, pred_visibility, last_frame)
        
        if subject_center_start is None or subject_center_end is None:
            return {'valid': False, 'reason': 'no_subject_detected'}
        
        # 计算主体中心的移动
        center_motion = subject_center_end - subject_center_start
        
        # 计算主体大小变化
        size_change = 0
        if subject_bbox_start is not None and subject_bbox_end is not None:
            area_start = (subject_bbox_start[2] - subject_bbox_start[0]) * (subject_bbox_start[3] - subject_bbox_start[1])
            area_end = (subject_bbox_end[2] - subject_bbox_end[0]) * (subject_bbox_end[3] - subject_bbox_end[1])
            size_change = (area_end - area_start) / (area_start + 1e-6)
        
        # 分析主体特征点的运动模式
        common_points = list(set(subject_points_start) & set(subject_points_end))
        
        subject_motion_pattern = None
        if len(common_points) > 0:
            subject_initial = pred_tracks[first_frame, common_points]
            subject_final = pred_tracks[last_frame, common_points]
            
            # 计算主体内部运动的一致性
            subject_motion_vectors = subject_final - subject_initial
            motion_consistency = np.std(subject_motion_vectors, axis=0)
            
            subject_motion_pattern = {
                'motion_vectors': subject_motion_vectors,
                'consistency': motion_consistency,
                'common_points_count': len(common_points)
            }
        
        return {
            'valid': True,
            'center_motion': center_motion,
            'size_change': size_change,
            'subject_points_start': subject_points_start,
            'subject_points_end': subject_points_end,
            'motion_pattern': subject_motion_pattern
        }
    
    def analyze_dolly_motion(self, pred_tracks, pred_visibility, stable_point_ids, first_frame=0, last_frame=-1):
        """
        综合分析dolly运动
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            dolly_result: dolly运动分析结果
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
        
        # 1. 透视变化分析
        perspective_result, depth_layers = self.analyze_perspective_change(
            pred_tracks, stable_point_ids, first_frame, last_frame)
        
        # 2. 主体运动分析
        subject_result = self.analyze_subject_motion(
            pred_tracks, pred_visibility, first_frame, last_frame)
        
        # 3. 综合判断dolly运动
        dolly_motion = {
            'type': 'static',
            'confidence': 0.0,
            'details': {
                'perspective': perspective_result,
                'subject': subject_result,
                'depth_layers': depth_layers
            }
        }
        
        if perspective_result and perspective_result.get('has_dolly_pattern', False):
            dolly_type = perspective_result['dolly_direction']
            confidence = perspective_result['confidence']
            
            # 验证主体运动是否与透视变化一致
            if subject_result.get('valid', False):
                # dolly in: 主体应该变大或位置变化不大
                # dolly out: 主体应该变小或位置变化不大
                subject_size_change = subject_result.get('size_change', 0)
                
                if dolly_type == 'dolly_in' and subject_size_change >= -0.1:  # 允许轻微缩小
                    confidence *= 1.2  # 提高置信度
                elif dolly_type == 'dolly_out' and subject_size_change <= 0.1:  # 允许轻微放大
                    confidence *= 1.2  # 提高置信度
                else:
                    confidence *= 0.8  # 降低置信度
            
            dolly_motion['type'] = dolly_type
            dolly_motion['confidence'] = min(1.0, confidence)
        
        return dolly_motion
    
    def detect_dolly_zoom_combination(self, pred_tracks, pred_visibility, stable_point_ids, first_frame=0, last_frame=-1):
        """
        检测dolly和zoom的复合运镜（如dolly in + zoom out）
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧
            last_frame: 结束帧
            
        Returns:
            combination_result: 复合运镜分析结果
        """
        # 分析dolly运动
        dolly_result = self.analyze_dolly_motion(
            pred_tracks, pred_visibility, stable_point_ids, first_frame, last_frame)
        
        # 分析zoom运动
        zoom_result, _ = self.analyze_zoom_motion(pred_tracks, stable_point_ids, first_frame, last_frame)
        
        # 检测复合运镜
        combination = {
            'type': 'none',
            'dolly_component': dolly_result,
            'zoom_component': zoom_result,
            'confidence': 0.0
        }
        
        dolly_type = dolly_result.get('type', 'static')
        dolly_confidence = dolly_result.get('confidence', 0.0)
        
        # 检测经典的复合运镜
        if dolly_type != 'static' and zoom_result != 'static':
            if dolly_type == 'dolly_in' and zoom_result == 'zoom_out':
                combination['type'] = 'dolly_in_zoom_out'
                combination['confidence'] = min(dolly_confidence, 0.8)  # 这种组合比较难检测
            elif dolly_type == 'dolly_out' and zoom_result == 'zoom_in':
                combination['type'] = 'dolly_out_zoom_in'
                combination['confidence'] = min(dolly_confidence, 0.8)
            elif dolly_type == 'dolly_in' and zoom_result == 'zoom_in':
                combination['type'] = 'dolly_in_zoom_in'
                combination['confidence'] = (dolly_confidence + 0.6) / 2  # 运动叠加，容易检测
            elif dolly_type == 'dolly_out' and zoom_result == 'zoom_out':
                combination['type'] = 'dolly_out_zoom_out'
                combination['confidence'] = (dolly_confidence + 0.6) / 2
        elif dolly_type != 'static':
            combination['type'] = dolly_type
            combination['confidence'] = dolly_confidence
        
        return combination
    
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
    
    def get_edge_point_from_stable_points(self, pred_tracks, stable_point_ids, frame_idx):
        """
        从稳定特征点中提取边缘区域的点
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            frame_idx: 帧索引
            
        Returns:
            top, down, left, right: 四个边缘区域的点列表
        """
        # 获取指定帧的稳定特征点坐标
        stable_points = pred_tracks[frame_idx, stable_point_ids, :]  # (N_stable, 2)
        
        # 定义边缘区域的阈值
        edge_threshold = min(self.width, self.height) * 0.15  # 边缘区域阈值
        
        # 根据位置将点分类到不同边缘区域
        top_points = []
        down_points = []
        left_points = []
        right_points = []
        
        for point in stable_points:
            x, y = point[0], point[1]
            
            # 检查是否在顶部边缘
            if y <= edge_threshold:
                top_points.append([float(x), float(y)])
            
            # 检查是否在底部边缘  
            if y >= self.height - edge_threshold:
                down_points.append([float(x), float(y)])
            
            # 检查是否在左侧边缘
            if x <= edge_threshold:
                left_points.append([float(x), float(y)])
            
            # 检查是否在右侧边缘
            if x >= self.width - edge_threshold:
                right_points.append([float(x), float(y)])
        
        # 如果某个边缘区域点数不足，使用最近的点补充
        min_points_per_edge = 2
        
        if len(top_points) < min_points_per_edge:
            # 找到Y坐标最小的点作为顶部点
            y_coords = stable_points[:, 1]
            top_indices = np.argsort(y_coords)[:min_points_per_edge]
            top_points = [[float(stable_points[i, 0]), float(stable_points[i, 1])] for i in top_indices]
        
        if len(down_points) < min_points_per_edge:
            # 找到Y坐标最大的点作为底部点
            y_coords = stable_points[:, 1]
            down_indices = np.argsort(y_coords)[-min_points_per_edge:]
            down_points = [[float(stable_points[i, 0]), float(stable_points[i, 1])] for i in down_indices]
        
        if len(left_points) < min_points_per_edge:
            # 找到X坐标最小的点作为左侧点
            x_coords = stable_points[:, 0]
            left_indices = np.argsort(x_coords)[:min_points_per_edge]
            left_points = [[float(stable_points[i, 0]), float(stable_points[i, 1])] for i in left_indices]
        
        if len(right_points) < min_points_per_edge:
            # 找到X坐标最大的点作为右侧点
            x_coords = stable_points[:, 0]
            right_indices = np.argsort(x_coords)[-min_points_per_edge:]
            right_points = [[float(stable_points[i, 0]), float(stable_points[i, 1])] for i in right_indices]
        
        return top_points, down_points, left_points, right_points
    
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
    
    def get_edge_direction_from_stable_points(self, pred_tracks, stable_point_ids, frame1_idx, frame2_idx):
        """
        基于稳定特征点分析边缘区域的运动方向
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            frame1_idx: 起始帧索引
            frame2_idx: 结束帧索引
            
        Returns:
            class_results: 四个边缘区域的运动分类结果
        """
        # 获取两帧的边缘点
        edge_points1 = self.get_edge_point_from_stable_points(pred_tracks, stable_point_ids, frame1_idx)
        edge_points2 = self.get_edge_point_from_stable_points(pred_tracks, stable_point_ids, frame2_idx)
        
        vector_results = []
        
        # 计算每个边缘区域的运动向量
        for points1, points2 in zip(edge_points1, edge_points2):
            if len(points1) == 0 or len(points2) == 0:
                # 如果某个边缘区域没有点，跳过
                vector_results.append([])
                continue
                
            # 计算运动向量（使用最近邻匹配或简单的对应关系）
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
                vector_results_pan.append([0, 0])  # 无运动
        
        # 分类运动方向
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
    
    def camera_classify_from_stable_points(self, pred_tracks, stable_point_ids, first_frame=0, last_frame=-1):
        """
        基于稳定特征点进行相机运动分类
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            stable_point_ids: 稳定特征点ID列表
            first_frame: 起始帧索引
            last_frame: 结束帧索引
            
        Returns:
            results: 检测到的相机运动类型列表
        """
        if last_frame == -1:
            last_frame = pred_tracks.shape[0] - 1
            
        if len(stable_point_ids) == 0:
            return ['static']
        
        # 使用边缘点分析运动方向
        top, down, left, right = self.get_edge_direction_from_stable_points(
            pred_tracks, stable_point_ids, first_frame, last_frame)
        
        # 分析上下边缘和左右边缘的运动模式
        top_results = self.classify_top_down(top, down)
        left_results = self.classify_left_right(left, right)
        
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
        
        return results if results else ["static"]
    
    def predict(self, video, fps, end_frame):
        pred_track, pred_visibility = self.infer(video, fps, end_frame)
        
        # 使用新的稳定特征点分析方法
        results = self.predict_with_stable_points(pred_track, pred_visibility)
        
        # 保持与原有方法的兼容性，如果新方法没有检测到运动，则使用原有方法
        if results == ['static'] or not results:
            track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
            track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
            tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
            results = self.camera_classify(track1, track2, tracks)

        return results
    
    def predict_with_stable_points(self, pred_tracks, pred_visibility, stability_threshold=0.8):
        """
        基于稳定特征点的相机运动预测
        
        Args:
            pred_tracks: (T, N, 2) 轨迹数据
            pred_visibility: (T, N) 可见性数据
            stability_threshold: 稳定性阈值，默认0.8（80%）
            
        Returns:
            results: 检测到的相机运动类型列表
        """
        # 1. 找到稳定的特征点
        stable_point_ids = self.find_stable_feature_points(pred_visibility, stability_threshold)
        
        if len(stable_point_ids) == 0:
            print("未找到稳定的特征点，无法进行运动分析")
            return ['static']
        
        results = []
        
        # 2. 使用新的边缘点分析方法（解决reshape维度问题）
        edge_results = self.camera_classify_from_stable_points(pred_tracks, stable_point_ids)
        print(f"边缘点分析结果: {edge_results}")
        
        if edge_results and edge_results != ['static']:
            results.extend(edge_results)
        
        # 3. 分析zoom运动（作为补充验证）
        zoom_type, zoom_details = self.analyze_zoom_motion(pred_tracks, stable_point_ids)
        print(f"Zoom分析结果: {zoom_type}")
        print(f"Zoom详细信息: {zoom_details}")
        
        if zoom_type in ['zoom_in', 'zoom_out'] and zoom_type not in results:
            results.append(zoom_type)
        
        # 4. 分析dolly运动（新增功能）
        dolly_zoom_result = self.detect_dolly_zoom_combination(pred_tracks, pred_visibility, stable_point_ids)
        print(f"Dolly/Zoom组合分析结果: {dolly_zoom_result}")
        
        # 添加dolly运动结果
        dolly_type = dolly_zoom_result.get('type', 'none')
        if dolly_type != 'none' and dolly_type not in results:
            if dolly_zoom_result.get('confidence', 0) > 0.3:  # 设置置信度阈值
                results.append(dolly_type)
        
        # 5. 分析pan/tilt运动（作为补充验证）
        pan_tilt_types, pan_tilt_details = self.analyze_pan_tilt_motion(pred_tracks, stable_point_ids)
        print(f"Pan/Tilt分析结果: {pan_tilt_types}")
        print(f"Pan/Tilt详细信息: {pan_tilt_details}")
        
        # 过滤掉静态运动，并与边缘分析结果融合
        non_static_pan_tilt = [t for t in pan_tilt_types if t != 'static']
        for motion_type in non_static_pan_tilt:
            if motion_type not in results:
                results.append(motion_type)
        
        # 6. 检测复杂运动组合
        if len(results) > 1:
            # 检查是否存在oblique运动（zoom + tilt的组合）
            has_zoom = any(r in ['zoom_in', 'zoom_out'] for r in results)
            has_tilt = any(r in ['tilt_up', 'tilt_down'] for r in results)
            has_dolly = any(r in ['dolly_in', 'dolly_out'] for r in results)
            
            if has_zoom and has_tilt and 'oblique' not in results:
                results.append('oblique')
            
            # 检测dolly zoom效果（著名的电影运镜技法）
            if has_dolly and has_zoom:
                dolly_zoom_effects = [r for r in results if 'dolly_' in r and 'zoom_' in r]
                if not dolly_zoom_effects:  # 如果没有检测到复合运镜，添加单独的标记
                    results.append('dolly_zoom_effect')
        
        # 7. 去重并过滤
        results = list(set(results))
        if 'static' in results and len(results) > 1:
            results.remove('static')
        
        return results if results else ['static']

def visualize_camera_motion(video_path, output_dir="./visualizations", device="cuda", submodules_dict=None, visualization_type="grid"):
    """
    独立的可视化函数，用于可视化视频中的CoTracker特征点
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        device: 设备类型 ("cuda" 或 "cpu")
        submodules_dict: CoTracker模型配置字典
        visualization_type: 可视化类型 ("grid" 或 "tracks")
    """
    if submodules_dict is None:
        # 默认配置
        submodules_dict = {
            "repo": "facebookresearch/co-tracker",
            "model": "cotracker2_online"
        }
    
    # 创建相机预测器
    camera = CameraPredict(device, submodules_dict)
    
    # 读取视频
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    video = video.permute(0, 3, 1, 2)[None].float()
    
    if device == "cuda":
        video = video.cuda()
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 进行可视化
    print(f"开始处理视频: {video_path}")
    print(f"视频尺寸: {video.shape}")
    print(f"输出目录: {output_dir}")
    
    _ = camera.infer(video, fps=fps, save_video=True, save_dir=output_dir, visualization_type=visualization_type)
    
    print("可视化完成！")

def camera_motion(prompt_dict_ls, camera, save_visualizations=False, vis_output_dir="./camera_motion_visualizations"):
    sim = []
    video_results = []

    if save_visualizations:
        os.makedirs(vis_output_dir, exist_ok=True)

    for prompt_dict in tqdm(prompt_dict_ls):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        for idx, video_path in enumerate(video_paths):
    
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
            
            # 如果需要保存可视化
            if save_visualizations:
                vis_save_dir = os.path.join(vis_output_dir, f"video_{idx}_{label}")
                os.makedirs(vis_save_dir, exist_ok=True)
                camera.infer(video, fps, end_frame, save_video=True, save_dir=vis_save_dir, visualization_type="grid")
            
            predict_results = camera.predict(video, fps, end_frame)
            video_score = 1.0 if label in predict_results else 0.0
            video_results.append({'video_path': video_path, 'video_results': video_score})
            sim.append(video_score)
    
    avg_score = np.mean(sim)
    return avg_score, video_results

def test_stable_point_edge_analysis(video_path, output_dir="./test_analysis", device="cuda", stability_threshold=0.8):
    """
    测试新的稳定特征点边缘分析功能（解决reshape维度问题）
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        device: 设备类型
        stability_threshold: 稳定性阈值
    """
    import decord
    
    # 默认模型配置
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 创建相机预测器
    camera = CameraPredict(device, submodules_dict)
    
    # 读取视频
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    video = video.permute(0, 3, 1, 2)[None].float()
    
    if device == "cuda":
        video = video.cuda()
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"开始分析视频: {video_path}")
    print(f"视频尺寸: {video.shape}")
    print(f"FPS: {fps}")
    
    # 进行预测和分析
    pred_tracks, pred_visibility = camera.infer(video, fps=fps, save_video=False)
    
    print(f"\n=== 轨迹数据统计 ===")
    print(f"轨迹形状: {pred_tracks.shape}")  # (T, N, 2)
    print(f"可见性形状: {pred_visibility.shape}")  # (T, N)
    
    # 分析稳定特征点
    stable_point_ids = camera.find_stable_feature_points(pred_visibility, stability_threshold)
    
    if len(stable_point_ids) > 0:
        print(f"\n=== 新边缘点分析方法测试 ===")
        
        # 测试新的边缘点提取
        edge_points = camera.get_edge_point_from_stable_points(pred_tracks, stable_point_ids, 0)
        print(f"边缘点数量统计:")
        print(f"  顶部: {len(edge_points[0])}")
        print(f"  底部: {len(edge_points[1])}")
        print(f"  左侧: {len(edge_points[2])}")
        print(f"  右侧: {len(edge_points[3])}")
        
        # 测试边缘运动方向分析
        edge_directions = camera.get_edge_direction_from_stable_points(
            pred_tracks, stable_point_ids, 0, -1)
        print(f"边缘运动方向: {edge_directions}")
        
        # 测试新的相机分类方法
        camera_results = camera.camera_classify_from_stable_points(pred_tracks, stable_point_ids)
        print(f"边缘点分析结果: {camera_results}")
        
        # 测试完整的稳定点预测
        final_results = camera.predict_with_stable_points(pred_tracks, pred_visibility, stability_threshold)
        print(f"\n=== 最终运动分类结果 ===")
        print(f"检测到的相机运动类型: {final_results}")
        
    else:
        print("未找到稳定的特征点，无法进行详细分析")
    
    # 可选：保存可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_output_path = os.path.join(output_dir, "edge_analysis_visualization.mp4")
        camera.visualize_grid_tracks(video, 
                                   torch.tensor(pred_tracks)[None], 
                                   torch.tensor(pred_visibility)[None], 
                                   vis_output_path, fps)
        print(f"\n可视化结果已保存到: {vis_output_path}")

def test_dolly_motion_detection(video_path, output_dir="./test_dolly_analysis", device="cuda", stability_threshold=0.8):
    """
    专门测试dolly运动检测功能
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        device: 设备类型
        stability_threshold: 稳定性阈值
    """
    import decord
    
    # 默认模型配置
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 创建相机预测器
    camera = CameraPredict(device, submodules_dict)
    
    # 读取视频
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    video = video.permute(0, 3, 1, 2)[None].float()
    
    if device == "cuda":
        video = video.cuda()
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"开始分析dolly运动: {video_path}")
    print(f"视频尺寸: {video.shape}")
    print(f"FPS: {fps}")
    
    # 进行预测和分析
    pred_tracks, pred_visibility = camera.infer(video, fps=fps, save_video=False)
    
    print(f"\n=== 轨迹数据统计 ===")
    print(f"轨迹形状: {pred_tracks.shape}")
    print(f"可见性形状: {pred_visibility.shape}")
    
    # 分析稳定特征点
    stable_point_ids = camera.find_stable_feature_points(pred_visibility, stability_threshold)
    
    if len(stable_point_ids) > 0:
        print(f"\n=== Dolly运动检测测试 ===")
        
        # 1. 测试主体检测
        subject_points, subject_center, subject_bbox = camera.detect_subject(pred_tracks, pred_visibility, 0)
        print(f"主体检测结果:")
        print(f"  主体特征点数: {len(subject_points)}")
        print(f"  主体中心: {subject_center}")
        print(f"  主体边界框: {subject_bbox}")
        
        # 2. 测试透视变化分析
        perspective_result, depth_layers = camera.analyze_perspective_change(pred_tracks, stable_point_ids)
        print(f"\n透视变化分析:")
        if perspective_result:
            print(f"  检测到dolly模式: {perspective_result.get('has_dolly_pattern', False)}")
            print(f"  Dolly方向: {perspective_result.get('dolly_direction', 'static')}")
            print(f"  置信度: {perspective_result.get('confidence', 0.0):.3f}")
            print(f"  近景运动: {perspective_result.get('near_motion', 0.0):.3f}")
            print(f"  中景运动: {perspective_result.get('mid_motion', 0.0):.3f}")
            print(f"  远景运动: {perspective_result.get('far_motion', 0.0):.3f}")
        
        # 3. 测试主体运动分析
        subject_motion = camera.analyze_subject_motion(pred_tracks, pred_visibility)
        print(f"\n主体运动分析:")
        if subject_motion.get('valid', False):
            print(f"  中心移动: {subject_motion.get('center_motion', [0, 0])}")
            print(f"  大小变化: {subject_motion.get('size_change', 0.0):.3f}")
            motion_pattern = subject_motion.get('motion_pattern')
            if motion_pattern:
                print(f"  一致性: {motion_pattern.get('consistency', [0, 0])}")
        
        # 4. 测试dolly运动综合分析
        dolly_result = camera.analyze_dolly_motion(pred_tracks, pred_visibility, stable_point_ids)
        print(f"\nDolly运动综合分析:")
        print(f"  运动类型: {dolly_result.get('type', 'static')}")
        print(f"  置信度: {dolly_result.get('confidence', 0.0):.3f}")
        
        # 5. 测试dolly+zoom复合运镜检测
        combination_result = camera.detect_dolly_zoom_combination(pred_tracks, pred_visibility, stable_point_ids)
        print(f"\nDolly+Zoom复合运镜分析:")
        print(f"  复合运镜类型: {combination_result.get('type', 'none')}")
        print(f"  整体置信度: {combination_result.get('confidence', 0.0):.3f}")
        print(f"  Dolly组件: {combination_result.get('dolly_component', {}).get('type', 'static')}")
        print(f"  Zoom组件: {combination_result.get('zoom_component', 'static')}")
        
        # 6. 完整的运动分类结果
        final_results = camera.predict_with_stable_points(pred_tracks, pred_visibility, stability_threshold)
        print(f"\n=== 最终运动分类结果（包含Dolly检测）===")
        print(f"检测到的所有相机运动类型: {final_results}")
        
    else:
        print("未找到稳定的特征点，无法进行dolly运动分析")
    
    # 可选：保存可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_output_path = os.path.join(output_dir, "dolly_motion_visualization.mp4")
        camera.visualize_grid_tracks(video, 
                                   torch.tensor(pred_tracks)[None], 
                                   torch.tensor(pred_visibility)[None], 
                                   vis_output_path, fps)
        print(f"\n可视化结果已保存到: {vis_output_path}")

def test_stable_point_analysis(video_path, output_dir="./test_analysis", device="cuda", stability_threshold=0.8):
    """
    测试稳定特征点分析功能
    
    Args:
        video_path: 输入视频路径
        output_dir: 输出目录
        device: 设备类型
        stability_threshold: 稳定性阈值
    """
    import decord
    
    # 默认模型配置
    submodules_dict = {
        "repo": "facebookresearch/co-tracker",
        "model": "cotracker2_online"
    }
    
    # 创建相机预测器
    camera = CameraPredict(device, submodules_dict)
    
    # 读取视频
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader)))
    video = video.permute(0, 3, 1, 2)[None].float()
    
    if device == "cuda":
        video = video.cuda()
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    
    print(f"开始分析视频: {video_path}")
    print(f"视频尺寸: {video.shape}")
    print(f"FPS: {fps}")
    
    # 进行预测和分析
    pred_tracks, pred_visibility = camera.infer(video, fps=fps, save_video=False)
    
    print(f"\n=== 轨迹数据统计 ===")
    print(f"轨迹形状: {pred_tracks.shape}")  # (T, N, 2)
    print(f"可见性形状: {pred_visibility.shape}")  # (T, N)
    
    # 分析稳定特征点
    stable_point_ids = camera.find_stable_feature_points(pred_visibility, stability_threshold)
    
    if len(stable_point_ids) > 0:
        print(f"\n=== 运动分析结果 ===")
        
        # Zoom运动分析
        zoom_type, zoom_details = camera.analyze_zoom_motion(pred_tracks, stable_point_ids)
        print(f"Zoom运动类型: {zoom_type}")
        print(f"Zoom运动详情:")
        for key, value in zoom_details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Pan/Tilt运动分析
        pan_tilt_types, pan_tilt_details = camera.analyze_pan_tilt_motion(pred_tracks, stable_point_ids)
        print(f"\nPan/Tilt运动类型: {pan_tilt_types}")
        print(f"Pan/Tilt运动详情:")
        for key, value in pan_tilt_details.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 综合分析结果
        final_results = camera.predict_with_stable_points(pred_tracks, pred_visibility, stability_threshold)
        print(f"\n=== 最终运动分类结果 ===")
        print(f"检测到的相机运动类型: {final_results}")
        
    else:
        print("未找到稳定的特征点，无法进行详细分析")
    
    # 可选：保存可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        vis_output_path = os.path.join(output_dir, "stable_point_visualization.mp4")
        camera.visualize_grid_tracks(video, 
                                   torch.tensor(pred_tracks)[None], 
                                   torch.tensor(pred_visibility)[None], 
                                   vis_output_path, fps)
        print(f"\n可视化结果已保存到: {vis_output_path}")

def compute_camera_motion(json_dir, device, submodules_dict, save_visualizations=False, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    all_results, video_results = camera_motion(prompt_dict_ls, camera, save_visualizations=save_visualizations)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results