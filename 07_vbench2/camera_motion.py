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
        
        # 2. 分析zoom运动（优先级最高，因为最明显）
        zoom_type, zoom_details = self.analyze_zoom_motion(pred_tracks, stable_point_ids)
        print(f"Zoom分析结果: {zoom_type}")
        print(f"Zoom详细信息: {zoom_details}")
        
        if zoom_type in ['zoom_in', 'zoom_out']:
            results.append(zoom_type)
        
        # 3. 分析pan/tilt运动
        pan_tilt_types, pan_tilt_details = self.analyze_pan_tilt_motion(pred_tracks, stable_point_ids)
        print(f"Pan/Tilt分析结果: {pan_tilt_types}")
        print(f"Pan/Tilt详细信息: {pan_tilt_details}")
        
        # 过滤掉静态运动，除非没有其他运动
        non_static_pan_tilt = [t for t in pan_tilt_types if t != 'static']
        if non_static_pan_tilt:
            results.extend(non_static_pan_tilt)
        elif not results:  # 如果zoom也是静态的，则添加static
            results.append('static')
        
        # 4. 检测复杂运动组合
        if len(results) > 1:
            # 检查是否存在oblique运动（zoom + tilt的组合）
            has_zoom = any(r in ['zoom_in', 'zoom_out'] for r in results)
            has_tilt = any(r in ['tilt_up', 'tilt_down'] for r in results)
            if has_zoom and has_tilt:
                results.append('oblique')
        
        # 5. 去重并过滤
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