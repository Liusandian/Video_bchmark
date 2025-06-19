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
            
            return pred_tracks[0].long().detach().cpu().numpy()
            
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        return pred_tracks[0].long().detach().cpu().numpy()
    
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

def compute_camera_motion(json_dir, device, submodules_dict, save_visualizations=False, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    all_results, video_results = camera_motion(prompt_dict_ls, camera, save_visualizations=save_visualizations)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results