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
from PIL import Image


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

def load_foreground_masks(mask_dir, video_path):
    """
    加载离线处理好的前景mask文件
    Args:
        mask_dir: mask文件目录
        video_path: 视频文件路径
    Returns:
        masks: 前景mask数组，形状为 [T, H, W]
    """
    video_name = os.path.basename(video_path).split('.')[0]
    mask_path = os.path.join(mask_dir, f"{video_name}_masks.npy")
    
    if os.path.exists(mask_path):
        masks = np.load(mask_path)
        return masks
    else:
        print(f"Warning: Mask file not found for {video_path}")
        return None

def calculate_foreground_center(mask):
    """
    计算前景区域的中心点
    Args:
        mask: 前景mask，形状为 [H, W]
    Returns:
        center: (x, y) 中心点坐标
    """
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return (center_x, center_y)

def extract_foreground_features(tracks, masks, visibility):
    """
    从前景区域提取特征点
    Args:
        tracks: cotracker特征点轨迹 [T, N, 2]
        masks: 前景mask [T, H, W]
        visibility: 特征点可见性 [T, N, 1]
    Returns:
        foreground_tracks: 前景区域的特征点轨迹
        foreground_visibility: 前景区域特征点的可见性
    """
    T, N, _ = tracks.shape
    foreground_tracks = []
    foreground_visibility = []
    
    for t in range(T):
        if masks is not None and t < len(masks):
            mask = masks[t]
            frame_fg_tracks = []
            frame_fg_visibility = []
            
            for n in range(N):
                x, y = int(tracks[t, n, 0]), int(tracks[t, n, 1])
                # 检查特征点是否在前景区域内
                if (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and 
                    mask[y, x] > 0 and visibility[t, n, 0] > 0.5):
                    frame_fg_tracks.append([x, y])
                    frame_fg_visibility.append(1.0)
            
            if len(frame_fg_tracks) == 0:
                # 如果没有前景特征点，使用所有可见特征点
                for n in range(N):
                    if visibility[t, n, 0] > 0.5:
                        frame_fg_tracks.append([tracks[t, n, 0], tracks[t, n, 1]])
                        frame_fg_visibility.append(1.0)
            
            foreground_tracks.append(frame_fg_tracks)
            foreground_visibility.append(frame_fg_visibility)
        else:
            # 如果没有mask，使用所有可见特征点
            frame_tracks = []
            frame_visibility = []
            for n in range(N):
                if visibility[t, n, 0] > 0.5:
                    frame_tracks.append([tracks[t, n, 0], tracks[t, n, 1]])
                    frame_visibility.append(1.0)
            foreground_tracks.append(frame_tracks)
            foreground_visibility.append(frame_visibility)
    
    return foreground_tracks, foreground_visibility

def analyze_zoom_motion(foreground_tracks, masks, video_shape):
    """
    基于前景特征点到中心点的平均距离变化分析zoom运镜
    Args:
        foreground_tracks: 前景特征点轨迹
        masks: 前景mask数组
        video_shape: 视频尺寸 (H, W)
    Returns:
        motion_type: "dolly_in", "dolly_out", "static", 或 None
    """
    if len(foreground_tracks) < 2:
        return None
    
    distances = []
    h, w = video_shape
    
    for t, tracks in enumerate(foreground_tracks):
        if len(tracks) == 0:
            continue
            
        # 计算前景中心点
        if masks is not None and t < len(masks):
            fg_center = calculate_foreground_center(masks[t])
        else:
            fg_center = None
        
        # 如果没有前景中心点，使用画面中心
        if fg_center is None:
            fg_center = (w / 2, h / 2)
        
        # 计算所有前景特征点到中心的平均距离
        if len(tracks) > 0:
            track_distances = []
            for track in tracks:
                x, y = track[0], track[1]
                dist = np.sqrt((x - fg_center[0])**2 + (y - fg_center[1])**2)
                track_distances.append(dist)
            avg_distance = np.mean(track_distances)
            distances.append(avg_distance)
    
    if len(distances) < 3:
        return None
    
    # 分析距离变化趋势
    # 计算开始、中间、结束的平均距离
    start_dist = np.mean(distances[:len(distances)//3])
    end_dist = np.mean(distances[-len(distances)//3:])
    
    # 计算整体趋势
    x = np.arange(len(distances))
    slope = np.polyfit(x, distances, 1)[0]
    
    # 阈值设定
    distance_threshold = min(h, w) * 0.02  # 相对阈值
    slope_threshold = 0.5  # 斜率阈值
    
    # 判断zoom类型
    if abs(end_dist - start_dist) > distance_threshold:
        if slope < -slope_threshold:  # 距离递减
            return "dolly_in"
        elif slope > slope_threshold:  # 距离递增
            return "dolly_out"
    
    return "static"

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
        return pred_tracks[0].long().detach().cpu().numpy(), pred_visibility[0].detach().cpu().numpy()
    
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
    
    def predict(self, video, fps, end_frame, masks=None):
        pred_track, pred_visibility = self.infer(video, fps, end_frame)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        results = self.camera_classify(track1, track2, tracks)

        # 添加基于前景的zoom检测
        if masks is not None:
            zoom_result = self.analyze_foreground_zoom(pred_track, pred_visibility, masks, (self.height, self.width))
            if zoom_result and zoom_result != "static":
                if zoom_result not in results:
                    results.append(zoom_result)

        return results
    
    def analyze_foreground_zoom(self, tracks, visibility, masks, video_shape):
        """
        基于前景区域分析zoom运镜
        Args:
            tracks: cotracker特征点轨迹 [T, N, 2]
            visibility: 特征点可见性 [T, N, 1]
            masks: 前景mask数组 [T, H, W]
            video_shape: 视频尺寸 (H, W)
        Returns:
            motion_type: "dolly_in", "dolly_out", "static", 或 None
        """
        # 提取前景特征点
        foreground_tracks, _ = extract_foreground_features(tracks, masks, visibility)
        
        # 分析zoom运镜
        zoom_motion = analyze_zoom_motion(foreground_tracks, masks, video_shape)
        
        return zoom_motion
    
def camera_motion(prompt_dict_ls, camera, mask_dir=None):
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
            
            # 加载前景mask
            masks = None
            if mask_dir:
                masks = load_foreground_masks(mask_dir, video_path)
                if masks is not None and end_frame != -1:
                    masks = masks[:end_frame]
            
            predict_results = camera.predict(video, fps, end_frame, masks)
            video_score = 1.0 if label in predict_results else 0.0
            video_results.append({'video_path': video_path, 'video_results': video_score})
            sim.append(video_score)
    
    avg_score = np.mean(sim)
    return avg_score, video_results

def compute_camera_motion(json_dir, device, submodules_dict, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    
    # 获取mask目录路径（如果提供）
    mask_dir = kwargs.get('mask_dir', None)
    
    all_results, video_results = camera_motion(prompt_dict_ls, camera, mask_dir)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results