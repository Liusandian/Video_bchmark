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

# 导入VGGT通用子弹时间检测模块
try:
    from .vggt_universal_bullet_time import VGGTUniversalBulletTimeDetector
    _HAS_VGGT_BULLET_TIME = True
except ImportError:
    _HAS_VGGT_BULLET_TIME = False
    print("警告: VGGT通用子弹时间检测模块未找到")


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
        
        # 初始化VGGT通用子弹时间检测器
        self.vggt_bullet_time_detector = None
        if _HAS_VGGT_BULLET_TIME:
            try:
                self.vggt_bullet_time_detector = VGGTUniversalBulletTimeDetector(device, submodules_list)
                print("✓ VGGT通用子弹时间检测器初始化成功")
            except Exception as e:
                print(f"VGGT通用子弹时间检测器初始化失败: {e}")
                self.vggt_bullet_time_detector = None

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
    
    def predict_with_vggt_bullet_time(self, video_path, fps, end_frame):
        """增强预测方法，集成VGGT子弹时间检测"""
        # 原有相机运动分类
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader))) 
        video = video.permute(0, 3, 1, 2)[None].float().to(self.device) # B T C H W
        
        pred_track = self.infer(video, fps, end_frame)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        standard_results = self.camera_classify(track1, track2, tracks)
        
        # VGGT子弹时间检测
        bullet_result = {'is_bullet_time': False, 'confidence': 0.0, 'summary': {}}
        if self.vggt_bullet_time_detector:
            bullet_result = self.vggt_bullet_time_detector.detect_vggt_bullet_time(video_path)
        
        # 构建增强结果
        enhanced_results = standard_results.copy()
        if bullet_result['is_bullet_time']:
            if "bullet_time" not in enhanced_results:
                enhanced_results.append("bullet_time")
        
        return {
            'standard_motions': standard_results,
            'enhanced_motions': enhanced_results,
            'vggt_bullet_time': bullet_result,
            'has_orbit_motion': "orbits" in standard_results,
            'method': 'vggt_enhanced'
        }

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

def compute_camera_motion_with_vggt_bullet_time(json_dir, device, submodules_dict, **kwargs):
    """增强的相机运动评估，集成VGGT子弹时间检测"""
    camera = CameraPredict(device, submodules_dict)
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='camera_motion', lang='en')
    
    sim = []
    video_results = []
    
    for prompt_dict in tqdm(prompt_dict_ls, desc="VGGT增强相机运动评估"):
        label = prompt_dict['auxiliary_info']
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            end_frame = -1
            scene_list = split_video_into_scenes(video_path, 5.0)
            if len(scene_list) != 0:
                end_frame = int(scene_list[0][1].get_frames())
            
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            # 使用VGGT增强预测方法
            enhanced_result = camera.predict_with_vggt_bullet_time(video_path, fps, end_frame)
            
            # 评分逻辑
            enhanced_motions = enhanced_result['enhanced_motions']
            video_score = 1.0 if label in enhanced_motions else 0.0
            
            # 特殊处理子弹时间标签
            if 'bullet_time' in label.lower() or label == 'bullet_time':
                bullet_detected = enhanced_result['vggt_bullet_time']['is_bullet_time']
                video_score = 1.0 if bullet_detected else 0.0
            
            video_results.append({
                'video_path': video_path, 
                'video_results': video_score,
                'standard_motions': enhanced_result['standard_motions'],
                'enhanced_motions': enhanced_result['enhanced_motions'],
                'vggt_bullet_time_details': enhanced_result['vggt_bullet_time']
            })
            sim.append(video_score)
    
    avg_score = np.mean(sim) if sim else 0.0
    return avg_score, video_results