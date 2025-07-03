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
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'DBFace-master'))
from model.DBFace import DBFace
import common as dbface_common


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
        
        # Initialize DBFace model for face detection
        self.face_model = DBFace()
        self.face_model.eval()
        if torch.cuda.is_available():
            self.face_model.cuda()
        # Load DBFace model weights
        dbface_model_path = os.path.join(os.path.dirname(__file__), 'third_party', 'DBFace-master', 'model', 'dbface.pth')
        if os.path.exists(dbface_model_path):
            self.face_model.load(dbface_model_path)
            self.face_detection_enabled = True
        else:
            print(f"Warning: DBFace model not found at {dbface_model_path}. Face detection will be disabled.")
            self.face_detection_enabled = False

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
    
    def detect_faces_in_frame(self, frame, threshold=0.4, nms_iou=0.5):
        """
        Detect faces in a single frame using DBFace
        Args:
            frame: numpy array of shape (H, W, C) in BGR format
            threshold: confidence threshold for face detection
            nms_iou: NMS IoU threshold
        Returns:
            List of BBox objects representing detected faces
        """
        if not self.face_detection_enabled:
            return []
            
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocessing for DBFace
        mean = [0.408, 0.447, 0.47]
        std = [0.289, 0.274, 0.278]
        
        image = dbface_common.pad(image)
        image = ((image / 255.0 - mean) / std).astype(np.float32)
        image = image.transpose(2, 0, 1)
        
        torch_image = torch.from_numpy(image)[None]
        if torch.cuda.is_available():
            torch_image = torch_image.cuda()
        
        # Forward pass
        with torch.no_grad():
            hm, box, landmark = self.face_model(torch_image)
            hm_pool = F.max_pool2d(hm, 3, 1, 1)
            scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
            hm_height, hm_width = hm.shape[2:]
        
        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices / hm_width).int().data.numpy())
        xs = list((indices % hm_width).int().data.numpy())
        scores = list(scores.data.numpy())
        box = box.cpu().squeeze().data.numpy()
        landmark = landmark.cpu().squeeze().data.numpy()
        
        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break
            
            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[:, cy, cx]
            x5y5 = (dbface_common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(dbface_common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
        
        # Apply NMS
        return self._nms_faces(objs, nms_iou)
    
    def _nms_faces(self, objs, iou=0.5):
        """Non-Maximum Suppression for face detection results"""
        if objs is None or len(objs) <= 1:
            return objs
        
        objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):
            if flags[index] != 0:
                continue
            
            keep.append(obj)
            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and obj.iou(objs[j]) > iou:
                    flags[j] = 1
        return keep
    
    def calculate_face_ratio(self, faces, frame_width, frame_height):
        """
        Calculate the ratio of face area to total frame area
        Args:
            faces: List of BBox objects representing detected faces
            frame_width: Width of the frame
            frame_height: Height of the frame
        Returns:
            Float: ratio of face area to total frame area
        """
        if not faces:
            return 0.0
        
        total_face_area = 0
        for face in faces:
            # Calculate face area
            face_area = face.area
            total_face_area += face_area
        
        total_frame_area = frame_width * frame_height
        return total_face_area / total_frame_area
    
    def analyze_face_zoom_motion(self, video_frames, start_idx=0, end_idx=-1):
        """
        Analyze face-based zoom motion by comparing face ratios between start and end frames
        Args:
            video_frames: List of video frames
            start_idx: Starting frame index
            end_idx: Ending frame index (-1 for last frame)
        Returns:
            Dict with face analysis results
        """
        if not self.face_detection_enabled:
            return {"face_motion": None, "confidence": 0.0}
        
        if end_idx == -1:
            end_idx = len(video_frames) - 1
        
        start_frame = video_frames[start_idx]
        end_frame = video_frames[end_idx]
        
        # Convert tensor frames to numpy if needed
        if isinstance(start_frame, torch.Tensor):
            start_frame = start_frame.permute(1, 2, 0).cpu().numpy()
            start_frame = (start_frame * 255).astype(np.uint8)
        if isinstance(end_frame, torch.Tensor):
            end_frame = end_frame.permute(1, 2, 0).cpu().numpy() 
            end_frame = (end_frame * 255).astype(np.uint8)
        
        # Detect faces in both frames
        start_faces = self.detect_faces_in_frame(start_frame)
        end_faces = self.detect_faces_in_frame(end_frame)
        
        # Calculate face ratios
        frame_height, frame_width = start_frame.shape[:2]
        start_ratio = self.calculate_face_ratio(start_faces, frame_width, frame_height)
        end_ratio = self.calculate_face_ratio(end_faces, frame_width, frame_height)
        
        # Analyze motion based on face ratio change
        ratio_change = end_ratio - start_ratio
        ratio_change_percent = (ratio_change / max(start_ratio, 0.001)) * 100 if start_ratio > 0 else 0
        
        # Determine face-based motion
        face_motion = None
        confidence = 0.0
        
        if abs(ratio_change_percent) > 20:  # Significant change threshold
            if ratio_change > 0:
                face_motion = "zoom_in"
                confidence = min(abs(ratio_change_percent) / 100, 1.0)
            else:
                face_motion = "zoom_out" 
                confidence = min(abs(ratio_change_percent) / 100, 1.0)
        
        return {
            "face_motion": face_motion,
            "confidence": confidence,
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "ratio_change_percent": ratio_change_percent,
            "start_faces_count": len(start_faces),
            "end_faces_count": len(end_faces)
        }

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


    def camera_classify(self, track1, track2, tracks, face_analysis=None):
        top, down, left, right = self.get_edge_direction(track1, track2)
        r360_results = self.get_edge_direction_360(tracks)
        top_results = self.classify_top_down(top, down)
        left_results = self.classify_left_right(left, right)
        results = list(set(top_results + left_results + r360_results))
        
        # Integrate face-based motion analysis
        if face_analysis and face_analysis.get("face_motion") and face_analysis.get("confidence", 0) > 0.3:
            face_motion = face_analysis["face_motion"]
            face_confidence = face_analysis["confidence"]
            
            # If face analysis suggests zoom in/out with high confidence, prioritize it
            if face_motion in ["zoom_in", "zoom_out"]:
                # Remove conflicting zoom predictions from traditional motion analysis
                results = [r for r in results if r not in ["zoom_in", "zoom_out"]]
                results.append(face_motion)
                
                # Add face-guided motion indicator
                print(f"Face-guided motion detected: {face_motion} (confidence: {face_confidence:.2f})")
                print(f"Face ratio change: {face_analysis.get('ratio_change_percent', 0):.1f}%")
        
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
        
        # Perform face-based motion analysis
        face_analysis = None
        if self.face_detection_enabled:
            try:
                # Extract frames for face analysis 
                video_frames = []
                b, t, c, h, w = video.shape
                for i in range(min(t, end_frame if end_frame != -1 else t)):
                    frame = video[0, i].permute(1, 2, 0).cpu().numpy()  # C,H,W -> H,W,C
                    frame = (frame * 255).astype(np.uint8)
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_frames.append(frame)
                
                if len(video_frames) > 1:
                    face_analysis = self.analyze_face_zoom_motion(video_frames, 0, -1)
            except Exception as e:
                print(f"Face analysis failed: {e}")
                face_analysis = None
        
        results = self.camera_classify(track1, track2, tracks, face_analysis)

        return results

def get_camera_motion_recommendations(video_path, camera, user_preference=None):
    """
    Comprehensive camera motion analysis with face detection integration
    Provides recommendations for camera motion types based on multiple analysis methods
    
    Args:
        video_path: Path to the video file
        camera: CameraPredict instance
        user_preference: Optional user preference for motion type
        
    Returns:
        Dict containing analysis results and recommendations
    """
    try:
        # Load and preprocess video
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader)))
        frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
        video = video.permute(0, 3, 1, 2)[None].float().cuda()  # B T C H W
        
        # Get FPS
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Detect scenes and limit analysis to first scene
        end_frame = -1
        scene_list = split_video_into_scenes(video_path, 5.0)
        if len(scene_list) != 0:
            end_frame = int(scene_list[0][1].get_frames())
        
        # Perform traditional camera motion analysis
        traditional_results = camera.predict(video, fps, end_frame)
        
        # Perform detailed face analysis if enabled
        face_analysis_detailed = None
        face_recommendations = {}
        
        if camera.face_detection_enabled:
            try:
                # Extract frames for detailed analysis
                video_frames = []
                b, t, c, h, w = video.shape
                for i in range(min(t, end_frame if end_frame != -1 else t)):
                    frame = video[0, i].permute(1, 2, 0).cpu().numpy()
                    frame = (frame * 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_frames.append(frame)
                
                if len(video_frames) > 1:
                    face_analysis_detailed = camera.analyze_face_zoom_motion(video_frames, 0, -1)
                    
                    # Generate face-based recommendations
                    if face_analysis_detailed["face_motion"]:
                        confidence = face_analysis_detailed["confidence"]
                        motion_type = face_analysis_detailed["face_motion"]
                        
                        face_recommendations = {
                            "recommended_motion": motion_type,
                            "confidence_level": "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low",
                            "confidence_score": confidence,
                            "reason": f"Face area changed by {face_analysis_detailed['ratio_change_percent']:.1f}%",
                            "face_count_start": face_analysis_detailed["start_faces_count"],
                            "face_count_end": face_analysis_detailed["end_faces_count"]
                        }
            except Exception as e:
                print(f"Detailed face analysis failed: {e}")
        
        # Combine results and generate final recommendations
        recommendations = {
            "video_path": video_path,
            "traditional_motion_analysis": traditional_results,
            "face_analysis": face_analysis_detailed,
            "face_recommendations": face_recommendations,
            "final_recommendations": []
        }
        
        # Generate final recommendations based on all analyses
        if face_recommendations and face_recommendations.get("confidence_score", 0) > 0.3:
            # Prioritize face-based analysis for zoom motions
            motion = face_recommendations["recommended_motion"]
            confidence = face_recommendations["confidence_level"]
            recommendations["final_recommendations"].append({
                "motion_type": motion,
                "confidence": confidence,
                "source": "Face Analysis",
                "description": f"Based on face detection: {face_recommendations['reason']}"
            })
        
        # Add traditional analysis results
        for motion in traditional_results:
            if motion not in ["None", "static"]:
                recommendations["final_recommendations"].append({
                    "motion_type": motion,
                    "confidence": "Medium",
                    "source": "Traditional Motion Tracking", 
                    "description": "Based on optical flow analysis"
                })
        
        # Remove duplicates and prioritize
        seen_motions = set()
        final_recs = []
        for rec in recommendations["final_recommendations"]:
            if rec["motion_type"] not in seen_motions:
                seen_motions.add(rec["motion_type"])
                final_recs.append(rec)
        
        recommendations["final_recommendations"] = final_recs
        
        # Add user guidance
        if len(final_recs) > 1:
            recommendations["user_guidance"] = {
                "message": "Multiple camera motions detected. Consider the following options:",
                "suggestions": [f"{rec['motion_type']} ({rec['confidence']} confidence)" for rec in final_recs]
            }
        elif len(final_recs) == 1:
            recommendations["user_guidance"] = {
                "message": f"Primary motion detected: {final_recs[0]['motion_type']}",
                "suggestions": [f"Recommended: {final_recs[0]['motion_type']}"]
            }
        else:
            recommendations["user_guidance"] = {
                "message": "No clear camera motion detected",
                "suggestions": ["Consider: static", "Consider: manual review"]
            }
        
        return recommendations
        
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "video_path": video_path,
            "recommendations": []
        }

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

def analyze_single_video_with_face_guidance(video_path, device, submodules_dict):
    """
    Analyze a single video with comprehensive camera motion detection including face guidance
    
    Args:
        video_path: Path to video file
        device: CUDA device
        submodules_dict: Dictionary containing model configuration
    
    Returns:
        Comprehensive analysis results with user guidance
    
    Example usage:
        submodules_dict = {"repo": "facebook/co-tracker", "model": "cotracker2"}
        results = analyze_single_video_with_face_guidance("path/to/video.mp4", "cuda", submodules_dict)
        print(results["user_guidance"]["message"])
        for suggestion in results["user_guidance"]["suggestions"]:
            print(f"  - {suggestion}")
    """
    camera = CameraPredict(device, submodules_dict)
    return get_camera_motion_recommendations(video_path, camera)

def demo_face_guided_camera_analysis():
    """
    Demo function showing how to use the face-guided camera motion analysis
    """
    # Example configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    submodules_dict = {"repo": "facebook/co-tracker", "model": "cotracker2"}
    
    # Example video path (replace with actual path)
    video_path = "path/to/your/video.mp4"
    
    try:
        # Perform comprehensive analysis
        results = analyze_single_video_with_face_guidance(video_path, device, submodules_dict)
        
        print("=== Camera Motion Analysis Results ===")
        print(f"Video: {results.get('video_path', 'Unknown')}")
        
        # Display traditional analysis
        print(f"\nTraditional Motion Analysis: {results.get('traditional_motion_analysis', [])}")
        
        # Display face analysis if available
        if results.get('face_analysis'):
            face_info = results['face_analysis']
            print(f"\nFace Analysis:")
            print(f"  - Motion detected: {face_info.get('face_motion', 'None')}")
            print(f"  - Confidence: {face_info.get('confidence', 0):.2f}")
            print(f"  - Face ratio change: {face_info.get('ratio_change_percent', 0):.1f}%")
            print(f"  - Faces at start: {face_info.get('start_faces_count', 0)}")
            print(f"  - Faces at end: {face_info.get('end_faces_count', 0)}")
        
        # Display recommendations
        print(f"\n=== Recommendations ===")
        print(results['user_guidance']['message'])
        for suggestion in results['user_guidance']['suggestions']:
            print(f"  ✓ {suggestion}")
        
        # Display detailed recommendations
        if results.get('final_recommendations'):
            print(f"\nDetailed Analysis:")
            for rec in results['final_recommendations']:
                print(f"  - {rec['motion_type']} ({rec['confidence']} confidence)")
                print(f"    Source: {rec['source']}")
                print(f"    Description: {rec['description']}")
        
        return results
        
    except Exception as e:
        print(f"Demo failed: {e}")
        return None