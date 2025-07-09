import os
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import traceback
from vbench2.camera_motion import compute_camera_motion
from vbench2.utils import init_submodules

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

class CameraMotionService:
    def __init__(self):
        self.device = None
        self.submodules_dict = None
        self.initialized = False
        
    def initialize(self, device='cuda:0', local_mode=False):
        """初始化评测模型"""
        try:
            # 设置设备
            if torch.cuda.is_available() and 'cuda' in device:
                self.device = device
                logger.info(f"Using device: {device}")
            else:
                self.device = 'cpu'
                logger.warning("CUDA not available, falling back to CPU")
            
            # 初始化子模块
            logger.info("Initializing Camera Motion submodules...")
            submodules_dict = init_submodules(['Camera_Motion'], local=local_mode)
            self.submodules_dict = submodules_dict['Camera_Motion']
            
            self.initialized = True
            logger.info("Camera Motion service initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Camera Motion service: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# 全局服务实例
camera_service = CameraMotionService()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'service': 'Camera Motion Evaluation',
        'initialized': camera_service.initialized
    })

@app.route('/initialize', methods=['POST'])
def initialize_service():
    """初始化服务接口"""
    try:
        data = request.json if request.json else {}
        device = data.get('device', 'cuda:0')
        local_mode = data.get('local_mode', False)
        
        if camera_service.initialize(device, local_mode):
            return jsonify({
                'status': 'success',
                'message': 'Service initialized successfully',
                'device': camera_service.device
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to initialize service'
            }), 500
            
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_camera_motion():
    """Camera Motion 评测接口"""
    try:
        # 检查服务是否已初始化
        if not camera_service.initialized:
            return jsonify({
                'status': 'error',
                'message': 'Service not initialized. Please call /initialize first.'
            }), 400
        
        # 获取请求参数
        data = request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        # 验证必需参数
        if 'video_paths' not in data:
            return jsonify({
                'status': 'error',
                'message': 'video_paths is required'
            }), 400
        
        video_paths = data['video_paths']
        if not isinstance(video_paths, list):
            video_paths = [video_paths]
        
        # 检查视频文件是否存在
        for video_path in video_paths:
            if not os.path.exists(video_path):
                return jsonify({
                    'status': 'error',
                    'message': f'Video file not found: {video_path}'
                }), 400
        
        # 获取运镜类型（可选）
        expected_motion = data.get('expected_motion', None)
        
        # 构造评测数据格式
        prompt_dict_ls = []
        for video_path in video_paths:
            prompt_dict = {
                'video_list': [video_path],
                'auxiliary_info': expected_motion if expected_motion else 'unknown'
            }
            prompt_dict_ls.append(prompt_dict)
        
        # 执行评测
        logger.info(f"Starting camera motion evaluation for {len(video_paths)} videos...")
        
        from vbench2.camera_motion import camera_motion, CameraPredict
        
        # 创建预测器
        camera_predictor = CameraPredict(camera_service.device, camera_service.submodules_dict)
        
        # 执行评测
        avg_score, video_results = camera_motion(prompt_dict_ls, camera_predictor)
        
        # 准备响应数据
        response = {
            'status': 'success',
            'overall_score': float(avg_score),
            'video_count': len(video_paths),
            'video_results': []
        }
        
        # 处理每个视频的结果
        for i, result in enumerate(video_results):
            video_result = {
                'video_path': result['video_path'],
                'score': float(result['video_results']),
                'expected_motion': expected_motion if expected_motion else 'unknown'
            }
            response['video_results'].append(video_result)
        
        logger.info(f"Evaluation completed. Overall score: {avg_score}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate_single', methods=['POST'])
def evaluate_single_video():
    """单个视频运镜评测接口（简化版）"""
    try:
        # 检查服务是否已初始化
        if not camera_service.initialized:
            return jsonify({
                'status': 'error',
                'message': 'Service not initialized. Please call /initialize first.'
            }), 400
        
        data = request.json
        if not data or 'video_path' not in data:
            return jsonify({
                'status': 'error',
                'message': 'video_path is required'
            }), 400
        
        video_path = data['video_path']
        expected_motion = data.get('expected_motion', 'unknown')
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return jsonify({
                'status': 'error',
                'message': f'Video file not found: {video_path}'
            }), 400
        
        # 直接使用 camera motion 预测
        from vbench2.camera_motion import CameraPredict
        import cv2
        import decord
        
        camera_predictor = CameraPredict(camera_service.device, camera_service.submodules_dict)
        
        # 读取视频
        decord.bridge.set_bridge('torch')
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader))) 
        video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
        
        # 获取FPS
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # 执行预测
        predicted_motions = camera_predictor.predict(video, fps, -1)
        
        # 计算分数
        score = 1.0 if expected_motion in predicted_motions else 0.0
        
        response = {
            'status': 'success',
            'video_path': video_path,
            'predicted_motions': predicted_motions,
            'expected_motion': expected_motion,
            'score': score,
            'match': expected_motion in predicted_motions
        }
        
        logger.info(f"Single video evaluation completed for {video_path}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during single video evaluation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict_motion', methods=['POST'])
def predict_motion():
    """仅预测运镜类型，不需要期望运镜类型"""
    try:
        # 检查服务是否已初始化
        if not camera_service.initialized:
            return jsonify({
                'status': 'error',
                'message': 'Service not initialized. Please call /initialize first.'
            }), 400
        
        data = request.json
        if not data or 'video_path' not in data:
            return jsonify({
                'status': 'error',
                'message': 'video_path is required'
            }), 400
        
        video_path = data['video_path']
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            return jsonify({
                'status': 'error',
                'message': f'Video file not found: {video_path}'
            }), 400
        
        # 使用 camera motion 预测
        from vbench2.camera_motion import CameraPredict
        import cv2
        import decord
        
        camera_predictor = CameraPredict(camera_service.device, camera_service.submodules_dict)
        
        # 读取视频
        decord.bridge.set_bridge('torch')
        video_reader = decord.VideoReader(video_path)
        video = video_reader.get_batch(range(len(video_reader))) 
        video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
        
        # 获取FPS
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # 执行预测
        predicted_motions = camera_predictor.predict(video, fps, -1)
        
        response = {
            'status': 'success',
            'video_path': video_path,
            'predicted_motions': predicted_motions,
            'motion_types': {
                'pan_left': 'pan_left' in predicted_motions,
                'pan_right': 'pan_right' in predicted_motions,
                'tilt_up': 'tilt_up' in predicted_motions,
                'tilt_down': 'tilt_down' in predicted_motions,
                'zoom_in': 'zoom_in' in predicted_motions,
                'zoom_out': 'zoom_out' in predicted_motions,
                'orbits': 'orbits' in predicted_motions,
                'oblique': 'oblique' in predicted_motions,
                'static': 'static' in predicted_motions
            }
        }
        
        logger.info(f"Motion prediction completed for {video_path}: {predicted_motions}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error during motion prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # 启动时的配置
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Camera Motion Evaluation Service on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    
    app.run(host=HOST, port=PORT, debug=DEBUG) 