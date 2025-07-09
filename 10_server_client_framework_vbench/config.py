"""
VBench Camera Motion 评测服务配置文件
"""

import os

class Config:
    # 服务器配置
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # GPU设备配置
    DEFAULT_DEVICE = os.getenv('DEFAULT_DEVICE', 'cuda:0')
    
    # 超时配置（秒）
    EVALUATION_TIMEOUT = int(os.getenv('EVALUATION_TIMEOUT', 300))
    INITIALIZATION_TIMEOUT = int(os.getenv('INITIALIZATION_TIMEOUT', 600))
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024 * 1024  # 16GB
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
    
    # 日志配置
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # VBench配置
    VBENCH_CACHE_DIR = os.getenv('VBENCH2_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'vbench2'))
    
    # 支持的运镜类型
    SUPPORTED_MOTIONS = [
        'pan_left', 'pan_right',
        'tilt_up', 'tilt_down',
        'zoom_in', 'zoom_out',
        'orbits', 'oblique', 'static'
    ]
    
    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 