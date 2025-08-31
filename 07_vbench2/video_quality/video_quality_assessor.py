"""
视频质量检测与评估系统
用于分析视频的基础质量指标，包括模糊度、信噪比、对比度、亮度等
目的是筛选质量较差的视频，提高视频数据集的整体质量
"""

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from scipy.fft import fft2, fftshift
from skimage import filters, measure, exposure
from skimage.metrics import structural_similarity as ssim
import argparse
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')


class VideoQualityAssessor:
    """
    视频质量评估器
    
    支持多种质量指标:
    1. 模糊度检测 - Laplacian variance, Tenengrad, FFT-based
    2. 信噪比分析 - PSNR, SNR estimation
    3. 对比度分析 - RMS contrast, Michelson contrast
    4. 亮度分析 - 平均亮度、亮度分布
    5. 色彩质量 - 饱和度、色彩丰富度
    6. 结构质量 - 边缘密度、纹理分析
    7. 压缩伪影 - 块效应、振铃效应检测
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化质量评估器
        
        Args:
            config: 配置参数字典
        """
        # 默认配置
        self.config = {
            # 模糊度检测参数
            'blur_laplacian_threshold': 100.0,      # Laplacian方差阈值
            'blur_tenengrad_threshold': 50.0,       # Tenengrad阈值
            'blur_fft_threshold': 0.1,              # FFT频域分析阈值
            
            # 对比度参数
            'contrast_rms_threshold': 30.0,         # RMS对比度阈值
            'contrast_michelson_threshold': 0.3,    # Michelson对比度阈值
            
            # 亮度参数
            'brightness_min_threshold': 20,         # 最小亮度阈值
            'brightness_max_threshold': 235,        # 最大亮度阈值
            'brightness_optimal_range': (50, 200),  # 最佳亮度范围
            
            # 信噪比参数
            'noise_estimation_method': 'laplacian', # 噪声估计方法
            'snr_threshold': 15.0,                  # 信噪比阈值(dB)
            
            # 色彩参数
            'saturation_threshold': 0.1,            # 饱和度阈值
            'color_diversity_threshold': 50,        # 色彩多样性阈值
            
            # 结构参数
            'edge_density_threshold': 0.05,         # 边缘密度阈值
            'texture_energy_threshold': 0.01,       # 纹理能量阈值
            
            # 压缩伪影参数
            'blocking_threshold': 5.0,              # 块效应阈值
            'ringing_threshold': 10.0,              # 振铃效应阈值
            
            # 采样参数
            'sample_frames': 30,                    # 采样帧数
            'skip_frames': 1,                       # 跳帧间隔
            
            # 质量评分参数
            'quality_weights': {                    # 各指标权重
                'blur': 0.25,
                'contrast': 0.15,
                'brightness': 0.15,
                'noise': 0.20,
                'color': 0.10,
                'structure': 0.10,
                'artifact': 0.05
            },
            'overall_threshold': 0.6,               # 整体质量阈值
            
            'debug_mode': False                     # 调试模式
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 检测结果存储
        self.frame_qualities = []
        self.video_quality = {}
        
        # 设置日志
        self._setup_logging()
        
        # 初始化计算缓存
        self._init_computation_cache()
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO if not self.config['debug_mode'] else logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_computation_cache(self):
        """初始化计算缓存"""
        self.cache = {
            'sobel_kernels': {
                'x': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
                'y': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            }
        }
    
    def assess_blur_laplacian(self, image: np.ndarray) -> Dict[str, float]:
        """
        基于Laplacian算子的模糊度检测
        
        Args:
            image: 输入图像 (灰度或彩色)
            
        Returns:
            blur_metrics: 模糊度指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Laplacian方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 归一化得分 (0-1, 1表示清晰)
        blur_score = min(1.0, laplacian_var / self.config['blur_laplacian_threshold'])
        
        return {
            'laplacian_variance': float(laplacian_var),
            'blur_score_laplacian': float(blur_score),
            'is_blurry_laplacian': laplacian_var < self.config['blur_laplacian_threshold']
        }
    
    def assess_blur_tenengrad(self, image: np.ndarray) -> Dict[str, float]:
        """
        基于Tenengrad算子的模糊度检测
        
        Args:
            image: 输入图像
            
        Returns:
            blur_metrics: 模糊度指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Sobel梯度
        sobelx = cv2.filter2D(gray, cv2.CV_64F, self.cache['sobel_kernels']['x'])
        sobely = cv2.filter2D(gray, cv2.CV_64F, self.cache['sobel_kernels']['y'])
        
        # Tenengrad测度
        tenengrad = np.sqrt(sobelx**2 + sobely**2)
        tenengrad_mean = np.mean(tenengrad)
        
        # 归一化得分
        blur_score = min(1.0, tenengrad_mean / self.config['blur_tenengrad_threshold'])
        
        return {
            'tenengrad_mean': float(tenengrad_mean),
            'blur_score_tenengrad': float(blur_score),
            'is_blurry_tenengrad': tenengrad_mean < self.config['blur_tenengrad_threshold']
        }
    
    def assess_blur_fft(self, image: np.ndarray) -> Dict[str, float]:
        """
        基于FFT频域分析的模糊度检测
        
        Args:
            image: 输入图像
            
        Returns:
            blur_metrics: 模糊度指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # FFT变换
        fft = fft2(gray)
        fft_shifted = fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # 计算高频能量比例
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # 定义高频区域（外围区域）
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        high_freq_mask = distance_from_center > min(h, w) // 4
        
        # 计算高频能量比例
        total_energy = np.sum(magnitude_spectrum**2)
        high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask]**2)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # 归一化得分
        blur_score = min(1.0, high_freq_ratio / self.config['blur_fft_threshold'])
        
        return {
            'high_freq_ratio': float(high_freq_ratio),
            'blur_score_fft': float(blur_score),
            'is_blurry_fft': high_freq_ratio < self.config['blur_fft_threshold']
        }
    
    def assess_contrast(self, image: np.ndarray) -> Dict[str, float]:
        """
        对比度分析
        
        Args:
            image: 输入图像
            
        Returns:
            contrast_metrics: 对比度指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # RMS对比度
        mean_intensity = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_intensity)**2))
        
        # Michelson对比度
        max_intensity = np.max(gray)
        min_intensity = np.min(gray)
        michelson_contrast = (max_intensity - min_intensity) / (max_intensity + min_intensity) if (max_intensity + min_intensity) > 0 else 0
        
        # 标准差对比度
        std_contrast = np.std(gray)
        
        # 对比度评分 (0-1)
        rms_score = min(1.0, rms_contrast / self.config['contrast_rms_threshold'])
        michelson_score = min(1.0, michelson_contrast / self.config['contrast_michelson_threshold'])
        
        return {
            'rms_contrast': float(rms_contrast),
            'michelson_contrast': float(michelson_contrast),
            'std_contrast': float(std_contrast),
            'contrast_score_rms': float(rms_score),
            'contrast_score_michelson': float(michelson_score),
            'is_low_contrast': rms_contrast < self.config['contrast_rms_threshold']
        }
    
    def assess_brightness(self, image: np.ndarray) -> Dict[str, float]:
        """
        亮度分析
        
        Args:
            image: 输入图像
            
        Returns:
            brightness_metrics: 亮度指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 基础亮度统计
        mean_brightness = np.mean(gray)
        median_brightness = np.median(gray)
        std_brightness = np.std(gray)
        
        # 亮度分布分析
        hist, bins = np.histogram(gray, bins=256, range=(0, 256))
        
        # 过暗和过亮像素比例
        dark_pixels_ratio = np.sum(gray < self.config['brightness_min_threshold']) / gray.size
        bright_pixels_ratio = np.sum(gray > self.config['brightness_max_threshold']) / gray.size
        
        # 最佳亮度范围内的像素比例
        min_optimal, max_optimal = self.config['brightness_optimal_range']
        optimal_pixels_ratio = np.sum((gray >= min_optimal) & (gray <= max_optimal)) / gray.size
        
        # 亮度评分
        brightness_score = optimal_pixels_ratio * (1 - dark_pixels_ratio - bright_pixels_ratio)
        
        return {
            'mean_brightness': float(mean_brightness),
            'median_brightness': float(median_brightness),
            'std_brightness': float(std_brightness),
            'dark_pixels_ratio': float(dark_pixels_ratio),
            'bright_pixels_ratio': float(bright_pixels_ratio),
            'optimal_pixels_ratio': float(optimal_pixels_ratio),
            'brightness_score': float(brightness_score),
            'is_too_dark': mean_brightness < self.config['brightness_min_threshold'],
            'is_too_bright': mean_brightness > self.config['brightness_max_threshold']
        }
    
    def assess_noise(self, image: np.ndarray) -> Dict[str, float]:
        """
        噪声和信噪比分析
        
        Args:
            image: 输入图像
            
        Returns:
            noise_metrics: 噪声指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 方法1: 基于Laplacian的噪声估计
        if self.config['noise_estimation_method'] == 'laplacian':
            # 使用Laplacian算子估计噪声
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_var = np.var(laplacian)
            noise_std = np.sqrt(noise_var)
        
        # 方法2: 基于小波变换的噪声估计
        elif self.config['noise_estimation_method'] == 'wavelet':
            # 简化的小波噪声估计
            # 使用高通滤波器近似小波细节
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            noise_std = np.std(filtered) / 6.0  # 经验系数
            noise_var = noise_std**2
        
        # 方法3: 基于邻域方差的噪声估计
        else:  # 'variance'
            # 计算局部方差
            mean_filter = cv2.blur(gray, (3, 3))
            variance = (gray.astype(float) - mean_filter)**2
            noise_var = np.mean(variance)
            noise_std = np.sqrt(noise_var)
        
        # 信号强度 (图像的平均强度)
        signal_strength = np.mean(gray)
        
        # 信噪比计算 (dB)
        if noise_std > 0:
            snr_db = 20 * np.log10(signal_strength / noise_std)
        else:
            snr_db = 100.0  # 无噪声情况
        
        # PSNR计算 (相对于最大可能信号)
        max_signal = 255.0
        if noise_std > 0:
            psnr_db = 20 * np.log10(max_signal / noise_std)
        else:
            psnr_db = 100.0
        
        # 噪声评分
        snr_score = min(1.0, max(0.0, (snr_db - 5) / (self.config['snr_threshold'] - 5)))
        
        return {
            'noise_variance': float(noise_var),
            'noise_std': float(noise_std),
            'signal_strength': float(signal_strength),
            'snr_db': float(snr_db),
            'psnr_db': float(psnr_db),
            'noise_score': float(snr_score),
            'is_noisy': snr_db < self.config['snr_threshold']
        }
    
    def assess_color_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        色彩质量分析
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            color_metrics: 色彩指标字典
        """
        if len(image.shape) != 3:
            return {
                'saturation_mean': 0.0,
                'saturation_std': 0.0,
                'color_diversity': 0.0,
                'color_score': 0.0,
                'is_desaturated': True
            }
        
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 饱和度分析
        saturation = hsv[:, :, 1] / 255.0
        saturation_mean = np.mean(saturation)
        saturation_std = np.std(saturation)
        
        # 色彩多样性 (色调分布的均匀性)
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=18, range=(0, 180))  # 18个色调区间
        hue_hist_norm = hue_hist / np.sum(hue_hist)
        
        # 计算熵作为色彩多样性指标
        epsilon = 1e-10
        color_diversity = -np.sum(hue_hist_norm * np.log2(hue_hist_norm + epsilon))
        
        # 色彩丰富度 (非灰色像素的比例)
        color_pixels = np.sum(saturation > 0.1)
        color_richness = color_pixels / saturation.size
        
        # 色彩评分
        saturation_score = min(1.0, saturation_mean / self.config['saturation_threshold'])
        diversity_score = min(1.0, color_diversity / 4.0)  # 最大熵约为4.17
        color_score = (saturation_score + diversity_score + color_richness) / 3.0
        
        return {
            'saturation_mean': float(saturation_mean),
            'saturation_std': float(saturation_std),
            'color_diversity': float(color_diversity),
            'color_richness': float(color_richness),
            'color_score': float(color_score),
            'is_desaturated': saturation_mean < self.config['saturation_threshold']
        }
    
    def assess_structure_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        结构质量分析 (边缘密度、纹理)
        
        Args:
            image: 输入图像
            
        Returns:
            structure_metrics: 结构指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 纹理分析 - 灰度共生矩阵的简化版本
        # 计算局部二进制模式 (LBP) 的近似
        def local_binary_pattern_simple(img, radius=1):
            """简化的LBP计算"""
            rows, cols = img.shape
            lbp = np.zeros_like(img, dtype=np.uint8)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = img[i, j]
                    pattern = 0
                    # 8邻域
                    neighbors = [
                        img[i-1, j-1], img[i-1, j], img[i-1, j+1],
                        img[i, j+1], img[i+1, j+1], img[i+1, j],
                        img[i+1, j-1], img[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            pattern |= (1 << k)
                    
                    lbp[i, j] = pattern
            
            return lbp
        
        # 计算纹理能量
        try:
            lbp = local_binary_pattern_simple(gray)
            lbp_hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            lbp_hist_norm = lbp_hist / np.sum(lbp_hist)
            texture_energy = np.sum(lbp_hist_norm**2)
        except:
            # 如果LBP计算失败，使用方差作为纹理测度
            texture_energy = np.var(gray) / (255**2)
        
        # 梯度幅值统计
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude)
        
        # 结构评分
        edge_score = min(1.0, edge_density / self.config['edge_density_threshold'])
        texture_score = min(1.0, texture_energy / self.config['texture_energy_threshold'])
        structure_score = (edge_score + texture_score) / 2.0
        
        return {
            'edge_density': float(edge_density),
            'texture_energy': float(texture_energy),
            'gradient_mean': float(gradient_mean),
            'structure_score': float(structure_score),
            'is_low_structure': edge_density < self.config['edge_density_threshold']
        }
    
    def assess_compression_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        压缩伪影检测 (块效应、振铃效应)
        
        Args:
            image: 输入图像
            
        Returns:
            artifact_metrics: 伪影指标字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 块效应检测
        def detect_blocking_artifacts(img):
            """检测JPEG块效应"""
            h, w = img.shape
            blocking_score = 0.0
            count = 0
            
            # 检测8x8块边界的不连续性
            for i in range(8, h, 8):
                if i < h - 1:
                    # 水平边界
                    diff = np.abs(img[i, :].astype(float) - img[i-1, :].astype(float))
                    blocking_score += np.mean(diff)
                    count += 1
            
            for j in range(8, w, 8):
                if j < w - 1:
                    # 垂直边界
                    diff = np.abs(img[:, j].astype(float) - img[:, j-1].astype(float))
                    blocking_score += np.mean(diff)
                    count += 1
            
            return blocking_score / count if count > 0 else 0.0
        
        # 振铃效应检测
        def detect_ringing_artifacts(img):
            """检测振铃效应"""
            # 使用Laplacian检测器检测高频振荡
            laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
            
            # 计算Laplacian的方差，高方差可能表示振铃
            ringing_score = np.var(laplacian)
            
            return ringing_score
        
        blocking_artifacts = detect_blocking_artifacts(gray)
        ringing_artifacts = detect_ringing_artifacts(gray)
        
        # 伪影评分 (越低越好)
        blocking_score = max(0.0, 1.0 - blocking_artifacts / self.config['blocking_threshold'])
        ringing_score = max(0.0, 1.0 - ringing_artifacts / self.config['ringing_threshold'])
        artifact_score = (blocking_score + ringing_score) / 2.0
        
        return {
            'blocking_artifacts': float(blocking_artifacts),
            'ringing_artifacts': float(ringing_artifacts),
            'artifact_score': float(artifact_score),
            'has_blocking': blocking_artifacts > self.config['blocking_threshold'],
            'has_ringing': ringing_artifacts > self.config['ringing_threshold']
        }
    
    def assess_frame_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        评估单帧质量
        
        Args:
            frame: 输入帧
            
        Returns:
            quality_metrics: 质量指标字典
        """
        metrics = {}
        
        try:
            # 模糊度检测
            blur_laplacian = self.assess_blur_laplacian(frame)
            blur_tenengrad = self.assess_blur_tenengrad(frame)
            blur_fft = self.assess_blur_fft(frame)
            
            # 综合模糊度评分
            blur_scores = [
                blur_laplacian['blur_score_laplacian'],
                blur_tenengrad['blur_score_tenengrad'],
                blur_fft['blur_score_fft']
            ]
            blur_score = np.mean(blur_scores)
            
            metrics['blur'] = {
                **blur_laplacian,
                **blur_tenengrad,
                **blur_fft,
                'blur_score_combined': float(blur_score)
            }
            
            # 对比度分析
            contrast_metrics = self.assess_contrast(frame)
            metrics['contrast'] = contrast_metrics
            
            # 亮度分析
            brightness_metrics = self.assess_brightness(frame)
            metrics['brightness'] = brightness_metrics
            
            # 噪声分析
            noise_metrics = self.assess_noise(frame)
            metrics['noise'] = noise_metrics
            
            # 色彩质量
            color_metrics = self.assess_color_quality(frame)
            metrics['color'] = color_metrics
            
            # 结构质量
            structure_metrics = self.assess_structure_quality(frame)
            metrics['structure'] = structure_metrics
            
            # 压缩伪影
            artifact_metrics = self.assess_compression_artifacts(frame)
            metrics['artifacts'] = artifact_metrics
            
            # 计算综合质量评分
            weights = self.config['quality_weights']
            quality_components = {
                'blur': blur_score,
                'contrast': (contrast_metrics['contrast_score_rms'] + contrast_metrics['contrast_score_michelson']) / 2,
                'brightness': brightness_metrics['brightness_score'],
                'noise': noise_metrics['noise_score'],
                'color': color_metrics['color_score'],
                'structure': structure_metrics['structure_score'],
                'artifact': artifact_metrics['artifact_score']
            }
            
            overall_score = sum(weights[k] * quality_components[k] for k in weights.keys())
            
            metrics['overall'] = {
                'quality_score': float(overall_score),
                'quality_components': quality_components,
                'is_high_quality': overall_score >= self.config['overall_threshold'],
                'quality_grade': self._get_quality_grade(overall_score)
            }
            
        except Exception as e:
            self.logger.error(f"Frame quality assessment failed: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _get_quality_grade(self, score: float) -> str:
        """根据质量评分获取质量等级"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.6:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def assess_video_quality(self, video_path: str) -> Dict[str, Any]:
        """
        评估整个视频的质量
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            video_quality_result: 视频质量评估结果
        """
        self.logger.info(f"开始评估视频质量: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        self.logger.info(f"视频信息: {width}x{height}, {fps}fps, {frame_count}帧, {duration:.2f}秒")
        
        # 确定采样策略
        sample_frames = min(self.config['sample_frames'], frame_count)
        if sample_frames < frame_count:
            # 均匀采样
            frame_indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)
        else:
            # 使用所有帧
            frame_indices = list(range(0, frame_count, self.config['skip_frames']))
        
        self.logger.info(f"将评估 {len(frame_indices)} 帧 (总帧数: {frame_count})")
        
        # 评估每一帧
        frame_qualities = []
        
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="评估帧质量")):
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                self.logger.warning(f"无法读取帧 {frame_idx}")
                continue
            
            # 评估帧质量
            frame_quality = self.assess_frame_quality(frame)
            frame_quality['frame_index'] = frame_idx
            frame_quality['timestamp'] = frame_idx / fps if fps > 0 else 0
            
            frame_qualities.append(frame_quality)
        
        cap.release()
        
        if not frame_qualities:
            raise ValueError("未能成功评估任何帧")
        
        # 汇总视频质量
        video_quality_summary = self._summarize_video_quality(frame_qualities)
        
        # 构建完整结果
        video_quality_result = {
            'video_path': video_path,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'assessed_frames': len(frame_qualities)
            },
            'quality_summary': video_quality_summary,
            'frame_qualities': frame_qualities,
            'recommendation': self._get_quality_recommendation(video_quality_summary)
        }
        
        self.video_quality = video_quality_result
        self.frame_qualities = frame_qualities
        
        self.logger.info(f"视频质量评估完成. 整体评分: {video_quality_summary['overall_score']:.3f}")
        
        return video_quality_result
    
    def _summarize_video_quality(self, frame_qualities: List[Dict]) -> Dict[str, Any]:
        """
        汇总视频质量统计
        
        Args:
            frame_qualities: 帧质量列表
            
        Returns:
            summary: 质量汇总统计
        """
        # 提取各项指标
        overall_scores = [fq['overall']['quality_score'] for fq in frame_qualities if 'overall' in fq]
        
        if not overall_scores:
            return {'error': 'No valid quality scores'}
        
        # 提取各维度评分
        dimensions = ['blur', 'contrast', 'brightness', 'noise', 'color', 'structure', 'artifact']
        dimension_scores = {dim: [] for dim in dimensions}
        
        for fq in frame_qualities:
            if 'overall' in fq and 'quality_components' in fq['overall']:
                components = fq['overall']['quality_components']
                for dim in dimensions:
                    if dim in components:
                        dimension_scores[dim].append(components[dim])
        
        # 计算统计信息
        def calc_stats(values):
            if not values:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # 整体质量统计
        overall_stats = calc_stats(overall_scores)
        
        # 各维度统计
        dimension_stats = {dim: calc_stats(scores) for dim, scores in dimension_scores.items()}
        
        # 质量等级分布
        quality_grades = [fq['overall']['quality_grade'] for fq in frame_qualities if 'overall' in fq]
        grade_distribution = {}
        for grade in ['Excellent', 'Good', 'Fair', 'Poor', 'Very Poor']:
            grade_distribution[grade] = quality_grades.count(grade)
        
        # 问题帧统计
        problem_frames = {
            'blurry_frames': len([fq for fq in frame_qualities if fq.get('blur', {}).get('blur_score_combined', 1.0) < 0.5]),
            'low_contrast_frames': len([fq for fq in frame_qualities if fq.get('contrast', {}).get('is_low_contrast', False)]),
            'noisy_frames': len([fq for fq in frame_qualities if fq.get('noise', {}).get('is_noisy', False)]),
            'dark_frames': len([fq for fq in frame_qualities if fq.get('brightness', {}).get('is_too_dark', False)]),
            'bright_frames': len([fq for fq in frame_qualities if fq.get('brightness', {}).get('is_too_bright', False)]),
            'desaturated_frames': len([fq for fq in frame_qualities if fq.get('color', {}).get('is_desaturated', False)])
        }
        
        # 视频质量稳定性分析
        stability_score = 1.0 - min(1.0, overall_stats['std'])  # 标准差越小越稳定
        
        return {
            'overall_score': overall_stats['mean'],
            'overall_stats': overall_stats,
            'dimension_stats': dimension_stats,
            'quality_grade': self._get_quality_grade(overall_stats['mean']),
            'grade_distribution': grade_distribution,
            'problem_frames': problem_frames,
            'stability_score': stability_score,
            'is_high_quality': overall_stats['mean'] >= self.config['overall_threshold'],
            'total_frames_assessed': len(frame_qualities)
        }
    
    def _get_quality_recommendation(self, quality_summary: Dict) -> Dict[str, Any]:
        """
        基于质量分析结果给出建议
        
        Args:
            quality_summary: 质量汇总
            
        Returns:
            recommendation: 质量改进建议
        """
        recommendations = []
        issues = []
        
        overall_score = quality_summary.get('overall_score', 0.0)
        dimension_stats = quality_summary.get('dimension_stats', {})
        problem_frames = quality_summary.get('problem_frames', {})
        
        # 检查各维度问题
        if dimension_stats.get('blur', {}).get('mean', 1.0) < 0.6:
            issues.append('模糊度问题')
            recommendations.append('建议: 检查拍摄时的对焦设置，避免相机抖动，使用更高的快门速度')
        
        if dimension_stats.get('contrast', {}).get('mean', 1.0) < 0.5:
            issues.append('对比度不足')
            recommendations.append('建议: 调整拍摄时的光照条件，后期可进行对比度增强')
        
        if dimension_stats.get('noise', {}).get('mean', 1.0) < 0.5:
            issues.append('噪声较高')
            recommendations.append('建议: 降低ISO设置，改善拍摄环境光照，使用降噪算法处理')
        
        if dimension_stats.get('brightness', {}).get('mean', 1.0) < 0.5:
            issues.append('亮度不佳')
            recommendations.append('建议: 调整曝光设置，确保充足的光照条件')
        
        if dimension_stats.get('color', {}).get('mean', 1.0) < 0.5:
            issues.append('色彩质量问题')
            recommendations.append('建议: 调整白平衡设置，提高色彩饱和度')
        
        # 整体建议
        if overall_score >= 0.8:
            action = 'ACCEPT'
            message = '视频质量优良，建议保留'
        elif overall_score >= 0.6:
            action = 'REVIEW'
            message = '视频质量一般，建议人工审查'
        else:
            action = 'REJECT'
            message = '视频质量较差，建议剔除或重新处理'
        
        return {
            'action': action,
            'message': message,
            'overall_score': overall_score,
            'issues_found': issues,
            'recommendations': recommendations,
            'problem_frame_ratio': sum(problem_frames.values()) / quality_summary.get('total_frames_assessed', 1)
        }
    
    def save_results(self, result: Dict, output_path: str):
        """
        保存质量评估结果
        
        Args:
            result: 评估结果
            output_path: 输出文件路径
        """
        # 处理numpy类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        converted_result = convert_numpy(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"质量评估结果已保存到: {output_path}")
    
    def generate_quality_report(self, result: Dict, output_path: str):
        """
        生成质量评估报告
        
        Args:
            result: 评估结果
            output_path: 报告输出路径
        """
        video_info = result['video_info']
        quality_summary = result['quality_summary']
        recommendation = result['recommendation']
        
        report = f"""
# 视频质量评估报告

## 视频信息
- **文件路径**: {result['video_path']}
- **分辨率**: {video_info['width']}x{video_info['height']}
- **帧率**: {video_info['fps']:.2f} fps
- **总帧数**: {video_info['frame_count']}
- **时长**: {video_info['duration']:.2f} 秒
- **评估帧数**: {video_info['assessed_frames']}

## 质量评估结果

### 整体质量
- **综合评分**: {quality_summary['overall_score']:.3f} / 1.0
- **质量等级**: {quality_summary['quality_grade']}
- **稳定性评分**: {quality_summary['stability_score']:.3f}
- **是否高质量**: {'是' if quality_summary['is_high_quality'] else '否'}

### 质量等级分布
"""
        
        for grade, count in quality_summary['grade_distribution'].items():
            percentage = count / quality_summary['total_frames_assessed'] * 100
            report += f"- **{grade}**: {count} 帧 ({percentage:.1f}%)\n"
        
        report += "\n### 各维度质量评分\n"
        
        for dimension, stats in quality_summary['dimension_stats'].items():
            report += f"""
**{dimension.upper()}**:
- 平均分: {stats['mean']:.3f}
- 标准差: {stats['std']:.3f}
- 最低分: {stats['min']:.3f}
- 最高分: {stats['max']:.3f}
"""
        
        report += "\n### 问题帧统计\n"
        
        problem_frames = quality_summary['problem_frames']
        total_frames = quality_summary['total_frames_assessed']
        
        for problem_type, count in problem_frames.items():
            percentage = count / total_frames * 100
            report += f"- **{problem_type.replace('_', ' ').title()}**: {count} 帧 ({percentage:.1f}%)\n"
        
        report += f"\n## 质量建议\n\n"
        report += f"**推荐操作**: {recommendation['action']}\n\n"
        report += f"**建议**: {recommendation['message']}\n\n"
        
        if recommendation['issues_found']:
            report += "### 发现的质量问题:\n"
            for issue in recommendation['issues_found']:
                report += f"- {issue}\n"
            
            report += "\n### 改进建议:\n"
            for rec in recommendation['recommendations']:
                report += f"- {rec}\n"
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"质量评估报告已保存到: {output_path}")
    
    def visualize_quality_metrics(self, result: Dict, output_dir: str):
        """
        生成质量指标可视化图表
        
        Args:
            result: 评估结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        frame_qualities = result['frame_qualities']
        if not frame_qualities:
            self.logger.warning("没有有效的帧质量数据用于可视化")
            return
        
        # 提取时间轴和各项指标
        timestamps = [fq['timestamp'] for fq in frame_qualities]
        overall_scores = [fq['overall']['quality_score'] for fq in frame_qualities if 'overall' in fq]
        
        # 1. 整体质量随时间变化
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, overall_scores, 'b-', linewidth=2, label='Overall Quality')
        plt.axhline(y=self.config['overall_threshold'], color='r', linestyle='--', 
                   label=f'Threshold ({self.config["overall_threshold"]})')
        plt.xlabel('时间 (秒)')
        plt.ylabel('质量评分')
        plt.title('视频质量随时间变化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quality_timeline.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 各维度质量对比
        dimensions = ['blur', 'contrast', 'brightness', 'noise', 'color', 'structure', 'artifact']
        dimension_scores = {dim: [] for dim in dimensions}
        
        for fq in frame_qualities:
            if 'overall' in fq and 'quality_components' in fq['overall']:
                components = fq['overall']['quality_components']
                for dim in dimensions:
                    if dim in components:
                        dimension_scores[dim].append(components[dim])
        
        # 箱线图
        plt.figure(figsize=(12, 8))
        data_for_box = [dimension_scores[dim] for dim in dimensions if dimension_scores[dim]]
        labels_for_box = [dim.title() for dim in dimensions if dimension_scores[dim]]
        
        if data_for_box:
            plt.boxplot(data_for_box, labels=labels_for_box)
            plt.ylabel('质量评分')
            plt.title('各维度质量分布')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_dimensions.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 3. 质量等级分布饼图
        grade_distribution = result['quality_summary']['grade_distribution']
        grades = list(grade_distribution.keys())
        counts = list(grade_distribution.values())
        
        # 只显示非零的等级
        non_zero_grades = [(grade, count) for grade, count in zip(grades, counts) if count > 0]
        
        if non_zero_grades:
            grades, counts = zip(*non_zero_grades)
            
            plt.figure(figsize=(8, 8))
            colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF6347', '#DC143C'][:len(grades)]
            plt.pie(counts, labels=grades, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('质量等级分布')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'quality_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 4. 问题帧统计柱状图
        problem_frames = result['quality_summary']['problem_frames']
        problem_types = list(problem_frames.keys())
        problem_counts = list(problem_frames.values())
        
        if any(problem_counts):
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(problem_types)), problem_counts, 
                          color=['red' if count > 0 else 'lightgray' for count in problem_counts])
            
            # 添加数值标签
            for i, (bar, count) in enumerate(zip(bars, problem_counts)):
                if count > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            str(count), ha='center', va='bottom')
            
            plt.xlabel('问题类型')
            plt.ylabel('帧数')
            plt.title('问题帧统计')
            plt.xticks(range(len(problem_types)), 
                      [ptype.replace('_', ' ').title() for ptype in problem_types], 
                      rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'problem_frames.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"质量可视化图表已保存到: {output_dir}")


def main():
    """
    命令行主函数
    """
    parser = argparse.ArgumentParser(description='视频质量检测与评估系统')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output', '-o', type=str, default='./quality_assessment_output', 
                       help='输出目录')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径(JSON格式)')
    parser.add_argument('--threshold', '-t', type=float, default=0.6, 
                       help='质量阈值')
    parser.add_argument('--sample-frames', '-s', type=int, default=30, 
                       help='采样帧数')
    parser.add_argument('--visualize', '-v', action='store_true', 
                       help='生成可视化结果')
    parser.add_argument('--report', '-r', action='store_true', 
                       help='生成质量报告')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if not config:
        config = {}
    
    # 更新配置
    config['overall_threshold'] = args.threshold
    config['sample_frames'] = args.sample_frames
    config['debug_mode'] = args.debug
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建质量评估器
    assessor = VideoQualityAssessor(config)
    
    try:
        # 执行质量评估
        result = assessor.assess_video_quality(args.input)
        
        # 保存结果
        result_path = output_dir / 'quality_assessment_result.json'
        assessor.save_results(result, str(result_path))
        
        # 生成报告
        if args.report:
            report_path = output_dir / 'quality_assessment_report.md'
            assessor.generate_quality_report(result, str(report_path))
        
        # 生成可视化
        if args.visualize:
            vis_dir = output_dir / 'visualizations'
            assessor.visualize_quality_metrics(result, str(vis_dir))
        
        # 输出简要结果
        quality_summary = result['quality_summary']
        recommendation = result['recommendation']
        
        print(f"\n质量评估完成!")
        print(f"视频路径: {args.input}")
        print(f"综合质量评分: {quality_summary['overall_score']:.3f}")
        print(f"质量等级: {quality_summary['quality_grade']}")
        print(f"推荐操作: {recommendation['action']}")
        print(f"建议: {recommendation['message']}")
        print(f"结果已保存到: {output_dir}")
        
        # 如果是低质量视频，显示问题详情
        if not quality_summary['is_high_quality']:
            print("\n检测到的质量问题:")
            for issue in recommendation['issues_found']:
                print(f"  - {issue}")
        
    except Exception as e:
        print(f"质量评估过程中出现错误: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

