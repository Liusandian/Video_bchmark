#!/usr/bin/env python3
"""
Video merging utilities for FL2V 720P model
包含处理torch tensor视频流与MP4视频合并的功能
"""

import torch
import numpy as np
import moviepy.editor as mp
from PIL import Image
import tempfile
import os

def merge_videos_with_tensor_transition(video_start_path, video_end_path, tensor_samples, output_path, fps=16):
    """
    Merge two MP4 videos with a torch tensor video stream as transition in the middle.
    将2个MP4视频与中间的torch tensor视频流合并成一个完整视频
    
    Args:
        video_start_path (str): Path to the first MP4 video (第一个MP4视频路径)
        video_end_path (str): Path to the second MP4 video (第二个MP4视频路径)
        tensor_samples (torch.Tensor): Torch tensor video data from wan_model.generate() (模型生成的torch tensor视频数据)
        output_path (str): Output path for the combined video (合并后视频的输出路径)
        fps (int): Frame rate for the tensor video (tensor视频的帧率，默认16)
        
    Returns:
        str: Path to the combined video file, or None if failed (成功返回合并视频路径，失败返回None)
    """
    try:
        print(f"开始合并视频: {video_start_path} + tensor + {video_end_path}")
        
        # Step 1: Convert torch tensor to CPU and numpy
        # 第一步：将torch tensor转换到CPU并转为numpy格式
        if isinstance(tensor_samples, torch.Tensor):
            print(f"检测到torch.Tensor，原始设备: {'GPU' if tensor_samples.is_cuda else 'CPU'}")
            # Move to CPU if on GPU
            if tensor_samples.is_cuda:
                tensor_samples = tensor_samples.cpu()
                print("已将tensor从GPU移动到CPU")
            
            # Convert to numpy
            tensor_np = tensor_samples.numpy()
            print("已将tensor转换为numpy数组")
        else:
            tensor_np = tensor_samples
            print("输入已经是numpy数组格式")
            
        # Step 2: Handle different tensor formats and convert to [Frames, Height, Width, Channels]
        # 第二步：处理不同的tensor格式，转换为[帧数, 高度, 宽度, 通道数]格式
        print(f"原始tensor形状: {tensor_np.shape}")
        
        # Common tensor formats from video generation models:
        # 常见的视频生成模型tensor格式：
        # [Batch, Channels, Frames, Height, Width] -> [Frames, Height, Width, Channels]
        # [Batch, Frames, Channels, Height, Width] -> [Frames, Height, Width, Channels]  
        # [Frames, Channels, Height, Width] -> [Frames, Height, Width, Channels]
        
        if len(tensor_np.shape) == 5:  # [Batch, ?, ?, ?, ?]
            # Remove batch dimension (移除batch维度)
            tensor_np = tensor_np[0]
            print("已移除batch维度")
            
        if len(tensor_np.shape) == 4:
            # Check if it's [Channels, Frames, Height, Width] or [Frames, Channels, Height, Width]
            # 检查是[C, F, H, W]还是[F, C, H, W]格式
            if tensor_np.shape[0] <= 4:  # Likely channels first: [C, F, H, W]
                tensor_np = tensor_np.transpose(1, 2, 3, 0)  # [F, H, W, C]
                print("已从[C, F, H, W]转换为[F, H, W, C]")
            elif tensor_np.shape[1] <= 4:  # Likely [F, C, H, W]
                tensor_np = tensor_np.transpose(0, 2, 3, 1)  # [F, H, W, C]
                print("已从[F, C, H, W]转换为[F, H, W, C]")
            # If neither dimension is <= 4, assume it's already [F, H, W, C]
            else:
                print("假设已经是[F, H, W, C]格式")
            
        print(f"转换后tensor形状: {tensor_np.shape}")
        
        # Step 3: Normalize tensor values to [0, 255] range
        # 第三步：将tensor值归一化到[0, 255]范围
        print(f"Tensor值范围: {tensor_np.min():.3f} 到 {tensor_np.max():.3f}")
        
        if tensor_np.max() <= 1.0 and tensor_np.min() >= -1.0:
            # Assume range is [-1, 1] or [0, 1]
            if tensor_np.min() < 0:
                # Range [-1, 1] -> [0, 255]
                tensor_np = ((tensor_np + 1.0) * 127.5).astype(np.uint8)
                print("已从[-1, 1]范围转换为[0, 255]")
            else:
                # Range [0, 1] -> [0, 255]
                tensor_np = (tensor_np * 255).astype(np.uint8)
                print("已从[0, 1]范围转换为[0, 255]")
        else:
            # Assume already in [0, 255] range
            tensor_np = np.clip(tensor_np, 0, 255).astype(np.uint8)
            print("已裁剪到[0, 255]范围")
            
        # Step 4: Create temporary video file from tensor
        # 第四步：从tensor创建临时视频文件
        temp_dir = tempfile.mkdtemp()
        temp_tensor_video = os.path.join(temp_dir, "tensor_transition.mp4")
        print(f"创建临时目录: {temp_dir}")
        
        # Convert numpy frames to PIL Images and then to video
        # 将numpy帧转换为PIL图像，然后转为视频
        frames = []
        print(f"开始处理 {tensor_np.shape[0]} 帧图像...")
        
        for i in range(tensor_np.shape[0]):
            frame = tensor_np[i]
            # Ensure frame has 3 channels (RGB)
            # 确保帧有3个通道(RGB)
            if frame.shape[-1] == 1:  # Grayscale (灰度图)
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] == 4:  # RGBA
                frame = frame[:, :, :3]  # Remove alpha channel (移除alpha通道)
                
            pil_frame = Image.fromarray(frame)
            frames.append(pil_frame)
        
        print(f"已处理完所有帧，开始创建视频剪辑...")
        
        # Create video clip from frames
        # 从帧创建视频剪辑
        def make_frame(t):
            frame_idx = min(int(t * fps), len(frames) - 1)
            return np.array(frames[frame_idx])
        
        duration = len(frames) / fps
        print(f"tensor视频时长: {duration:.2f}秒 ({len(frames)}帧 @ {fps}fps)")
        
        tensor_clip = mp.VideoClip(make_frame, duration=duration)
        tensor_clip.fps = fps
        
        # Write tensor video to temporary file
        # 将tensor视频写入临时文件
        print(f"正在将tensor视频写入临时文件: {temp_tensor_video}")
        tensor_clip.write_videofile(temp_tensor_video, codec='libx264', audio=False, verbose=False, logger=None)
        
        # Step 5: Load and combine all three videos
        # 第五步：加载并合并三个视频
        print("正在加载视频文件...")
        start_clip = mp.VideoFileClip(video_start_path)
        transition_clip = mp.VideoFileClip(temp_tensor_video)
        end_clip = mp.VideoFileClip(video_end_path)
        
        print(f"第一段视频时长: {start_clip.duration:.2f}秒")
        print(f"转场视频时长: {transition_clip.duration:.2f}秒")
        print(f"第二段视频时长: {end_clip.duration:.2f}秒")
        
        # Combine videos
        # 合并视频
        print("正在合并视频...")
        final_video = mp.concatenate_videoclips([start_clip, transition_clip, end_clip])
        
        print(f"合并后总时长: {final_video.duration:.2f}秒")
        
        # Write final combined video
        # 写入最终合并的视频
        print(f"正在保存最终视频到: {output_path}")
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Clean up
        # 清理资源
        print("正在清理临时资源...")
        start_clip.close()
        transition_clip.close()
        end_clip.close()
        final_video.close()
        tensor_clip.close()
        
        # Remove temporary files
        # 删除临时文件
        try:
            os.remove(temp_tensor_video)
            os.rmdir(temp_dir)
            print("已清理临时文件")
        except Exception as cleanup_error:
            print(f"清理临时文件时出现警告: {cleanup_error}")
            
        print(f"✅ 视频合并成功完成: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 视频合并过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_transition_video_flf2v(video_start_source, video_end_source, transition_video, output_path):
    """
    Create a combined video by merging start video + transition + end video for FL2V 720P model
    为FL2V 720P模型创建合并视频：开始视频 + 转场视频 + 结束视频
    
    Args:
        video_start_source (str): Path to the first video
        video_end_source (str): Path to the second video
        transition_video (str): Path to the transition video
        output_path (str): Output path for combined video
        
    Returns:
        str: Path to combined video or None if failed
    """
    if not all([video_start_source, video_end_source, transition_video]):
        print("❌ 缺少必要的视频文件路径")
        return None
    
    try:
        print(f"开始合并视频文件...")
        print(f"第一段: {video_start_source}")
        print(f"转场段: {transition_video}")
        print(f"第二段: {video_end_source}")
        
        # Load videos
        start_clip = mp.VideoFileClip(video_start_source)
        transition_clip = mp.VideoFileClip(transition_video)
        end_clip = mp.VideoFileClip(video_end_source)
        
        # Combine all three videos
        final_video = mp.concatenate_videoclips([start_clip, transition_clip, end_clip])
        
        # Save the combined video
        final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Clean up
        start_clip.close()
        transition_clip.close() 
        end_clip.close()
        final_video.close()
        
        print(f"✅ 视频文件合并成功: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ 视频文件合并失败: {e}")
        return None

if __name__ == "__main__":
    # 示例用法
    print("视频合并工具 - 示例用法:")
    print("from video_merge_utils import merge_videos_with_tensor_transition")
    print("result = merge_videos_with_tensor_transition('video1.mp4', 'video2.mp4', tensor_samples, 'output.mp4')") 