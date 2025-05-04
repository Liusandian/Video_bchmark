import cv2
import numpy as np
import argparse

def join_videos_side_by_side(video1_path, video2_path, output_path):
    # 打开两个视频文件
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # 检查视频是否成功打开
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one or both videos")
        return
    
    # 获取第一个视频的属性
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    
    # 获取第二个视频的属性
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算输出视频的尺寸
    # 使用两个视频中较大的高度
    output_height = max(height1, height2)
    output_width = width1 + width2
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps1, (output_width, output_height))
    
    while True:
        # 读取两个视频的帧
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # 如果任一视频结束，则退出循环
        if not ret1 or not ret2:
            break
        
        # 调整两个帧的大小以匹配输出高度
        frame1 = cv2.resize(frame1, (width1, output_height))
        frame2 = cv2.resize(frame2, (width2, output_height))
        
        # 水平拼接两个帧
        combined_frame = np.hstack((frame1, frame2))
        
        # 写入输出视频
        out.write(combined_frame)
    
    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Join two videos side by side')
    # parser.add_argument('video1', help='Path to first video')
    # parser.add_argument('video2', help='Path to second video')
    # parser.add_argument('output', help='Path to output video')
    
    # args = parser.parse_args()
    # path 1/2 is D:\00-Media\01-相册摄影\视频
    path1 = f'D:\\00-Media\\01-相册摄影\\视频\\VID_20250501_154220.mp4'
    path2 = path1.replace('VID_20250501_154220','VID_20250501_164835')

    output = f'output.mp4'
    
    
    # join_videos_side_by_side(args.video1, args.video2, args.output)
    join_videos_side_by_side(path1,path2,output)
    print("Video processing completed!")
