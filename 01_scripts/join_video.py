import cv2
import numpy as np
import argparse

def join_videos(video_paths, output_path):
    """
    拼接2-4个视频
    - 2个视频：左右各半
    - 3个视频：上面左右各半，下面居中
    - 4个视频：2x2网格布局
    """
    # 检查视频数量是否合法
    if not 2 <= len(video_paths) <= 4:
        print("Error: Number of videos must be between 2 and 4")
        return

    # 打开所有视频文件
    caps = [cv2.VideoCapture(path) for path in video_paths]
    
    # 检查所有视频是否成功打开
    if not all(cap.isOpened() for cap in caps):
        print("Error: Could not open one or more videos")
        for cap in caps:
            cap.release()
        return

    # 获取所有视频的属性
    video_props = []
    for cap in caps:
        props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }
        video_props.append(props)

    # 使用第一个视频的帧率
    fps = video_props[0]['fps']

    # 根据视频数量决定输出尺寸和布局
    if len(video_paths) == 2:
        # 两个视频左右排列
        output_width = sum(prop['width'] for prop in video_props)
        output_height = max(prop['height'] for prop in video_props)
    elif len(video_paths) == 3:
        # 上面两个左右排列，下面一个居中
        output_width = max(video_props[0]['width'] + video_props[1]['width'],
                         video_props[2]['width'])
        output_height = max(video_props[0]['height'], video_props[1]['height']) + video_props[2]['height']
    else:  # 4个视频
        # 2x2网格布局
        output_width = max(video_props[0]['width'] + video_props[1]['width'],
                         video_props[2]['width'] + video_props[3]['width'])
        output_height = max(video_props[0]['height'] + video_props[2]['height'],
                          video_props[1]['height'] + video_props[3]['height'])

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

    while True:
        frames = []
        # 读取所有视频的帧
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        # 如果任一视频结束，则退出循环
        if len(frames) != len(caps):
            break

        if len(video_paths) == 2:
            # 调整两个帧的大小
            frames[0] = cv2.resize(frames[0], (output_width // 2, output_height))
            frames[1] = cv2.resize(frames[1], (output_width // 2, output_height))
            # 水平拼接
            combined_frame = np.hstack(frames)

        elif len(video_paths) == 3:
            # 调整帧大小
            top_height = output_height // 2
            bottom_height = output_height - top_height
            
            frames[0] = cv2.resize(frames[0], (output_width // 2, top_height))
            frames[1] = cv2.resize(frames[1], (output_width // 2, top_height))
            frames[2] = cv2.resize(frames[2], (output_width, bottom_height))
            
            # 先合并上面两个
            top_frame = np.hstack((frames[0], frames[1]))
            # 然后与下面的拼接
            combined_frame = np.vstack((top_frame, frames[2]))

        else:  # 4个视频
            # 调整所有帧的大小为相同尺寸
            frame_width = output_width // 2
            frame_height = output_height // 2
            
            resized_frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in frames]
            
            # 先合并上下两行
            top_row = np.hstack((resized_frames[0], resized_frames[1]))
            bottom_row = np.hstack((resized_frames[2], resized_frames[3]))
            
            # 再合并两行
            combined_frame = np.vstack((top_row, bottom_row))

        # 写入输出视频
        out.write(combined_frame)

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()

if __name__ == "__main__":
    # 示例视频路径
    path1 = f'D:\\00-Media\\01-相册摄影\\视频\\VID_20250501_154220.mp4'
    path2 = path1.replace('VID_20250501_154220','VID_20250501_164835')
    path3 = path1.replace('VID_20250501_154220','VID_20250501_165536')  # 请替换为实际的第三个视频路径
    path4 = path1.replace('VID_20250501_154220','VID_20250501_164906')    # 请替换为实际的第四个视频路径
    
    output1 = f'output.mp4'
    output2 = f'output_merge4_video.mp4'
    
    # 可以选择传入2-4个视频
    # video_paths = [path1, path2]  # 两个视频
    video_paths = [path1, path2, path3]  # 三个视频
    video_paths2 = [path1, path2, path3, path4]  # 四个视频
    
    join_videos(video_paths, output1)
    print("Video processing completed!")
    join_videos(video_paths2, output2)
