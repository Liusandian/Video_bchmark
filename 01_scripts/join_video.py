import cv2
import numpy as np
import argparse
import math
import os

def join_videos(video_paths, output_path):
    """
    拼接多个视频，支持2,3,4,6,8,9,12,16个视频的不同布局
    - 2个视频：左右各半
    - 3个视频：上面左右各半，下面居中
    - 4个视频：2x2网格布局
    - 6个视频：2x3网格布局 
    - 8个视频：2x4网格布局
    - 9个视频：3x3网格布局
    - 12个视频：3x4网格布局
    - 16个视频：4x4网格布局
    
    在每个视频的左上角显示原始文件名
    """
    # 检查视频数量是否合法
    valid_counts = [2, 3, 4, 6, 8, 9, 12, 16]
    if len(video_paths) not in valid_counts:
        print(f"Error: Number of videos must be one of {valid_counts}")
        return

    # 提取文件名（不含路径和扩展名）
    video_names = []
    for path in video_paths:
        # 获取不含路径的文件名
        full_name = os.path.basename(path)
        # 分离文件名和扩展名
        name_without_ext = os.path.splitext(full_name)[0]
        video_names.append(name_without_ext)
    
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

    # 根据视频数量确定布局（行数和列数）
    num_videos = len(video_paths)
    if num_videos == 2:
        rows, cols = 1, 2  # 1x2 布局
    elif num_videos == 3:
        rows, cols = 2, 2  # 特殊布局：上面两个，下面一个居中
    elif num_videos == 4:
        rows, cols = 2, 2  # 2x2 布局
    elif num_videos == 6:
        rows, cols = 2, 3  # 2x3 布局
    elif num_videos == 8:
        rows, cols = 2, 4  # 2x4 布局
    elif num_videos == 9:
        rows, cols = 3, 3  # 3x3 布局
    elif num_videos == 12:
        rows, cols = 3, 4  # 3x4 布局
    elif num_videos == 16:
        rows, cols = 4, 4  # 4x4 布局

    # 计算单个视频的统一尺寸
    # 采用视频平均宽高的策略确保统一缩放
    avg_width = sum(prop['width'] for prop in video_props) // len(video_props)
    avg_height = sum(prop['height'] for prop in video_props) // len(video_props)
    
    # 保持视频比例，确定每个单元格的尺寸
    if num_videos == 3:
        # 特殊处理3个视频的情况
        cell_width = avg_width
        cell_height = avg_height
        output_width = cell_width * 2  # 上排2个视频
        output_height = cell_height * 2  # 上下各一排
    else:
        cell_width = avg_width
        cell_height = avg_height
        output_width = cell_width * cols
        output_height = cell_height * rows

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    print(f"输出视频尺寸: {output_width}x{output_height}, 布局: {rows}x{cols}")

    # 根据视频尺寸自动调整文本参数
    # 确保文本大小与视频尺寸成比例
    base_font_scale = 1.0  # 增加字体基础大小
    # 对于小尺寸视频，调整字体大小
    font_scale = base_font_scale * (min(cell_width, cell_height) / 500)
    # 限制字体大小不超过上限
    font_scale = min(max(font_scale, 0.8), 1.5)  # 确保最小/最大字体大小
    
    # 设置文本参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = max(2, int(font_scale * 2))  # 增加字体粗细，提高清晰度
    text_color = (255, 255, 255)  # 白色文本
    text_bg_color = (0, 0, 0)  # 黑色背景
    text_padding = max(10, int(font_scale * 8))  # 增加文本周围的填充
    outline_thickness = max(1, int(font_scale * 1.5))  # 文字描边粗细

    frame_count = 0
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

        if num_videos == 2:
            # 调整两个帧的大小为单元格尺寸
            frames[0] = cv2.resize(frames[0], (cell_width, cell_height))
            frames[1] = cv2.resize(frames[1], (cell_width, cell_height))
            
            # 添加文件名到每个视频帧
            for i in range(2):
                add_filename_with_outline(frames[i], video_names[i], font, font_scale, 
                                        font_thickness, text_color, text_bg_color, 
                                        text_padding, outline_thickness)
            
            # 水平拼接
            combined_frame = np.hstack(frames)

        elif num_videos == 3:
            # 特殊处理3个视频：上面两个，下面一个居中
            top_width = cell_width
            
            # 调整三个视频的大小
            frames[0] = cv2.resize(frames[0], (top_width, cell_height))
            frames[1] = cv2.resize(frames[1], (top_width, cell_height))
            frames[2] = cv2.resize(frames[2], (top_width * 2, cell_height))
            
            # 添加文件名到每个视频帧
            for i in range(3):
                add_filename_with_outline(frames[i], video_names[i], font, font_scale, 
                                        font_thickness, text_color, text_bg_color, 
                                        text_padding, outline_thickness)
            
            # 先合并上面两个
            top_row = np.hstack((frames[0], frames[1]))
            # 然后与下面的拼接
            combined_frame = np.vstack((top_row, frames[2]))

        else:  # 4, 6, 8, 9, 12, 16个视频的情况
            # 将所有视频调整为相同尺寸
            resized_frames = [cv2.resize(frame, (cell_width, cell_height)) for frame in frames]
            
            # 添加文件名到每个视频帧
            for i in range(len(resized_frames)):
                add_filename_with_outline(resized_frames[i], video_names[i], font, font_scale, 
                                        font_thickness, text_color, text_bg_color, 
                                        text_padding, outline_thickness)
            
            # 创建空白画布
            combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # 根据行列布局放置每个视频
            for i in range(min(num_videos, rows * cols)):
                row = i // cols
                col = i % cols
                y_start = row * cell_height
                y_end = (row + 1) * cell_height
                x_start = col * cell_width
                x_end = (col + 1) * cell_width
                
                # 将调整大小后的帧放置到对应位置
                combined_frame[y_start:y_end, x_start:x_end] = resized_frames[i]

        # 写入输出视频
        out.write(combined_frame)
        
        # 显示进度
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")

    # 释放资源
    for cap in caps:
        cap.release()
    out.release()
    print(f"处理完成！视频已保存到 {output_path}")

def add_filename_with_outline(frame, filename, font, font_scale, font_thickness, text_color, bg_color, padding, outline_thickness):
    """
    在帧的左上角添加文件名，具有更好的可见性和可读性，添加文字描边效果使文字更清晰
    
    Args:
        frame: 视频帧
        filename: 要显示的文件名
        font: 字体
        font_scale: 字体大小
        font_thickness: 字体粗细
        text_color: 文本颜色
        bg_color: 背景颜色
        padding: 文本周围的填充像素
        outline_thickness: 文字描边粗细
    """
    # 获取帧的高度和宽度
    h, w = frame.shape[:2]
    
    # 根据帧的宽度调整文本长度
    # 计算每个字符的近似宽度
    char_width = int(12 * font_scale)  # 假设每个字符大约12像素宽（取决于字体）
    max_chars = max(10, int((w * 0.9) / char_width))  # 使用帧宽度的90%确定最大字符数
    
    # 如果文件名太长，则截断并添加省略号
    if len(filename) > max_chars:
        display_text = filename[:max_chars-3] + "..."
    else:
        display_text = filename
    
    # 获取文本大小
    text_size, _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    # 文本位置（左上角）
    x, y = padding, padding + text_h
    
    # 创建有足够边距的背景
    bg_padding = padding // 2
    # 背景矩形比文本稍大，添加额外的空间用于文本描边
    cv2.rectangle(frame, 
                 (x - bg_padding - outline_thickness, y - text_h - bg_padding - outline_thickness), 
                 (x + text_w + bg_padding + outline_thickness, y + bg_padding + outline_thickness), 
                 bg_color, -1)  # -1表示填充矩形
    
    # 使用描边技术提高文字清晰度：先绘制黑色描边，再绘制白色文本
    # 绘制文字描边（在8个方向上偏移并绘制黑色文字）
    outline_color = (0, 0, 0)  # 黑色描边
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
        cv2.putText(frame, display_text, 
                   (x + dx*outline_thickness, y + dy*outline_thickness), 
                   font, font_scale, outline_color, font_thickness, cv2.LINE_AA)
    
    # 绘制主要文本（白色）
    cv2.putText(frame, display_text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame

if __name__ == "__main__":
    # 示例视频路径
    rootPath = f'D:\\02-Study\\02-dataset\\02-data-vbench2\\sora\\Sora\\Dynamic_Attribute\\'
    
    # 创建16个视频路径（为了演示，这里重复使用相同的视频）
    base_path = f"{rootPath}\\A butterfly's wings change from white to yellow.-0.mp4"
    
    # add 4 videos path 
    rootPath2 =f'D:\\04-dataset\\Vbench-data\\Text2vIDEO\\sora\\Sora\\Motion_Rationality\\'
    video_paths = [
        f"{rootPath2}\\A person is biting into an apple.-0.mp4",
        f"{rootPath2}\\A person is biting into an apple.-1.mp4",
        f"{rootPath2}\\A person is biting into an apple.-2.mp4",
        # f"{rootPath2}\\a bear hunting for prey.-3.mp4"
        # f"{rootPath}\\A butterfly's wings change from white to yellow.-0.mp4",
        # f"{rootPath}\\A butterfly's wings change from white to yellow.-1.mp4",
        # f"{rootPath}\\A butterfly's wings change from white to yellow.-2.mp4",
        # f"{rootPath}\\A butterfly's wings change from yellow to white.-2.mp4"
    ]
    
    # 输出视频文件名
    output_names = {
        2: "output_2videos.mp4",
        3: "output_3videos_more_clear.mp4",
        4: "output_4videos_large_name.mp4",
        6: "output_6videos.mp4",
        8: "output_8videos.mp4",
        9: "output_9videos.mp4",
        12: "output_12videos.mp4",
        16: "output_16videos.mp4"
    }
    
    # 测试2、3、4视频拼接
    # 要添加更多视频时，可以复制现有的路径或添加新路径
    videos_2 = video_paths[:2]  # 取前2个视频
    videos_3 = video_paths[:3]  # 取前3个视频
    # videos_4 = video_paths[:4]  # 取前4个视频
    videos_8 = video_paths[:8]  # 取前4个视频
    
    # 创建更多视频以测试其他布局（这里仅为示例，实际使用时替换为真实视频路径）
    videos_6 = video_paths[:2] * 3  # 复制前2个视频3次，得到6个视频
    videos_8 = video_paths[:2] * 4  # 复制前2个视频4次，得到8个视频
    videos_9 = video_paths[:3] * 3  # 复制前3个视频3次，得到9个视频
    videos_12 = video_paths[:3] * 4  # 复制前3个视频4次，得到12个视频
    videos_16 = video_paths[:4] * 4  # 复制前4个视频4次，得到16个视频
    
    # 根据需要取消注释来测试不同数量的视频拼接
    join_videos(videos_3, output_names[3])  # 测试4个视频拼接
    # join_videos(videos_2, output_names[2])  # 测试2个视频拼接
    # join_videos(videos_3, output_names[3])  # 测试3个视频拼接
    # join_videos(videos_6, output_names[6])  # 测试6个视频拼接
    # join_videos(videos_8, output_names[8])  # 测试8个视频拼接
    # join_videos(videos_9, output_names[9])  # 测试9个视频拼接
    # join_videos(videos_12, output_names[12])  # 测试12个视频拼接
    # join_videos(videos_16, output_names[16])  # 测试16个视频拼接
