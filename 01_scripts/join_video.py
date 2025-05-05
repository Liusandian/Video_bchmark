import cv2
import numpy as np
import argparse
import math
import os
import glob

def maintain_aspect_ratio(frame, target_width, target_height):
    """
    Resize frame while maintaining aspect ratio and padding with black bars if needed
    """
    h, w = frame.shape[:2]
    target_ratio = target_width / target_height
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is wider than target, fit to width
        new_width = target_width
        new_height = int(target_width / current_ratio)
        resized = cv2.resize(frame, (new_width, new_height))
        # Add black bars on top and bottom
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        y_offset = (target_height - new_height) // 2
        result[y_offset:y_offset+new_height, :] = resized
    else:
        # Image is taller than target, fit to height
        new_height = target_height
        new_width = int(target_height * current_ratio)
        resized = cv2.resize(frame, (new_width, new_height))
        # Add black bars on left and right
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x_offset = (target_width - new_width) // 2
        result[:, x_offset:x_offset+new_width] = resized
    
    return result

def send_desktop_notification(title, message):
    """
    Send a desktop notification using plyer
    """
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
            timeout=10,  # seconds
        )
    except ImportError:
        print("Please install plyer package for desktop notifications: pip install plyer")
        print(f"{title}: {message}")

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
            # 调整两个帧的大小为单元格尺寸，保持纵横比
            frames[0] = maintain_aspect_ratio(frames[0], cell_width, cell_height)
            frames[1] = maintain_aspect_ratio(frames[1], cell_width, cell_height)
            
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
            
            # 调整三个视频的大小，保持纵横比
            frames[0] = maintain_aspect_ratio(frames[0], top_width, cell_height)
            frames[1] = maintain_aspect_ratio(frames[1], top_width, cell_height)
            frames[2] = maintain_aspect_ratio(frames[2], top_width * 2, cell_height)
            
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
            # 将所有视频调整为相同尺寸，保持纵横比
            resized_frames = [maintain_aspect_ratio(frame, cell_width, cell_height) for frame in frames]
            
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
    
    # 发送桌面通知
    send_desktop_notification(
        "视频处理完成",
        f"已成功合并 {len(video_paths)} 个视频并保存到 {output_path}"
    )
    print(f"处理完成！视频已保存到 {output_path}")

def add_filename_with_outline(frame, filename, font, font_scale, font_thickness, text_color, bg_color, padding, outline_thickness):
    """
    在帧的左上角添加文件名，具有更好的可见性和可读性，添加文字描边效果使文字更清晰
    文件名会根据长度自动调整行数，确保完整显示
    
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
    max_chars_per_line = max(10, int((w * 0.45) / char_width))  # 使用帧宽度的45%确定每行最大字符数
    
    # 计算最大可显示行数（根据帧高度，预留上下空间）
    text_height, _ = cv2.getTextSize("Test", font, font_scale, font_thickness)[0]
    max_lines = min(5, int((h * 0.3) / (text_height + padding)))  # 最多使用帧高度的30%，且不超过5行
    
    def split_text_into_lines(text, max_chars, max_lines):
        """将文本分割成多行"""
        lines = []
        remaining_text = text
        line_count = 0  # 重命名为line_count，避免与文本内容混淆
        
        while remaining_text and line_count < max_lines:
            # 如果剩余文本小于等于每行最大字符数，直接添加
            if len(remaining_text) <= max_chars:
                lines.append(remaining_text)
                break
            
            # 查找分割点
            split_point = max_chars
            # 向前查找最近的分割字符（空格、连字符、下划线）
            while split_point > 0 and remaining_text[split_point-1] not in [' ', '-', '_']:
                split_point -= 1
            
            # 如果没找到合适的分割点，就强制分割
            if split_point == 0:
                split_point = max_chars
            
            # 添加当前行
            current_line_text = remaining_text[:split_point].strip()  # 重命名为current_line_text
            lines.append(current_line_text)
            
            # 更新剩余文本
            remaining_text = remaining_text[split_point:].strip()
            line_count += 1  # 使用line_count作为计数器
            
            # 如果是最后一行且还有剩余文本，添加省略号
            if line_count == max_lines - 1 and remaining_text:
                last_line = remaining_text[:max_chars-3] + "..."
                lines.append(last_line)
                break
        
        return lines
    
    # 分割文本成多行
    display_lines = split_text_into_lines(filename, max_chars_per_line, max_lines)
    
    # 获取每行文本的尺寸
    line_heights = []
    line_widths = []
    for line in display_lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        line_widths.append(text_size[0])
        line_heights.append(text_size[1])
    
    # 计算总高度和最大宽度
    total_height = sum(line_heights) + (len(line_heights) - 1) * padding  # 行间距等于padding
    max_width = max(line_widths)
    
    # 文本起始位置（左上角）
    x = padding
    y = padding + line_heights[0]  # 第一行的y坐标
    
    # 创建有足够边距的背景
    bg_padding = padding // 2
    # 背景矩形比文本稍大，添加额外的空间用于文本描边
    cv2.rectangle(frame, 
                 (x - bg_padding - outline_thickness, 
                  padding - bg_padding - outline_thickness),
                 (x + max_width + bg_padding + outline_thickness,
                  y + total_height - line_heights[0] + bg_padding + outline_thickness),
                 bg_color, -1)  # -1表示填充矩形
    
    # 绘制每一行文本
    current_y = y
    for i, line in enumerate(display_lines):
        # 绘制文字描边（在8个方向上偏移并绘制黑色文字）
        outline_color = (0, 0, 0)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            cv2.putText(frame, line,
                       (x + dx*outline_thickness, current_y + dy*outline_thickness),
                       font, font_scale, outline_color, font_thickness, cv2.LINE_AA)
        
        # 绘制主要文本（白色）
        cv2.putText(frame, line, (x, current_y),
                   font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # 更新下一行的y坐标
        if i < len(display_lines) - 1:
            current_y += line_heights[i] + padding
    
    return frame

def scan_and_merge_series_videos(root_path,outputDir):
    """
    扫描指定目录下的所有子文件夹，找到同系列的视频并合并
    
    Args:
        root_path: Sora视频根目录路径
    """
    # 遍历所有子文件夹
    for subdir in os.listdir(root_path):
        subdir_path = os.path.join(root_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
            
        print(f"\n处理文件夹: {subdir}")
        
        # 获取所有mp4文件
        video_files = glob.glob(os.path.join(subdir_path, "*.mp4"))
        
        # 按文件名前缀（不含序号）分组
        video_groups = {}
        for index,video_file in enumerate(video_files):
            # skip the file if it endswith include "merge.mp4"
            if "merge" in video_file:
                continue

            if index >= 7:
                break
            # 获取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(video_file))[0]
            # 移除末尾的 "-数字" 部分来获取系列名
            series_name = "-".join(filename.split("-")[:-1])
            
            if series_name not in video_groups:
                video_groups[series_name] = []
            video_groups[series_name].append(video_file)
        # add path to merge video
        path_to_merge_video = os.path.join(outputDir,subdir)
        if not os.path.exists(path_to_merge_video):
            os.makedirs(path_to_merge_video)

        # 处理每个视频系列
        for series_name, video_paths in video_groups.items():
            if len(video_paths) > 1:  # 只处理有多个视频的系列
                print(f"\n合并系列: {series_name}")
                print(f"找到 {len(video_paths)} 个视频文件")
                
                # 按照数字序号排序
                video_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]))
                
                # 构建输出文件路径
                output_filename = f"{path_to_merge_video}\\{series_name}-merge.mp4"
                output_path = os.path.join(subdir_path, output_filename)
                
                # 调用合并函数
                join_videos(video_paths, output_path)

if __name__ == "__main__":
    # 指定Sora视频根目录
    sora_root_path = "D:\\04-dataset\\Vbench-data\\Text2vIDEO\\sora\\Sora"

    outputDir = f'D:\\04-dataset\\Vbench-data\\MergeVideoSkip2'

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    # 扫描并合并所有系列视频
    scan_and_merge_series_videos(sora_root_path,outputDir)
