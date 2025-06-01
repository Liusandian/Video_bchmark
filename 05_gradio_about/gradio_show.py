'''
可以先在线下写个gradio  demo；
1 比如上传2个MP4视频(比如分辨率是W*H；可以预览视频1/2
2 然后视频1和视频2融合后，保存为新视频video3，显示新视频并预览，python gradio实现
'''

import gradio as gr
import cv2
import numpy as np
import os
os.environ["no_proxy"]="localhost,127.0.0.1,::1"  # 避免gradio访问本地时走代理
import tempfile
from moviepy.editor import VideoFileClip, concatenate_videoclips
import time
from pathlib import Path

def get_video_info(video_path):
    """获取视频信息"""
    if video_path is None:
        return "未上传视频"
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "无法打开视频文件"
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # 获取文件大小
        file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
        
        info = f"""
        📹 视频信息:
        • 分辨率: {width} × {height}
        • 帧率: {fps:.2f} FPS
        • 时长: {duration:.2f} 秒
        • 总帧数: {frame_count}
        • 文件大小: {file_size:.2f} MB
        """
        return info
        
    except Exception as e:
        return f"获取视频信息失败: {str(e)}"

def merge_videos(video1_path, video2_path, merge_method="concatenate"):
    """合并两个视频"""
    if video1_path is None or video2_path is None:
        return None, "请先上传两个视频文件"
    
    try:
        # 创建临时输出文件
        output_dir = tempfile.mkdtemp()
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"merged_video_{timestamp}.mp4")
        
        # 使用moviepy处理视频
        clip1 = VideoFileClip(video1_path)
        clip2 = VideoFileClip(video2_path)
        
        if merge_method == "concatenate":
            # 串联合并（视频1后接视频2）
            final_clip = concatenate_videoclips([clip1, clip2])
            merge_info = f"✅ 合并完成 - 串联模式\n视频1时长: {clip1.duration:.2f}s\n视频2时长: {clip2.duration:.2f}s\n总时长: {final_clip.duration:.2f}s"
            
        elif merge_method == "side_by_side":
            # 并排合并（左右分屏）
            # 调整视频尺寸以适应并排显示
            w, h = clip1.w, clip1.h
            clip1_resized = clip1.resize(width=w//2)
            clip2_resized = clip2.resize(width=w//2)
            
            # 确保两个视频时长一致（取较短的）
            min_duration = min(clip1.duration, clip2.duration)
            clip1_resized = clip1_resized.subclip(0, min_duration)
            clip2_resized = clip2_resized.subclip(0, min_duration)
            
            # 并排合并
            final_clip = clip1_resized.set_position(('left')).set_duration(min_duration)
            final_clip = final_clip.set_mask(None)
            clip2_positioned = clip2_resized.set_position(('right')).set_duration(min_duration)
            
            from moviepy.editor import CompositeVideoClip
            final_clip = CompositeVideoClip([
                clip1_resized.set_position(('left')),
                clip2_resized.set_position(('right'))
            ], size=(w, h))
            
            merge_info = f"✅ 合并完成 - 并排模式\n合并时长: {min_duration:.2f}s\n输出分辨率: {w} × {h}"
            
        elif merge_method == "overlay":
            # 叠加合并（视频2叠加在视频1上，半透明）
            min_duration = min(clip1.duration, clip2.duration)
            clip1 = clip1.subclip(0, min_duration)
            clip2 = clip2.subclip(0, min_duration).resize(clip1.size)
            
            # 设置视频2的透明度
            clip2 = clip2.set_opacity(0.5)
            
            from moviepy.editor import CompositeVideoClip
            final_clip = CompositeVideoClip([clip1, clip2])
            merge_info = f"✅ 合并完成 - 叠加模式\n合并时长: {min_duration:.2f}s\n视频2透明度: 50%"
        
        # 输出视频
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True,
            verbose=False,
            logger=None
        )
        
        # 清理资源
        clip1.close()
        clip2.close()
        final_clip.close()
        
        # 获取输出视频信息
        output_info = get_video_info(output_path)
        full_info = f"{merge_info}\n\n{output_info}"
        
        return output_path, full_info
        
    except Exception as e:
        return None, f"❌ 合并失败: {str(e)}"

def create_video_merger_interface():
    """创建视频合并界面"""
    
    with gr.Blocks(
        title="视频合并工具",
        theme=gr.themes.Soft(),
        css="""
        .video-container { max-height: 400px; }
        .info-box { background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
    ) as interface:
        
        gr.Markdown("""
        # 🎬 视频合并工具
        
        上传两个MP4视频文件，选择合并方式，生成新的合并视频。
        
        **支持的合并方式：**
        - **串联合并**: 视频1播放完后播放视频2
        - **并排合并**: 两个视频左右分屏同时播放
        - **叠加合并**: 视频2半透明叠加在视频1上
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 视频1上传")
                video1_input = gr.Video(
                    label="选择第一个视频文件",
                    format="mp4",
                    elem_classes=["video-container"]
                )
                video1_info = gr.Markdown("等待上传视频1...", elem_classes=["info-box"])
                
            with gr.Column(scale=1):
                gr.Markdown("### 📁 视频2上传")
                video2_input = gr.Video(
                    label="选择第二个视频文件", 
                    format="mp4",
                    elem_classes=["video-container"]
                )
                video2_info = gr.Markdown("等待上传视频2...", elem_classes=["info-box"])
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ 合并设置")
                merge_method = gr.Radio(
                    choices=[
                        ("串联合并 (视频1→视频2)", "concatenate"),
                        ("并排合并 (左右分屏)", "side_by_side"), 
                        ("叠加合并 (半透明叠加)", "overlay")
                    ],
                    value="concatenate",
                    label="选择合并方式"
                )
                
                merge_btn = gr.Button(
                    "🎬 开始合并视频",
                    variant="primary",
                    size="lg"
                )
        
        gr.Markdown("### 📺 合并结果")
        
        with gr.Row():
            with gr.Column(scale=2):
                output_video = gr.Video(
                    label="合并后的视频",
                    elem_classes=["video-container"]
                )
                
            with gr.Column(scale=1):
                merge_info = gr.Markdown(
                    "等待合并...",
                    elem_classes=["info-box"]
                )
                
                download_btn = gr.DownloadButton(
                    "💾 下载合并视频",
                    visible=False
                )
        
        # 事件处理
        def update_video1_info(video):
            return get_video_info(video)
            
        def update_video2_info(video):
            return get_video_info(video)
            
        def handle_merge(video1, video2, method):
            if video1 is None or video2 is None:
                return None, "❌ 请先上传两个视频文件", gr.update(visible=False)
            
            # 显示处理中状态
            processing_info = "🔄 正在合并视频，请稍候..."
            
            result_video, info = merge_videos(video1, video2, method)
            
            if result_video:
                return result_video, info, gr.update(visible=True, value=result_video)
            else:
                return None, info, gr.update(visible=False)
        
        # 绑定事件
        video1_input.change(
            fn=update_video1_info,
            inputs=[video1_input],
            outputs=[video1_info]
        )
        
        video2_input.change(
            fn=update_video2_info,
            inputs=[video2_input],
            outputs=[video2_info]
        )
        
        merge_btn.click(
            fn=handle_merge,
            inputs=[video1_input, video2_input, merge_method],
            outputs=[output_video, merge_info, download_btn]
        )
    
    return interface

if __name__ == "__main__":
    # 创建界面
    app = create_video_merger_interface()
    
    # 启动应用
    app.launch(
        server_name="0.0.0.0",
        server_port=3361,
        inbrowser=False,
        share=True,
        debug=True,
        show_error=True
    )