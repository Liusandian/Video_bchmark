import torch.multiprocessing as mp
import gradio as gr
import tempfile
import os
from PIL import Image
import torch

from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


# 全局变量存储管道
pipe = None


def initialize_pipeline():
    """初始化视频生成管道"""
    global pipe
    if pipe is None:
        print("正在初始化模型，请稍候...")
        mp.set_start_method("spawn", force=True)
        config = WanModelConfig(
            model_path=fetch_model("muse/wan2.1-i2v-14b-480p-bf16", path="dit.safetensors"),
            t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
            vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
            image_encoder_path=fetch_model(
                "muse/open-clip-xlm-roberta-large-vit-huge-14",
                path="open-clip-xlm-roberta-large-vit-huge-14.safetensors",
            ),
            dit_fsdp=True,
        )
        pipe = WanVideoPipeline.from_pretrained(
            config, 
            parallelism=4, 
            use_cfg_parallel=True, 
            offload_mode="cpu_offload"
        )
        print("模型初始化完成！")


def generate_video(
    input_image,
    prompt,
    negative_prompt,
    num_frames,
    width,
    height,
    seed,
    progress=gr.Progress()
):
    """生成视频的主要函数"""
    try:
        # 检查输入图像
        if input_image is None:
            return None, "请先上传一张图像！"
        
        # 初始化管道
        progress(0.1, desc="初始化模型...")
        initialize_pipeline()
        
        # 预处理图像
        progress(0.2, desc="处理输入图像...")
        if isinstance(input_image, str):
            image = Image.open(input_image).convert("RGB")
        else:
            image = input_image.convert("RGB")
        
        # 生成视频
        progress(0.3, desc="开始生成视频...")
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            num_frames=num_frames,
            width=width,
            height=height,
            seed=seed,
        )
        
        # 保存视频
        progress(0.9, desc="保存视频文件...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        save_video(video, output_path, fps=15)
        progress(1.0, desc="完成！")
        
        return output_path, "视频生成成功！"
        
    except Exception as e:
        error_msg = f"生成视频时出错: {str(e)}"
        print(error_msg)
        return None, error_msg


def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="WAN 图像到视频生成器", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎬 WAN 图像到视频生成器")
        gr.Markdown("上传一张图像，输入描述文字，生成精彩的视频！")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入控件
                input_image = gr.Image(
                    label="上传图像",
                    type="pil",
                    height=300
                )
                
                prompt = gr.Textbox(
                    label="提示词 (描述你想要的视频内容)",
                    placeholder="例如：Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard...",
                    lines=4,
                    value="Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression."
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词 (描述不想要的内容)",
                    placeholder="例如：blurry, low quality, distorted...",
                    lines=2,
                    value="blurry, low quality, distorted, ugly, deformed"
                )
                
                with gr.Row():
                    num_frames = gr.Slider(
                        label="视频帧数",
                        minimum=25,
                        maximum=121,
                        value=81,
                        step=8,
                        info="更多帧数 = 更长视频，但生成时间更久"
                    )
                    
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                        info="相同种子会产生相似结果"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        label="视频宽度",
                        minimum=256,
                        maximum=768,
                        value=480,
                        step=64
                    )
                    
                    height = gr.Slider(
                        label="视频高度", 
                        minimum=256,
                        maximum=768,
                        value=480,
                        step=64
                    )
                
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 输出控件
                output_video = gr.Video(
                    label="生成的视频",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="状态",
                    interactive=False,
                    value="等待开始..."
                )
        
        # 示例
        gr.Markdown("## 💡 使用提示")
        gr.Markdown("""
        - **上传图像**: 支持 JPG、PNG 等常见格式
        - **提示词**: 详细描述你希望视频中发生的动作和场景
        - **帧数**: 25-121帧，建议81帧以获得较好的效果
        - **尺寸**: 建议使用480x480或类似比例，过大会影响生成速度
        - **种子**: 固定种子可以获得可重复的结果
        """)
        
        # 绑定事件
        generate_btn.click(
            fn=generate_video,
            inputs=[
                input_image,
                prompt, 
                negative_prompt,
                num_frames,
                width,
                height,
                seed
            ],
            outputs=[output_video, status_text],
            show_progress=True
        )
    
    return interface


if __name__ == "__main__":
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        print(f"CUDA 可用，GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA 不可用，将使用 CPU（速度较慢）")
    
    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为 True 可以获得公共链接
        debug=True
    ) 