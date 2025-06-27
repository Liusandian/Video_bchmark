import torch.multiprocessing as mp
import gradio as gr
import tempfile
import os
from typing import Optional
import torch

from diffsynth_engine.pipelines import WanVideoPipeline, WanModelConfig
from diffsynth_engine.utils.download import fetch_model
from diffsynth_engine.utils.video import save_video


# 全局变量
pipe = None
current_lora_path = None
current_lora_scale = None

# 预定义的 LoRA 模型选项
LORA_OPTIONS = {
    "无 LoRA": None,
    "WAN Silver Style": "VoidOc/wan_silver",
    # 可以添加更多 LoRA 模型选项
}

# 预定义的提示词模板
PROMPT_TEMPLATES = {
    "樱花女性": "一张亚洲女性在水中被樱花环绕的照片。她拥有白皙的肌肤和精致的面容，樱花散落在她的脸颊，她轻盈地漂浮在水中。阳光透过水面洒下，形成斑驳的光影，营造出一种宁静而超凡脱俗的氛围。她的长发在水中轻轻飘动，眼神温柔而宁静，仿佛与周围的自然环境融为一体。背景是淡粉色的樱花花瓣缓缓随着水波上漂浮，增添了一抹梦幻色彩。画面整体呈现出柔和的色调，带有细腻的光影效果。中景脸部人像特写，俯视视角。,wan_silver,wan_silver",
    "城市夜景": "繁华都市的夜景，霓虹灯闪烁，车流如织，高楼大厦灯火通明，营造出现代都市的繁华景象",
    "自然风光": "山水风光，青山绿水，云雾缭绕，瀑布飞流直下，鸟语花香，展现大自然的壮美景色",
    "人物特写": "年轻女性肖像，温柔的眼神，柔和的光线，细腻的皮肤质感，专业摄影风格",
    "科幻场景": "未来科技城市，飞行汽车穿梭，全息投影广告，机器人行走在街道上，充满科技感的未来世界",
}


def initialize_pipeline():
    """初始化视频生成管道"""
    global pipe
    if pipe is None:
        print("正在初始化模型，请稍候...")
        mp.set_start_method("spawn", force=True)
        config = WanModelConfig(
            model_path=fetch_model("MusePublic/wan2.1-1.3b", path="dit.safetensors"),
            t5_path=fetch_model("muse/wan2.1-umt5", path="umt5.safetensors"),
            vae_path=fetch_model("muse/wan2.1-vae", path="vae.safetensors"),
            dit_fsdp=True,
        )
        pipe = WanVideoPipeline.from_pretrained(
            config, 
            parallelism=4, 
            use_cfg_parallel=True, 
            offload_mode="cpu_offload"
        )
        print("模型初始化完成！")


def load_lora_model(lora_model_name: str, lora_scale: float, progress=gr.Progress()):
    """加载 LoRA 模型"""
    global pipe, current_lora_path, current_lora_scale
    
    try:
        initialize_pipeline()
        
        if lora_model_name == "无 LoRA" or lora_model_name not in LORA_OPTIONS:
            current_lora_path = None
            current_lora_scale = None
            return "✅ 已移除 LoRA 模型"
        
        lora_path = LORA_OPTIONS[lora_model_name]
        
        # 检查是否需要重新加载 LoRA
        if current_lora_path != lora_path or current_lora_scale != lora_scale:
            progress(0.5, desc="加载 LoRA 模型...")
            
            if lora_model_name == "WAN Silver Style":
                pipe.load_lora(
                    path=fetch_model("VoidOc/wan_silver", revision="ckpt-15", path="15.safetensors"),
                    scale=lora_scale,
                    fused=False,
                )
            
            current_lora_path = lora_path
            current_lora_scale = lora_scale
            progress(1.0, desc="LoRA 加载完成")
            return f"✅ 已加载 LoRA: {lora_model_name} (权重: {lora_scale})"
        else:
            return f"✅ LoRA 已加载: {lora_model_name} (权重: {lora_scale})"
            
    except Exception as e:
        return f"❌ 加载 LoRA 失败: {str(e)}"


def generate_video(
    prompt: str,
    negative_prompt: str,
    lora_model: str,
    lora_scale: float,
    num_frames: int,
    width: int,
    height: int,
    seed: int,
    progress=gr.Progress()
):
    """生成视频的主要函数"""
    try:
        # 检查提示词
        if not prompt.strip():
            return None, "请输入提示词！"
        
        # 初始化管道
        progress(0.1, desc="初始化模型...")
        initialize_pipeline()
        
        # 加载 LoRA 模型
        progress(0.2, desc="加载 LoRA 模型...")
        lora_status = load_lora_model(lora_model, lora_scale, progress)
        print(lora_status)
        
        # 生成视频
        progress(0.3, desc="开始生成视频...")
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
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
        
        return output_path, f"视频生成成功！\n{lora_status}"
        
    except Exception as e:
        error_msg = f"生成视频时出错: {str(e)}"
        print(error_msg)
        return None, error_msg


def update_prompt_from_template(template_name: str):
    """根据模板更新提示词"""
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]
    return ""


def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="WAN LoRA 文生视频生成器", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎬 WAN LoRA 文生视频生成器")
        gr.Markdown("使用 LoRA 模型增强的文本到视频生成，支持多种艺术风格！")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 提示词设置
                gr.Markdown("### 📝 提示词设置")
                
                with gr.Row():
                    prompt_template = gr.Dropdown(
                        label="提示词模板",
                        choices=list(PROMPT_TEMPLATES.keys()),
                        value="樱花女性",
                        info="选择预设模板快速开始"
                    )
                    
                prompt = gr.Textbox(
                    label="提示词 (描述你想要的视频内容)",
                    placeholder="详细描述你想要生成的视频场景、人物、动作、风格等...",
                    lines=4,
                    value=PROMPT_TEMPLATES["樱花女性"],
                    info="支持中英文，建议详细描述"
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    placeholder="描述不想要的内容...",
                    lines=2,
                    value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    info="帮助排除不想要的元素"
                )
                
                # LoRA 设置
                gr.Markdown("### 🎨 LoRA 模型设置")
                
                with gr.Row():
                    lora_model = gr.Dropdown(
                        label="LoRA 模型",
                        choices=list(LORA_OPTIONS.keys()),
                        value="WAN Silver Style",
                        info="选择艺术风格模型"
                    )
                    
                    lora_scale = gr.Slider(
                        label="LoRA 权重",
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="控制风格强度"
                    )
                
                # 视频参数
                gr.Markdown("### ⚙️ 视频参数")
                
                with gr.Row():
                    num_frames = gr.Slider(
                        label="视频帧数",
                        minimum=25,
                        maximum=121,
                        value=81,
                        step=8,
                        info="更多帧数 = 更长视频"
                    )
                    
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                        info="固定种子获得一致结果"
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
                        maximum=832,
                        value=832,
                        step=64,
                        info="推荐竖屏比例"
                    )
                
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 输出区域
                output_video = gr.Video(
                    label="生成的视频",
                    height=450
                )
                
                status_text = gr.Textbox(
                    label="状态信息",
                    interactive=False,
                    lines=3,
                    value="等待开始生成..."
                )
        
        # 使用示例和提示
        gr.Markdown("## 💡 使用指南")
        
        with gr.Accordion("📋 详细说明", open=False):
            gr.Markdown("""
            ### LoRA 模型说明
            - **WAN Silver Style**: 银色艺术风格，适合人像和唯美场景
            - **LoRA 权重**: 0.0-2.0，建议从1.0开始调试
            
            ### 提示词技巧
            - 支持中英文混合输入
            - 可以在提示词中加入 LoRA 触发词（如 "wan_silver"）
            - 详细描述场景、人物、动作、光影效果
            - 指定视角和构图（如"中景脸部人像特写，俯视视角"）
            
            ### 参数建议
            - **帧数**: 81帧约5秒视频（15fps）
            - **分辨率**: 480x832适合竖屏观看
            - **种子**: 相同设置下固定种子产生一致结果
            
            ### 性能优化
            - 首次使用需要下载模型文件
            - 推荐使用 GPU 加速
            - 生成时间约 3-10 分钟（取决于硬件）
            """)
        
        with gr.Accordion("🎯 提示词示例", open=False):
            gr.Markdown("""
            **人物场景**:
            - "一位优雅的女性在古典花园中漫步，穿着飘逸的白色长裙，阳光透过树叶洒在她身上，营造出梦幻般的光影效果"
            
            **自然风光**:
            - "壮观的瀑布从高山上倾泻而下，水雾弥漫，彩虹横跨其中，周围绿树环绕，鸟儿在空中飞翔"
            
            **城市夜景**:
            - "现代都市的夜晚，霓虹灯璀璨夺目，车流穿梭在街道上，摩天大楼灯火通明，展现都市的繁华与活力"
            
            **艺术风格**:
            - "水彩画风格的春日樱花飞舞场景，粉色花瓣在微风中翩翩起舞，背景是朦胧的山峦，整体色调温柔梦幻"
            """)
        
        # 事件绑定
        prompt_template.change(
            fn=update_prompt_from_template,
            inputs=[prompt_template],
            outputs=[prompt]
        )
        
        generate_btn.click(
            fn=generate_video,
            inputs=[
                prompt,
                negative_prompt,
                lora_model,
                lora_scale,
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
    # 检查环境
    if torch.cuda.is_available():
        print(f"🎮 检测到GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  未检测到GPU，将使用CPU运行（速度很慢）")
    
    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # 使用不同端口避免冲突
        share=False,
        debug=True,
        show_error=True
    ) 