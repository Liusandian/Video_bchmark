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

# 推荐的分辨率选项
RESOLUTION_OPTIONS = {
    "1280x720 (横屏高清)": (1280, 720),
    "960x960 (方形)": (960, 960),
    "720x1280 (竖屏高清)": (720, 1280),
    "832x480 (横屏标准)": (832, 480),
    "480x832 (竖屏标准)": (480, 832),
}

# 预设提示词模板
PROMPT_TEMPLATES = {
    "小鸟飞翔": "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
    "花朵绽放": "唯美动画风格，一朵含苞待放的花蕾缓缓绽放，花瓣层层展开，露出精致的花蕊。阳光柔和地洒在花瓣上，微风轻拂，花朵轻微摇摆。背景虚化的绿色叶片，营造出宁静自然的氛围。微距拍摄，柔光效果。",
    "人物转身": "电影级画质，一位优雅的女性缓缓转身，长发飘逸，眼神温柔。她身穿飘逸的裙装，动作优美流畅。背景是温暖的黄昏光线，营造出浪漫的氛围。中景人物拍摄，柔和光影。",
    "水墨渲染": "中国水墨画风格，墨水在纸上缓缓渲染开来，形成山水意境。浓淡相宜的墨色变化，展现出传统水墨画的韵味。画面从空白逐渐丰富，最终形成完整的山水画作。俯视角度，艺术质感。",
    "城市变迁": "延时摄影风格，城市街道从白天过渡到夜晚，灯光逐渐亮起，车流穿梭。高楼大厦的灯火点亮夜空，展现都市的繁华与活力。云彩在天空中流动，时间感明显。远景俯拍，动态节奏。",
}


def initialize_pipeline():
    """初始化视频生成管道"""
    global pipe
    if pipe is None:
        print("正在初始化 FLF2V 模型，请稍候...")
        mp.set_start_method("spawn", force=True)
        config = WanModelConfig(
            model_path=fetch_model("muse/wan2.1-flf2v-14b-720p-bf16", path="dit.safetensors"),
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
        print("FLF2V 模型初始化完成！")


def generate_video(
    first_frame,
    last_frame,
    prompt,
    negative_prompt,
    resolution,
    num_frames,
    cfg_scale,
    seed,
    progress=gr.Progress()
):
    """生成视频的主要函数"""
    try:
        # 检查输入图像
        if first_frame is None or last_frame is None:
            return None, "请上传首帧和尾帧图像！"
        
        # 检查提示词
        if not prompt.strip():
            return None, "请输入提示词！"
        
        # 初始化管道
        progress(0.1, desc="初始化模型...")
        initialize_pipeline()
        
        # 预处理图像
        progress(0.2, desc="处理输入图像...")
        if isinstance(first_frame, str):
            first_img = Image.open(first_frame).convert("RGB")
        else:
            first_img = first_frame.convert("RGB")
            
        if isinstance(last_frame, str):
            last_img = Image.open(last_frame).convert("RGB")
        else:
            last_img = last_frame.convert("RGB")
        
        # 获取分辨率
        width, height = RESOLUTION_OPTIONS[resolution]
        
        # 调整图像尺寸
        first_img = first_img.resize((width, height), Image.Resampling.LANCZOS)
        last_img = last_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # 生成视频
        progress(0.3, desc="开始生成视频...")
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=[first_img, last_img],
            num_frames=num_frames,
            width=width,
            height=height,
            cfg_scale=cfg_scale,
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


def update_prompt_from_template(template_name: str):
    """根据模板更新提示词"""
    if template_name in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[template_name]
    return ""


def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="WAN 首尾帧视频生成器", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎬 WAN 首尾帧视频生成器 (FLF2V)")
        gr.Markdown("上传首帧和尾帧图像，AI 将自动生成流畅的过渡视频！")
        
        with gr.Row():
            with gr.Column(scale=1):
                # 图像上传区域
                gr.Markdown("### 📸 上传图像")
                
                with gr.Row():
                    first_frame = gr.Image(
                        label="首帧图像",
                        type="pil",
                        height=200,
                        info="视频的第一帧"
                    )
                    
                    last_frame = gr.Image(
                        label="尾帧图像", 
                        type="pil",
                        height=200,
                        info="视频的最后一帧"
                    )
                
                # 提示词设置
                gr.Markdown("### 📝 提示词设置")
                
                prompt_template = gr.Dropdown(
                    label="提示词模板",
                    choices=list(PROMPT_TEMPLATES.keys()),
                    value="小鸟飞翔",
                    info="选择预设模板或自定义"
                )
                
                prompt = gr.Textbox(
                    label="提示词 (描述首尾帧之间的过渡过程)",
                    placeholder="详细描述首帧到尾帧之间发生的动作、变化过程...",
                    lines=4,
                    value=PROMPT_TEMPLATES["小鸟飞翔"],
                    info="描述图像间的运动和变化"
                )
                
                negative_prompt = gr.Textbox(
                    label="负面提示词",
                    lines=2,
                    value="镜头切换，镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    info="描述不想要的效果"
                )
                
                # 参数设置
                gr.Markdown("### ⚙️ 生成参数")
                
                with gr.Row():
                    resolution = gr.Dropdown(
                        label="分辨率",
                        choices=list(RESOLUTION_OPTIONS.keys()),
                        value="960x960 (方形)",
                        info="推荐分辨率以获得最佳效果"
                    )
                    
                    num_frames = gr.Slider(
                        label="视频帧数",
                        minimum=25,
                        maximum=121,
                        value=81,
                        step=8,
                        info="更多帧数 = 更长视频"
                    )
                
                with gr.Row():
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=10.0,
                        value=5.5,
                        step=0.5,
                        info="控制提示词遵循程度"
                    )
                    
                    seed = gr.Number(
                        label="随机种子",
                        value=42,
                        precision=0,
                        info="固定种子获得一致结果"
                    )
                
                generate_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # 输出区域
                output_video = gr.Video(
                    label="生成的视频",
                    height=400
                )
                
                status_text = gr.Textbox(
                    label="状态信息",
                    interactive=False,
                    lines=3,
                    value="等待上传图像和开始生成..."
                )
        
        # 使用指南
        gr.Markdown("## 💡 使用指南")
        
        with gr.Accordion("📋 详细说明", open=False):
            gr.Markdown("""
            ### 首尾帧视频生成 (FLF2V) 说明
            
            **功能特点**:
            - 通过首帧和尾帧自动生成中间过渡帧
            - 适合制作物体运动、变化过程的视频
            - 支持多种分辨率和参数调节
            
            **使用步骤**:
            1. **上传首帧**: 视频开始时的画面
            2. **上传尾帧**: 视频结束时的画面  
            3. **描述过程**: 详细描述首尾帧之间的变化过程
            4. **调整参数**: 选择合适的分辨率和生成参数
            5. **生成视频**: 点击按钮开始生成
            
            **参数说明**:
            - **分辨率**: 推荐使用预设的高质量分辨率
            - **帧数**: 81帧约5秒视频，适合大多数场景
            - **CFG Scale**: 5.5是平衡质量和创意的好选择
            - **种子**: 固定种子确保结果可重现
            
            **最佳实践**:
            - 首尾帧构图相似，主体位置接近
            - 详细描述中间的运动过程
            - 避免过于复杂的场景变化
            - 使用推荐的分辨率比例
            """)
        
        with gr.Accordion("🎯 应用场景示例", open=False):
            gr.Markdown("""
            ### 1. 物体运动
            **首帧**: 小鸟站在地面
            **尾帧**: 小鸟在空中飞翔
            **提示词**: "小鸟煽动翅膀从地面起飞，展翅高飞"
            
            ### 2. 变形动画  
            **首帧**: 花蕾含苞待放
            **尾帧**: 花朵完全绽放
            **提示词**: "花蕾逐渐绽放，花瓣层层展开"
            
            ### 3. 表情变化
            **首帧**: 人物严肃表情
            **尾帧**: 人物微笑表情  
            **提示词**: "面部表情从严肃逐渐转为温和的微笑"
            
            ### 4. 场景过渡
            **首帧**: 白天的城市
            **尾帧**: 夜晚的城市
            **提示词**: "时间流逝，从白天过渡到夜晚，灯光逐渐点亮"
            
            ### 5. 艺术创作
            **首帧**: 空白画布
            **尾帧**: 完成的画作
            **提示词**: "画家在画布上作画，画面逐渐丰富完整"
            """)
        
        with gr.Accordion("⚠️ 注意事项", open=False):
            gr.Markdown("""
            **图像要求**:
            - 首尾帧应该有明确的对应关系
            - 建议使用相同或相似的构图角度
            - 图像质量要清晰，避免模糊或低分辨率
            
            **提示词技巧**:
            - 重点描述运动过程而非静态画面
            - 使用动词描述变化：绽放、飞翔、转动、流动等
            - 包含时间流逝的概念：逐渐、缓缓、慢慢等
            
            **性能优化**:
            - 首次使用需要下载大约 15GB 的模型文件
            - 推荐使用 12GB+ 显存的 GPU
            - 生成时间约 5-15 分钟（取决于参数和硬件）
            
            **常见问题**:
            - 如果首尾帧差异过大，可能生成不自然的过渡
            - 复杂场景建议降低分辨率提升成功率
            - CFG Scale 过高可能导致过拟合，过低可能偏离提示词
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
                first_frame,
                last_frame,
                prompt,
                negative_prompt,
                resolution,
                num_frames,
                cfg_scale,
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
        print("💾 FLF2V 模型需要较大显存，推荐 12GB+")
    else:
        print("⚠️  未检测到GPU，将使用CPU运行（极其缓慢，不推荐）")
    
    # 创建并启动界面
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7862,  # 使用不同端口避免冲突
        share=False,
        debug=True,
        show_error=True
    ) 