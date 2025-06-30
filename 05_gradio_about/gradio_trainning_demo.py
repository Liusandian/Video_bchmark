import gradio as gr
import time
import json
import os
from PIL import Image
import numpy as np

# 知识点8：全局配置
DEMO_CONFIG = {
    "title": "AI智能助手 - Gradio核心功能演示",
    "description": "这个演示包含了Gradio开发的8个核心知识点",
    "version": "1.0.0"
}

# 模拟AI模型函数
def text_processor(text, mode):
    """知识点1：基础处理函数"""
    if mode == "大写转换":
        return text.upper()
    elif mode == "小写转换":
        return text.lower()
    elif mode == "反转文本":
        return text[::-1]
    elif mode == "字符统计":
        return f"字符数: {len(text)}, 单词数: {len(text.split())}"
    return text

def image_processor(image):
    """知识点3：图像处理组件"""
    if image is None:
        return None, "请上传图像"
    
    # 简单图像处理：转灰度
    gray_image = image.convert('L')
    info = f"原图尺寸: {image.size}, 处理完成"
    return gray_image, info

def file_analyzer(file):
    """知识点6：文件处理"""
    if file is None:
        return "请上传文件"
    
    try:
        file_size = os.path.getsize(file.name)
        file_name = os.path.basename(file.name)
        
        # 读取文件内容（如果是文本文件）
        content_preview = ""
        try:
            with open(file.name, 'r', encoding='utf-8') as f:
                content_preview = f.read()[:200] + "..." if len(f.read()) > 200 else f.read()
        except:
            content_preview = "二进制文件或编码不支持"
        
        return f"""文件分析结果：
📁 文件名: {file_name}
📏 文件大小: {file_size / 1024:.2f} KB
📄 内容预览: {content_preview}"""
    except Exception as e:
        return f"文件分析失败: {str(e)}"

def slow_ai_process(text, progress=gr.Progress()):
    """知识点7：进度条和异步处理"""
    progress(0, desc="初始化AI模型...")
    time.sleep(1)
    
    progress(0.3, desc="文本预处理...")
    time.sleep(1)
    
    progress(0.6, desc="AI推理中...")
    time.sleep(1)
    
    progress(0.9, desc="后处理...")
    time.sleep(1)
    
    progress(1.0, desc="完成!")
    return f"🤖 AI深度分析结果：\n'{text}' 是一个很有意思的输入！\n✨ 处理完成时间: {time.strftime('%H:%M:%S')}"

def chat_bot(message, history):
    """知识点5：状态管理 - 聊天机器人"""
    history = history or []
    
    # 简单的聊天逻辑
    if "你好" in message or "hello" in message.lower():
        response = "👋 你好！我是AI助手，有什么可以帮你的吗？"
    elif "时间" in message:
        response = f"🕐 现在时间是: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    elif "功能" in message:
        response = "🛠️ 我支持文本处理、图像处理、文件分析等功能！"
    else:
        response = f"🤔 你说的是: '{message}'，这很有趣！我正在学习更多功能。"
    
    history.append((message, response))
    return "", history

def update_text_options(mode):
    """知识点4：事件处理 - 动态更新界面"""
    if mode == "字符统计":
        return gr.Textbox(placeholder="输入文本进行统计分析...")
    else:
        return gr.Textbox(placeholder=f"输入文本进行{mode}...")

# 知识点2：Blocks布局架构
with gr.Blocks(
    theme=gr.themes.Soft(),
    title=DEMO_CONFIG["title"],
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .tab-nav button {
        font-size: 16px !important;
    }
    """
) as demo:
    
    # 标题和说明
    gr.Markdown(f"""
    # 🚀 {DEMO_CONFIG['title']}
    
    > {DEMO_CONFIG['description']} (版本: {DEMO_CONFIG['version']})
    
    本演示涵盖以下知识点：**Blocks布局** | **多种组件** | **事件处理** | **状态管理** | **文件处理** | **进度条** | **异步处理** | **部署配置**
    """)
    
    # 知识点2：使用Tab创建多页面布局
    with gr.Tabs():
        
        # Tab 1: 文本处理 (知识点1,3,4)
        with gr.TabItem("📝 文本处理"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="输入你想处理的文本...",
                        lines=3
                    )
                    mode_selector = gr.Radio(
                        ["大写转换", "小写转换", "反转文本", "字符统计"],
                        label="处理模式",
                        value="大写转换"
                    )
                    process_btn = gr.Button("🔄 处理文本", variant="primary")
                
                with gr.Column():
                    text_output = gr.Textbox(label="处理结果", lines=3)
                    
            # 知识点4：事件处理
            mode_selector.change(
                fn=update_text_options,
                inputs=mode_selector,
                outputs=text_input
            )
            
            process_btn.click(
                fn=text_processor,
                inputs=[text_input, mode_selector],
                outputs=text_output
            )
            
            # 实时处理
            text_input.change(
                fn=text_processor,
                inputs=[text_input, mode_selector],
                outputs=text_output
            )
        
        # Tab 2: 图像处理 (知识点3)
        with gr.TabItem("🖼️ 图像处理"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="上传图像",
                        type="pil"
                    )
                    image_btn = gr.Button("🎨 转换为灰度图", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="处理结果")
                    image_info = gr.Textbox(label="图像信息")
            
            image_btn.click(
                fn=image_processor,
                inputs=image_input,
                outputs=[image_output, image_info]
            )
        
        # Tab 3: 文件分析 (知识点6)
        with gr.TabItem("📁 文件分析"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="上传文件",
                        file_types=[".txt", ".json", ".csv", ".py"]
                    )
                    analyze_btn = gr.Button("🔍 分析文件", variant="primary")
                
                with gr.Column():
                    file_output = gr.Textbox(
                        label="分析结果",
                        lines=10
                    )
            
            analyze_btn.click(
                fn=file_analyzer,
                inputs=file_input,
                outputs=file_output
            )
        
        # Tab 4: AI深度处理 (知识点7)
        with gr.TabItem("🤖 AI深度处理"):
            with gr.Row():
                with gr.Column():
                    ai_input = gr.Textbox(
                        label="AI输入",
                        placeholder="输入需要AI深度分析的内容...",
                        lines=3
                    )
                    ai_btn = gr.Button("🧠 开始AI处理", variant="primary")
                
                with gr.Column():
                    ai_output = gr.Textbox(
                        label="AI处理结果",
                        lines=6
                    )
            
            ai_btn.click(
                fn=slow_ai_process,
                inputs=ai_input,
                outputs=ai_output
            )
        
        # Tab 5: 聊天机器人 (知识点5)
        with gr.TabItem("💬 智能聊天"):
            gr.Markdown("### 🤖 与AI助手对话")
            
            chatbot = gr.Chatbot(
                label="聊天记录",
                height=400
            )
            
            with gr.Row():
                chat_input = gr.Textbox(
                    label="消息",
                    placeholder="输入你的消息...",
                    scale=4
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清除", variant="secondary", scale=1)
            
            # 知识点5：状态管理
            send_btn.click(
                fn=chat_bot,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            
            chat_input.submit(
                fn=chat_bot,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot]
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, chat_input]
            )
    
    # 底部信息
    gr.Markdown("""
    ---
    ### 🎯 核心知识点总结：
    1. **Interface创建** - 基础功能封装
    2. **Blocks布局** - 灵活的页面架构  
    3. **多种组件** - 文本、图像、文件等输入输出
    4. **事件处理** - 实时交互和响应
    5. **状态管理** - 聊天历史等会话状态
    6. **文件处理** - 上传和分析功能
    7. **进度条** - 长时间任务的用户体验
    8. **部署配置** - 启动参数和环境设置
    
    💡 **最佳实践**: 根据应用需求选择合适的组件和布局，注重用户体验和错误处理！
    """)

# 知识点8：部署和启动配置
if __name__ == "__main__":
    demo.queue(concurrency_count=3)  # 支持并发
    demo.launch(
        server_name="0.0.0.0",      # 允许局域网访问
        server_port=7860,           # 指定端口
        share=False,                # 不生成公共链接
        debug=True,                 # 开发模式
        show_tips=True,             # 显示提示
        quiet=False                 # 显示启动日志
    )