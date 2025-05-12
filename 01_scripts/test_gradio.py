# 写一个gradio的界面，界面中有一个输入框，一个按钮，一个输出框
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def generate_image(text):
    """
    简单的函数，将文本转换为图片
    """
    # 创建一个白色背景图片
    width, height = 400, 200
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        # 尝试加载一个字体，如果失败则使用默认字体
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # 绘制文本
    text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:4]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, text, fill='black', font=font)
    
    return image

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 简单文本到图片转换器")
        
        with gr.Row():
            # 输入框
            text_input = gr.Textbox(
                label="请输入文字",
                placeholder="在这里输入要转换为图片的文字",
                lines=3
            )
        
        # 按钮
        generate_button = gr.Button("生成图片")
        
        # 输出框
        image_output = gr.Image(label="生成的图片")
        
        # 设置点击按钮时的操作
        generate_button.click(
            fn=generate_image,
            inputs=[text_input],
            outputs=[image_output]
        )
    
    return demo

if __name__ == "__main__":
    app = gradio_interface()
    app.launch()