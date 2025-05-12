# 写一个gradio的界面，界面中有一个输入框，一个按钮，一个输出框
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中
# 当用户输入文字后，点击按钮，会调用一个函数，函数会返回一个图片
# 然后图片会显示在输出框中

import gradio as gr
from PIL import Image
import numpy as np

def process_image(input_image):
    """
    处理图像：水平翻转并缩小到原来的1/4大小
    """
    if input_image is None:
        return None
    
    # 转换为PIL图像（如果是numpy数组）
    if isinstance(input_image, np.ndarray):
        img = Image.fromarray(input_image)
    else:
        img = input_image
    
    # 水平翻转
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 获取当前尺寸
    width, height = flipped_img.size
    
    # 缩小到原来的1/4大小（宽和高各减半）
    new_width = width // 2
    new_height = height // 2
    resized_img = flipped_img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## 图像处理工具")
        gr.Markdown("上传一张图片，程序会将其水平翻转并缩小到原来的1/4大小")
        
        with gr.Row():
            # 输入图像
            image_input = gr.Image(
                label="上传图像",
                type="pil"
            )
        
        # 按钮
        process_button = gr.Button("处理图像")
        
        # 输出图像
        image_output = gr.Image(label="处理后的图像")
        
        # 设置点击按钮时的操作
        process_button.click(
            fn=process_image,
            inputs=[image_input],
            outputs=[image_output]
        )
    
    return demo

if __name__ == "__main__":
    app = gradio_interface()
    app.launch()