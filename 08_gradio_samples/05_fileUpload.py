import gradio as gr
import os

def process_file(file):
    if file is None:
        return "请上传文件"
    
    file_info = {
        "文件名": os.path.basename(file.name),
        "文件大小": f"{os.path.getsize(file.name) / 1024:.2f} KB",
        "文件路径": file.name
    }
    return str(file_info)

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="上传文件"),
    outputs=gr.Textbox(label="文件信息")
)
demo.launch()