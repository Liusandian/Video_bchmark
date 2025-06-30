import gradio as gr

def multi_modal_process(image, audio, text):
    return {
        "图片尺寸": f"{image.size}" if image else "无图片",
        "音频长度": f"{len(audio)}" if audio is not None else "无音频",
        "文本长度": len(text)
    }

demo = gr.Interface(
    fn=multi_modal_process,
    inputs=[
        gr.Image(type="pil"),
        gr.Audio(type="numpy"),
        gr.Textbox()
    ],
    outputs=gr.JSON()
)
demo.launch()