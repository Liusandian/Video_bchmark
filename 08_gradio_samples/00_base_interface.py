import gradio as gr

def predict(text):
    return f"AI处理结果: {text.upper()}"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="输入文本"),
    outputs=gr.Textbox(label="输出结果"),
    title="AI文本处理器"
)
iface.launch()