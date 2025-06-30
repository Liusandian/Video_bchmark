import gradio as gr

def model_inference(input_text):
    # 模拟AI模型推理
    return f"模型输出: {input_text}"

demo = gr.Interface(
    fn=model_inference,
    inputs=gr.Textbox(label="模型输入"),
    outputs=gr.Textbox(label="模型输出"),
    title="AI模型演示",
    description="这是一个AI模型的演示界面"
)

# 不同部署方式
demo.launch(
    server_name="0.0.0.0",  # 允许外部访问
    server_port=7860,       # 指定端口
    share=True,             # 生成公共链接
    debug=True              # 开发模式
)