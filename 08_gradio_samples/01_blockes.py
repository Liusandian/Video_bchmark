import gradio as gr

def process(text, num):
    return f"处理: {text} × {num}"

with gr.Blocks() as demo:
    gr.Markdown("# AI处理工具")
    with gr.Row():
        text_input = gr.Textbox(label="文本输入")
        num_input = gr.Slider(1, 10, label="倍数")
    output = gr.Textbox(label="结果")
    btn = gr.Button("处理")
    btn.click(process, [text_input, num_input], output)

demo.launch()