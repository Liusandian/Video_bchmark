import gradio as gr

def update_output(choice, text):
    if choice == "大写":
        return text.upper()
    elif choice == "小写":
        return text.lower()
    return text

with gr.Blocks() as demo:
    text = gr.Textbox(label="输入文本")
    choice = gr.Radio(["大写", "小写", "原样"], label="处理方式")
    output = gr.Textbox(label="输出")
    
    # 实时更新
    text.change(update_output, [choice, text], output)
    choice.change(update_output, [choice, text], output)

demo.launch()