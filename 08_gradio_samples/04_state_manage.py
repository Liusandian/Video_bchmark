import gradio as gr

def chat(message, history):
    history = history or []
    response = f"AI回复: {message}"
    history.append((message, response))
    return "", history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="消息")
    clear = gr.Button("清除")
    
    msg.submit(chat, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: ([], ""), outputs=[chatbot, msg])

demo.launch()