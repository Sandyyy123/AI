#!/usr/bin/env python3
"""
App 1: Simple GPT-4o Chat — polished UI with model selector and system prompt.

Run:
  cd /root/AI && source .venv/bin/activate
  python gradio_apps/1_simple_chat.py
"""
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODELS = ["gpt-4o", "gpt-4o-mini"]

CSS = """
.container { max-width: 860px; margin: auto; }
.chat-header { text-align: center; padding: 8px 0 4px; }
footer { display: none !important; }
"""


def chat(message: str, history: list, system_prompt: str, model: str):
    messages = [{"role": "system", "content": system_prompt}]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    partial = ""
    for chunk in client.chat.completions.create(
        model=model, messages=messages, stream=True, max_tokens=2000
    ):
        partial += chunk.choices[0].delta.content or ""
        yield partial


with gr.Blocks(title="GPT-4o Chat") as demo:
    gr.Markdown("# GPT-4o Chat", elem_classes="chat-header")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                model_dd = gr.Dropdown(MODELS, value="gpt-4o", label="Model", scale=1)
                sys_prompt = gr.Textbox(
                    value="You are a helpful AI assistant. Be concise and clear.",
                    label="System prompt",
                    scale=4,
                )
            chatbot = gr.ChatInterface(
                fn=chat,
                additional_inputs=[sys_prompt, model_dd],
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## About This App
A streaming GPT-4o chatbot built with Gradio.

**Features:**
- Switch between `gpt-4o` and `gpt-4o-mini`
- Customise the system prompt per conversation
- Streaming responses
- Example questions for RAG topics

**Part of the WAT framework** — `/root/AI/gradio_apps/`
""")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860, share=True)
