#!/usr/bin/env python3
"""
App 2: Repo Chat — chat with cached GitHub repos + class notebooks.
Tabs: Chat | Repos | About

Run:
  cd /root/AI && source .venv/bin/activate
  python gradio_apps/2_repo_chat.py
"""
import sys
import os
from pathlib import Path
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.chat_with_repos import build_context, REPO_MAP, CACHE_DIR

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

AVAILABLE_REPOS = [name for name, d in REPO_MAP.items() if (CACHE_DIR / d).exists()]

CSS = """
footer { display: none !important; }
.repo-tag { background: #e8f4fd; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
.stat-box { background: #f8f9fa; border-radius: 8px; padding: 12px; text-align: center; }
"""


def chat(message: str, history: list, selected_repos: list, model: str):
    repos = selected_repos or []
    context = build_context(message, repos)
    n_repos = len(repos) if repos else len(AVAILABLE_REPOS)

    system = f"""You are a RAG engineering expert helping a student at /root/AI improve their pipeline.

You have access to {n_repos} reference repos and class notebooks as context:
{context}

When answering:
- Reference specific files and repos where relevant
- Give concrete, actionable answers with code examples
- Name the exact technique, file, and how to implement it"""

    messages = [{"role": "system", "content": system}]
    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    header = f"*Searching {n_repos} repos — {len(context):,} chars of context*\n\n"
    partial = header
    for chunk in client.chat.completions.create(
        model=model, messages=messages, stream=True, max_tokens=4000
    ):
        partial += chunk.choices[0].delta.content or ""
        yield partial


with gr.Blocks(title="Repo Chat") as demo:
    gr.Markdown("# Repo Chat — GPT-4o + GitHub Context")

    with gr.Tabs():
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    model_dd = gr.Dropdown(
                        ["gpt-4o", "gpt-4o-mini"], value="gpt-4o",
                        label="Model"
                    )
                    repo_selector = gr.CheckboxGroup(
                        choices=AVAILABLE_REPOS,
                        value=[],
                        label=f"Filter repos (blank = all {len(AVAILABLE_REPOS)})",
                    )

                with gr.Column(scale=3):
                    gr.ChatInterface(
                        fn=chat,
                        additional_inputs=[repo_selector, model_dd],
                    )

        with gr.Tab("📚 Repos"):
            gr.Markdown("## Cached Repositories")
            with gr.Row():
                for name, dirname in REPO_MAP.items():
                    path = CACHE_DIR / dirname
                    status = "✅" if path.exists() else "❌"
                    with gr.Column(scale=1, min_width=160):
                        gr.Markdown(f"**{status} {name}**\n\n`{dirname[:30]}`")

        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## About This App
Chat with **{len(AVAILABLE_REPOS)} cached GitHub repos** using GPT-4o.

**How it works:**
1. You ask a question
2. Files in each repo are scored by keyword relevance
3. Top files are loaded as context (up to 80,000 chars)
4. GPT-4o answers with repo references

**Repos included:**
{chr(10).join(f"- **{name}**" for name in AVAILABLE_REPOS)}

**Part of the WAT framework** — `/root/AI/gradio_apps/`
""")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7861, share=True)
