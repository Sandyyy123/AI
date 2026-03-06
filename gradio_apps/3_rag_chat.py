#!/usr/bin/env python3
"""
App 3: RAG Pipeline Chat — chat over your ChromaDB vector store.
Tabs: Chat | Upload & Index | Pipeline Status | About

Run:
  cd /root/AI && source .venv/bin/activate
  PYTHONPATH=. python gradio_apps/3_rag_chat.py
"""
import sys
import os
import subprocess
import logging
from pathlib import Path
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from scripts.phase_c_query import get_embeddings, VECTOR_STORE_REGISTRY
from config import load_settings
import yaml

logging.basicConfig(level=logging.WARNING)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
VECTOR_STORES_BASE = PROJECT_ROOT / "vector_stores"
DATA_DIR = PROJECT_ROOT / "data"

full_config = {}
if CONFIG_PATH.exists():
    with CONFIG_PATH.open() as f:
        full_config = yaml.safe_load(f)
rag_config = full_config.get("rag", {})
settings = load_settings()

CSS = """
footer { display: none !important; }
.status-ok { color: #14a800; font-weight: bold; }
.status-err { color: #d9534f; font-weight: bold; }
"""


def list_indexes() -> list[str]:
    if not VECTOR_STORES_BASE.exists():
        return []
    return [p.name for p in sorted(VECTOR_STORES_BASE.iterdir()) if p.is_dir()]


def query_rag(question: str, index_name: str, top_k: int) -> str:
    persist_path = VECTOR_STORES_BASE / index_name
    if not persist_path.exists():
        return f"Index `{index_name}` not found. Build it in the **Upload & Index** tab."

    embedder, embedding_model = "openai", "text-embedding-3-small"
    for part in index_name.split("_"):
        if part in ("openai", "openrouter", "local_hf", "ollama"):
            embedder = part
            break

    try:
        q_emb = get_embeddings(embedder, [question], embedding_model, settings, rag_config)[0]
        store_class = VECTOR_STORE_REGISTRY.get("chroma")
        child_store = store_class(config={
            "persist_directory": persist_path,
            "collection_name": "rag_child_chunks",
            "is_vector_indexed": True,
        })
        results = child_store.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant documents found. Try a different question or rebuild the index."

        context = "\n\n---\n\n".join(results["documents"][0])
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Answer using only the context below. If the answer isn't in the context, say so.\n\nContext:\n{context}"},
                {"role": "user", "content": question},
            ],
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def chat(message: str, history: list, index_name: str, top_k: int):
    if not index_name:
        yield "No index selected. Build one in the **Upload & Index** tab first."
        return
    yield "*Searching vector store...*"
    yield query_rag(message, index_name, top_k)


def upload_and_index(files, chunking_strategy: str, doc_type: str):
    if not files:
        return "No files uploaded."

    upload_dir = DATA_DIR / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        dest = upload_dir / Path(f.name).name
        dest.write_bytes(Path(f.name).read_bytes())
        saved.append(dest.name)

    yield f"Saved {len(saved)} file(s): {', '.join(saved)}\n\nRunning Phase A (chunking)..."

    result_a = subprocess.run(
        [sys.executable, "scripts/phase_a_build_chunks.py",
         "--input-dir", str(upload_dir),
         "--chunking-strategy", chunking_strategy,
         "--output-suffix-doc-type", doc_type],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    )
    if result_a.returncode != 0:
        yield f"Phase A failed:\n```\n{result_a.stderr[-1000:]}\n```"
        return

    yield f"Phase A done.\n\nRunning Phase B (embedding)..."

    result_b = subprocess.run(
        [sys.executable, "scripts/phase_b_embed.py",
         "--embedder", "openai",
         "--model", "text-embedding-3-small",
         "--output-suffix-chunking-strategy", chunking_strategy,
         "--output-suffix-doc-type", doc_type],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    )
    if result_b.returncode != 0:
        yield f"Phase B failed:\n```\n{result_b.stderr[-1000:]}\n```"
        return

    yield f"Index built! Go to the **Chat** tab and select the new index."


def get_pipeline_status():
    manifest = PROJECT_ROOT / ".tmp" / "manifest.json"
    if not manifest.exists():
        return "No pipeline runs found. `.tmp/manifest.json` does not exist yet."
    import json
    data = json.loads(manifest.read_text())
    # manifest.json is a dict of {tool_name: {status, timestamp, ...}}
    if isinstance(data, dict):
        entries = [{"tool": k, **v} for k, v in data.items()]
    else:
        entries = data
    lines = ["## Pipeline Manifest\n"]
    for e in list(reversed(entries))[:20]:
        status_icon = "✅" if e.get("status") == "success" else "❌"
        lines.append(f"{status_icon} **{e.get('tool', 'unknown')}** — {e.get('timestamp', '')[:19]}")
        if e.get("message"):
            lines.append(f"   {e['message']}")
    return "\n".join(lines)


indexes = list_indexes()

with gr.Blocks(title="RAG Chat") as demo:
    gr.Markdown("# RAG Pipeline Chat")

    with gr.Tabs():
        # ── Tab 1: Chat ──────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    index_dd = gr.Dropdown(
                        choices=indexes,
                        value=indexes[0] if indexes else None,
                        label="Vector Store Index",
                    )
                    top_k = gr.Slider(1, 20, value=5, step=1, label="Top K chunks")
                    refresh_btn = gr.Button("🔄 Refresh indexes")

                with gr.Column(scale=3):
                    gr.ChatInterface(
                        fn=chat,
                        additional_inputs=[index_dd, top_k],
                    )

            refresh_btn.click(
                lambda: gr.Dropdown(choices=list_indexes()),
                outputs=[index_dd]
            )

        # ── Tab 2: Upload & Index ────────────────────────────────
        with gr.Tab("📂 Upload & Index"):
            gr.Markdown("Upload PDF/TXT files and build a searchable vector index.")
            with gr.Row():
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload documents",
                        file_types=[".pdf", ".txt", ".md"],
                        file_count="multiple",
                    )
                    chunking_dd = gr.Dropdown(
                        ["recursive_character", "sentence", "fixed_size"],
                        value="recursive_character",
                        label="Chunking strategy",
                    )
                    doc_type_txt = gr.Textbox(value="my_docs", label="Document type label")
                    build_btn = gr.Button("Build Index", variant="primary")

                with gr.Column():
                    build_output = gr.Markdown("Upload files and click **Build Index**.")

            build_btn.click(
                upload_and_index,
                inputs=[file_upload, chunking_dd, doc_type_txt],
                outputs=[build_output],
            )

        # ── Tab 3: Pipeline Status ───────────────────────────────
        with gr.Tab("📊 Pipeline Status"):
            status_md = gr.Markdown(get_pipeline_status())
            gr.Button("Refresh").click(get_pipeline_status, outputs=[status_md])

        # ── Tab 4: About ─────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
## RAG Pipeline Chat

Chat over your own documents using a ChromaDB vector store.

**Architecture (WAT framework):**
1. **Phase A** — Chunk documents (recursive, sentence, or fixed-size)
2. **Phase B** — Embed chunks with `text-embedding-3-small`
3. **Phase C** — Retrieve top-K chunks → GPT-4o generates answer

**Available indexes:** {len(indexes)}
{chr(10).join(f"- `{i}`" for i in indexes) if indexes else "- *None built yet*"}

**To build manually:**
```bash
cd /root/AI
PYTHONPATH=. python scripts/phase_a_build_chunks.py --input-dir data/ --chunking-strategy recursive_character
PYTHONPATH=. python scripts/phase_b_embed.py --embedder openai --model text-embedding-3-small
```
""")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7862, share=True)
