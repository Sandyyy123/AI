"""
Path 2: FastAPI backend for the RAG chat web app.
Exposes /chat and /repos endpoints consumed by index.html.

Run:
  cd /root/AI && source .venv/bin/activate
  uvicorn web_app.backend:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
import os
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.chat_with_repos import build_context, REPO_MAP, CACHE_DIR

load_dotenv()

app = FastAPI(title="RAG Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WEB_DIR = Path(__file__).parent


# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    repos: list[str] = []
    model: str = "gpt-4o"
    system_prompt: str = "You are a RAG engineering expert. Be concise and give code examples."
    use_repo_context: bool = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/repos")
def list_repos():
    return {
        "repos": [
            {"name": name, "available": (CACHE_DIR / dirname).exists()}
            for name, dirname in REPO_MAP.items()
        ]
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    """Stream chat response as Server-Sent Events."""

    async def generate() -> AsyncGenerator[str, None]:
        context_info = ""
        if req.use_repo_context:
            context = build_context(req.message, req.repos)
            context_info = f"\n\nReference context ({len(context):,} chars):\n{context}"

        messages = [{"role": "system", "content": req.system_prompt + context_info}]
        for m in req.history:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": req.message})

        stream = client.chat.completions.create(
            model=req.model,
            messages=messages,
            stream=True,
            max_tokens=4000,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content or ""
            if text:
                # Server-Sent Events format
                yield f"data: {text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
