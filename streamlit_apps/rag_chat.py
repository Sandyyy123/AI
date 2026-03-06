"""
Path 1: Streamlit RAG Chat App
Polished UI — dark sidebar, streaming chat, repo selector, file upload.

Run:
  cd /root/AI && source .venv/bin/activate
  streamlit run streamlit_apps/rag_chat.py
"""
import sys
import os
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.chat_with_repos import build_context, REPO_MAP, CACHE_DIR

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1e1e2e; }
    [data-testid="stSidebar"] * { color: #cdd6f4 !important; }
    .stChatMessage { border-radius: 12px; }
    .stButton > button { border-radius: 8px; width: 100%; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "repo_chat"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 RAG Chat")
    st.divider()

    mode = st.radio(
        "Mode",
        ["repo_chat", "simple_chat"],
        format_func=lambda x: "📚 Repo Chat" if x == "repo_chat" else "💬 Simple Chat",
    )
    st.session_state.mode = mode

    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a RAG engineering expert. Be concise and give code examples.",
        height=100,
    )

    if mode == "repo_chat":
        st.divider()
        st.markdown("**Repos to search**")
        available = [n for n, d in REPO_MAP.items() if (CACHE_DIR / d).exists()]
        selected_repos = st.multiselect(
            "Filter (blank = all)",
            options=available,
            default=[],
            label_visibility="collapsed",
        )
        st.caption(f"{len(available)} repos cached")

    st.divider()
    if st.button("🗑 Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Built with Streamlit + GPT-4o")

# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    if mode == "repo_chat":
        st.header("📚 Chat with GitHub Repos")
        st.caption("Ask about RAG, embeddings, agents — answers reference your cached repos")
    else:
        st.header("💬 Simple Chat")
        st.caption("Direct GPT-4o conversation")

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything...")

if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build context if repo mode
    context_info = ""
    if mode == "repo_chat":
        with st.spinner("Searching repos..."):
            context = build_context(prompt, selected_repos)
        context_info = f"\n\nReference context from repos ({len(context):,} chars):\n{context}"

    # Build messages
    openai_messages = [{"role": "system", "content": system_prompt + context_info}]
    for m in st.session_state.messages[:-1]:  # exclude current user msg
        openai_messages.append({"role": m["role"], "content": m["content"]})
    openai_messages.append({"role": "user", "content": prompt})

    # Stream response
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with st.chat_message("assistant"):
        response = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in client.chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=True,
                max_tokens=4000,
            )
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
