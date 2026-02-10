#!/usr/bin/env python3
"""
Phase C: RAG Query Application with MQR, Reranking, and Context Distillation.

This script takes a user query, applies optional query transformations (MQR),
retrieves relevant chunks from the configured Vector Store (Chroma or FAISS),
optionally reranks the retrieved context, optionally distills the context,
and finally generates an answer using an LLM.

It dynamically reconstructs the Vector Store's persistence path based on the same parameters
used during Phase B embedding, allowing users to query specific RAG configurations.

Usage Example:
  # Basic query to ChromaDB
  PYTHONPATH=. python scripts/phase_c_query.py \
    --query "What are the main findings about the 2002 Gujarat violence?" \
    --vector-store-provider chroma \
    --output-suffix-chunking-strategy recursive_character \
    --output-suffix-doc-type my_corpus \
    --top-k-chunks 10 --top-k-parents 5 \
    --id-prefix my_unique_run \
    --embedder openai --model text-embedding-3-small

  # Query to FAISS
  PYTHONPATH=. python scripts/phase_c_query.py \
    --query "What is the capital of India?" \
    --vector-store-provider faiss \
    --faiss-dimension 1536 \
    --output-suffix-chunking-strategy sentence \
    --output-suffix-doc-type general_knowledge \
    --top-k-chunks 5 \
    --embedder openai --model text-embedding-3-small

  # Query with MQR and Flashrank reranking
  PYTHONPATH=. python scripts/phase_c_query.py \
    --query "Summarize Modi's early life and political career." \
    --output-suffix-chunking-strategy recursive_character \
    --output-suffix-doc-type my_corpus \
    --vector-store-provider chroma \
    --query-transformer-strategy multi_query \
    --reranker-strategy flashrank \
    --top-k-chunks 20 --top-n-rerank 5 \
    --id-prefix my_unique_run \
    --embedder openrouter --model text-embedding-3-small
"""

# ---- SQLITE PATCH (must be first as chromadb indirectly uses it) ----
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
# -------------------------------------------------------------------

# ---- LOAD .env EARLY (needed for Settings) ----
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

# NEW: Import for Chromadb
import chromadb # Even though abstracted, direct import might be used by ChromaVectorStore
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Internal RAG system imports ---
try:
    from config import Settings, load_settings
    # The get_config_hash is also in phase_a_build_chunks, but it's a generic utility.
    from scripts.phase_a_build_chunks import get_config_hash
except ImportError as e:
    logger.critical(f"FATAL: Missing internal RAG imports (Settings, get_config_hash): {e}. "
                    f"Ensure project structure and config.py are correct.")
    sys.exit(1)

# --- Embedder functions and dispatcher (same as in phase_b_embed.py) ---
# These functions would ideally be in a shared 'embedders.py' or 'utils.py'
# but are included here for completeness of a runnable script.
# (Omitting full definitions for brevity in this response, as they were provided and checked in Phase B)
def embed_openai(texts: List[str], model: str, api_key: str, base_url: Optional[str] = None) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
def embed_xai(texts: List[str], model: str, base_url: str, api_key: str) -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]
def embed_gemini_batch(texts: List[str], model: str, task_type: Optional[str], api_key: str) -> List[List[float]]:
    import requests # Needs requests for this
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    reqs = []
    for t in texts:
        r: Dict[str, Any] = {"content": {"parts": [{"text": t}]}}
        if task_type: r["taskType"] = task_type
        reqs.append(r)
    payload = {"requests": reqs}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json(); embeddings = data.get("embeddings")
    if not embeddings: raise ValueError("Gemini API response missing embeddings or malformed.")
    return [e["values"] for e in embeddings]
_hf_model_cache_query: Dict[str, Any] = {} # Query-specific cache for HF models
def embed_local_hf(texts: List[str], model: str, device: str, normalize: bool) -> List[List[float]]:
    from sentence_transformers import SentenceTransformer
    if model not in _hf_model_cache_query:
        logger.info(f"Loading local_hf model: {model} to device: {device}")
        _hf_model_cache_query[model] = SentenceTransformer(model, device=device)
    m = _hf_model_cache_query[model]
    vecs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=normalize)
    return vecs.tolist()
def embed_ollama(texts: List[str], model: str, ollama_url: str) -> List[List[float]]:
    import requests # Needs requests for this
    base = ollama_url.rstrip("/"); endpoint = f"{base}/api/embeddings"
    out: List[List[float]] = []
    for t in texts:
        r = requests.post(endpoint, json={"model": model, "prompt": t}, timeout=60)
        r.raise_for_status()
        data = r.json(); emb = data.get("embedding")
        if not emb: raise ValueError("Ollama API response missing 'embedding'.")
        out.append(emb)
    return out
def embed_litellm(texts: List[str], model: str) -> List[List[float]]:
    from litellm import embedding
    resp = embedding(model=model, input=texts); data = resp.get("data", [])
    if not data: raise ValueError("LiteLLM 'embedding' call returned no data.")
    return [row["embedding"] for row in data]
def embed_voyage(texts: List[str], model: str, input_type: str = "document") -> List[List[float]]:
    import voyageai
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY")); res = vo.embed(texts, model=model, input_type=input_type)
    return res.embeddings

def get_embeddings(
    embedder: str, texts: List[str], model: str, settings: Settings, rag_config: Dict[str, Any],
) -> List[List[float]]:
    if embedder == "openai":
        if not settings.OPENAI_API_KEY: raise SystemExit("Missing OPENAI_API_KEY environment variable.")
        return embed_openai(texts, model=model, api_key=settings.OPENAI_API_KEY)
    if embedder == "openrouter":
        if not settings.OPENROUTER_API_KEY or not settings.OPENROUTER_BASE_URL: raise SystemExit("Missing OPENROUTER_API_KEY/BASE_URL environment variables.")
        return embed_openai(texts, model=model, api_key=settings.OPENROUTER_API_KEY, base_url=settings.OPENROUTER_BASE_URL)
    if embedder == "xai":
        if not settings.XAI_API_KEY: raise SystemExit("Missing XAI_API_KEY environment variable.")
        xai_base_url = os.getenv("XAI_BASE_URL", rag_config.get("xai_base_url", "https://api.x.ai/v1"))
        return embed_xai(texts, model=model, base_url=xai_base_url, api_key=settings.XAI_API_KEY)
    if embedder == "gemini":
        if not settings.GOOGLE_API_KEY: raise SystemExit("Missing GOOGLE_API_KEY environment variable.")
        gemini_task_type = rag_config.get("gemini_embed_task_type", "RETRIEVAL_DOCUMENT")
        return embed_gemini_batch(texts,model=model, task_type=gemini_task_type, api_key=settings.GOOGLE_API_KEY)
    if embedder == "local_hf":
        hf_device = rag_config.get("local_hf_embed_device", "cpu"); hf_normalize = rag_config.get("local_hf_embed_normalize", False)
        return embed_local_hf(texts, model=model, device=hf_device, normalize=hf_normalize)
    if embedder == "ollama":
        if not os.getenv("OLLAMA_BASE_URL") and not rag_config.get("ollama_base_url"): raise SystemExit("Missing OLLAMA_BASE_URL env var or in config.yaml.")
        ollama_url = os.getenv("OLLAMA_BASE_URL", rag_config.get("ollama_base_url", "http://localhost:11434"))
        return embed_ollama(texts, model=model, ollama_url=ollama_url)
    if embedder == "litellm": return embed_litellm(texts, model=model)
    if embedder == "voyage":
        if not settings.VOYAGE_API_KEY: raise SystemExit("Missing VOYAGE_API_KEY environment variable.")
        voyage_input_type = rag_config.get("voyage_embed_input_type", "document")
        return embed_voyage(texts, model=model, input_type=voyage_input_type)
    raise SystemExit(f"Unknown embedder: {embedder}")


def model_slug(s: str) -> str:
    """Converts a string to a URL-friendly slug."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]


# --- NEW IMPORTS: Vector Store Abstraction ---
try:
    from scripts.rag.components.vector_stores import VECTOR_STORE_REGISTRY, BaseVectorStore
except ImportError as e:
    logger.critical(f"FATAL: Could not import VECTOR_STORE_REGISTRY from scripts.rag.components.vector_stores: {e}")
    sys.exit(1)

# --- IMPORTS for MQR, Reranking, and Context Distillation components ---
try:
    from scripts.rag.components.rerankers import RERANKER_REGISTRY, BaseReranker
    from scripts.rag.components.query_transformers import QUERY_TRANSFORMER_REGISTRY, BaseQueryTransformer
    from scripts.rag.components.context_distillers import CONTEXT_DISTILLER_REGISTRY, BaseContextDistiller
    
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_openai import ChatOpenAI
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser

    logger.info("Successfully imported MQR, Reranker, and Distiller components.")
except ImportError as e:
    logger.error(f"Error importing RAG components: {e}. All related functionalities will be disabled.", exc_info=True)
    RERANKER_REGISTRY = {}
    QUERY_TRANSFORMER_REGISTRY = {}
    CONTEXT_DISTILLER_REGISTRY = {}
    BaseChatModel = None
    ChatOpenAI = None
    ChatGoogleGenerativeAI = None
    ChatPromptTemplate = None
    RunnablePassthrough = None
    RunnableLambda = None
    StrOutputParser = None


# Helper to initialize an LLM based on config (useful for MQR and Distiller)
def initialize_llm_for_component(
    llm_purpose: str, # e.g., "main_generation", "query_transformer", "context_distiller"
    settings: Settings,
    config: Dict[str, Any], # full_config (or just rag_config if component LLM is defined within rag_config)
    default_model: str = "gpt-4o-mini",
    default_provider: str = "openrouter",
    default_temperature: float = 0.1
) -> Optional[BaseChatModel]:
    
    # Priority: 1. specific component LLM config in rag_config (e.g., rag.query_transformer.llm_model)
    #           2. general LLM config in llm section
    component_config_key = llm_purpose
    if llm_purpose == "main_generation": # Main generation typically uses the top-level 'llm' config
        llm_provider = config.get("llm", {}).get("provider", default_provider)
        llm_model = config.get("llm", {}).get("model", default_model)
        llm_temperature = config.get("llm", {}).get("temperature", default_temperature)
    else: # For MQR, Distiller, etc., look within rag_config.component_name
        component_llm_config = config.get("rag", {}).get(component_config_key, {})
        llm_provider = component_llm_config.get("llm_provider", config.get("llm", {}).get("provider", default_provider))
        llm_model = component_llm_config.get("llm_model", config.get("llm", {}).get("model", default_model))
        llm_temperature = component_llm_config.get("llm_temperature", config.get("llm", {}).get("temperature", default_temperature))

    llm_instance: Optional[BaseChatModel] = None

    if llm_provider == "openrouter" and ChatOpenAI and settings and settings.OPENROUTER_API_KEY and settings.OPENROUTER_BASE_URL:
        llm_instance = ChatOpenAI(
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            model_name=llm_model,
            temperature=llm_temperature,
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "https://your-rag-app.com"),
                "X-Title": os.getenv("APP_NAME", "RAG-Query-App"),
            }
        )
    elif llm_provider == "openai" and ChatOpenAI and settings and settings.OPENAI_API_KEY:
        llm_instance = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=llm_model,
            temperature=llm_temperature,
            default_headers={
                "HTTP-Referer": os.getenv("APP_URL", "https://your-rag-app.com"),
                "X-Title": os.getenv("APP_NAME", "RAG-Query-App"),
            }
        )
    elif llm_provider == "gemini" and ChatGoogleGenerativeAI and settings and settings.GOOGLE_API_KEY:
        llm_instance = ChatGoogleGenerativeAI(
            google_api_key=settings.GOOGLE_API_KEY,
            model=llm_model,
            temperature=llm_temperature,
        )
    else:
        logger.warning(
            f"LLM for {llm_purpose} not properly configured. "
            f"Provider: {llm_provider}, Model: {llm_model}. "
            f"Check config.yaml and API keys. Skipping LLM initialization for this component."
        )
        return None
    
    logger.info(f"Initialized LLM for '{llm_purpose}': Model='{llm_model}', Provider='{llm_provider}', Temp='{llm_temperature}'.")
    return llm_instance


# -----------------------------
# Main function for Phase C
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase C: RAG Query Application with MQR, Reranking, and Context Distillation.")
    parser.add_argument("--query", required=True, help="The query to ask the RAG system.")
    
    # --- Vector Store settings (now generic) ---
    parser.add_argument(
        "--vector-store-provider",
        type=str,
        default=None, # Will default to config.yaml
        choices=list(VECTOR_STORE_REGISTRY.keys()),
        help="Vector store provider to use (e.g., 'chroma', 'faiss')."
    )
    parser.add_argument(
        "--vector-store-base-path",
        type=Path,
        default="vector_stores", # Default to generic folder name
        help="Base path for local vector store persistence.",
    )
    # Chroma-specific args (ignored by FAISS)
    parser.add_argument("--chroma-mode", choices=["local", "http"], default="local", help="ChromaDB client mode (local or http).")
    parser.add_argument("--chroma-host", default=None, help="HTTP only: Chroma host")
    parser.add_argument("--chroma-port", type=int, default=None, help="HTTP only: Chroma port")
    # FAISS-specific arg (ignored by Chroma)
    parser.add_argument(
        "--faiss-dimension",
        type=int,
        default=None, # will default to config.yaml setting if not provided
        help="Dimension of embeddings for FAISS index (required for FAISS, e.g., 1536)."
    )

    parser.add_argument("--child-collection-name", type=str, default="rag_child_chunks", help="Vector store collection/index name for child chunks.")
    parser.add_argument("--parent-collection-name", type=str, default="rag_parent_documents", help="Vector store collection/index name for parent documents.")
    parser.add_argument("--config-path", type=Path, default="config.yaml", help="Path to the main config YAML file.")

    # Parameters to reconstruct the dynamic Vector Store path (MUST MATCH Phase A/B)
    parser.add_argument(
        "--output-suffix-chunking-strategy",
        type=str,
        required=True, # Critical argument: which type of data are we querying?
        help="Chunking strategy that was used to build the Vector Store."
    )
    parser.add_argument(
        "--output-suffix-doc-type",
        type=str,
        default="docs", # Match the default used in Phase A/B
        help="Document type that was used to build the Vector Store."
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default=None,
        help="Prefix used for IDs in Vector Store during embedding, if any."
    )
    # Embedder settings (MUST MATCH Phase B creation parameters)
    parser.add_argument(
        "--embedder",
        type=str,
        default=None, # Default to use config.yaml
        choices=["openai", "openrouter", "gemini", "xai", "local_hf", "ollama", "litellm", "voyage"],
        help="Embedding provider used by Phase B to build the Vector Store."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None, # Default to use config.yaml
        help="Embedding model used by Phase B to build the Vector Store."
    )

    parser.add_argument("--top-k-chunks", type=int, default=10, help="Number of child chunks to retrieve (before reranking).")
    parser.add_argument("--top-k-parents", type=int, default=3, help="Number of parent documents to use for context (if Parent Document Retrieval).")
    
    # --- Argpase arguments for MQR, Reranking, and Context Distillation ---
    parser.add_argument(
        "--query-transformer-strategy",
        type=str,
        default=None,
        choices=list(QUERY_TRANSFORMER_REGISTRY.keys()) if QUERY_TRANSFORMER_REGISTRY else [],
        help="Strategy to use for generating multiple queries (e.g., 'multi_query')."
    )
    parser.add_argument(
        "--reranker-strategy",
        type=str,
        default=None, # No reranker by default
        choices=list(RERANKER_REGISTRY.keys()) if RERANKER_REGISTRY else [],
        help="Strategy to use for reranking retrieved chunks (e.g., 'flashrank')."
    )
    parser.add_argument(
        "--top-n-rerank",
        type=int,
        default=5, # Number of top documents to keep after reranking
        help="Number of top chunks to retain after reranking."
    )
    parser.add_argument(
        "--context-distiller-strategy",
        type=str,
        default=None,
        choices=list(CONTEXT_DISTILLER_REGISTRY.keys()) if CONTEXT_DISTILLER_REGISTRY else [],
        help="Strategy to use for distilling the retrieved context (e.g., 'llm_summarizer')."
    )

    args = parser.parse_args()

    # --- Setup ---
    project_root = Path(__file__).resolve().parents[1] 
    
    settings: Settings = load_settings() 
    
    full_config = {}
    if args.config_path.exists():
        with args.config_path.open("r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
    rag_config = full_config.get("rag", {})
    llm_generation_config = full_config.get("llm", {}) # Config for the final answer generation LLM

    # Determine embedder parameters (MUST MATCH Phase B creation parameters)
    embedder_provider = args.embedder or rag_config.get("embedding_provider", llm_generation_config.get("provider", "openrouter"))
    embedding_model = args.model or rag_config.get("embedding_model", "text-embedding-3-small")

    # Determine vector store provider from CLI --vector-store-provider or config.yaml
    vector_store_global_config = full_config.get("vector_store", {}) # Get the new vector_store section from config.yaml
    selected_vector_store_provider = args.vector_store_provider or vector_store_global_config.get("provider", "chroma")

    logger.info(f"Using embedder provider: {embedder_provider}")
    logger.info(f"Using embedding model: {embedding_model}")
    logger.info(f"Selected Vector Store provider: {selected_vector_store_provider}")

    # --- RECONSTRUCT DYNAMIC VECTOR STORE PATH (MUST MATCH Phase B) ---
    dynamic_parts = []
    dynamic_parts.append(model_slug(args.output_suffix_chunking_strategy))
    if args.output_suffix_doc_type:
        dynamic_parts.append(model_slug(args.output_suffix_doc_type))
    dynamic_parts.append(model_slug(embedder_provider))
    dynamic_parts.append(model_slug(embedding_model))
    
    dynamic_persist_sub_path_name = "_".join(part for part in dynamic_parts if part)
    if not dynamic_persist_sub_path_name:
        dynamic_persist_sub_path_name = "default_rag_index"

    final_vector_store_persist_path = project_root / args.vector_store_base_path / dynamic_persist_sub_path_name
    # --- END RECONSTRUCTION ---

    if selected_vector_store_provider == "local":
        logger.info(f"Attempting to load {selected_vector_store_provider.upper()} from: {final_vector_store_persist_path}")
    else: # HTTP type connections for Chroma
        logger.info(f"Attempting to connect to {selected_vector_store_provider.upper()} remote: {args.chroma_host}:{args.chroma_port}")


    # 1. Initialize Embedder for the query (same as used for embedding chunks in Phase B)
    try:
        query_embedder_callable = lambda texts_to_embed: get_embeddings(
            embedder=embedder_provider,
            texts=texts_to_embed,
            model=embedding_model,
            settings=settings,
            rag_config=rag_config # Pass rag_config as embedders might need specific values
        )
        _ = query_embedder_callable(["test text for query embedding"])
        logger.info(f"Query embedder '{embedder_provider}' for model '{embedding_model}' initialized successfully.")
    except SystemExit as e:
        logger.error(f"Failed to initialize query embedder: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during query embedder initialization: {e}", exc_info=True)
        sys.exit(1)


    # 2. Initialize Vector Store Collections using VECTOR_STORE_REGISTRY
    vector_store_class: Type[BaseVectorStore] = VECTOR_STORE_REGISTRY.get(selected_vector_store_provider)
    if not vector_store_class:
        logger.critical(f"Configured vector_store provider '{selected_vector_store_provider}' is not registered. Aborting.")
        sys.exit(1)

    # Common config for the VectorStore instance
    common_store_config = {
        "persist_directory": final_vector_store_persist_path, # Base path for persistent data
    }
    # Add provider-specific config parts
    if selected_vector_store_provider == "chroma":
        common_store_config.update({
            "mode": args.chroma_mode,
            "host": args.chroma_host,
            "port": args.chroma_port,
        })
    elif selected_vector_store_provider == "faiss":
        faiss_dimension_config = vector_store_global_config.get("faiss", {}).get("embedding_dimension")
        faiss_dimension = args.faiss_dimension or faiss_dimension_config
        
        if not faiss_dimension:
            logger.critical("FAISS requires 'embedding_dimension' in config.yaml (vector_store.faiss.embedding_dimension) or via --faiss-dimension CLI arg.")
            sys.exit(1)
        common_store_config["dimension"] = faiss_dimension

    child_chunks_store: BaseVectorStore
    parent_docs_store: BaseVectorStore

    try:
        # Child Chunks Store (for queryable embeddings)
        child_chunks_store = vector_store_class(
            config={
                **common_store_config,
                "collection_name": args.child_collection_name,
                "is_vector_indexed": True # Always true for child chunks
            }
        )
        logger.info(f"Vector Store for child chunks '{args.child_collection_name}' ready. Count: {child_chunks_store.count()} items.")

        # Parent Documents Store (for metadata/text lookup)
        parent_docs_store = vector_store_class(
            config={
                **common_store_config,
                "collection_name": args.parent_collection_name,
                "is_vector_indexed": False # Crucial for FAISS: this store only manages metadata
            }
        )
        logger.info(f"Vector Store for parent documents '{args.parent_collection_name}' ready. Count: {parent_docs_store.count()} items.")

    except Exception as e:
        logger.critical(f"Failed to initialize vector store collections: {e}. Ensure Phase B was run correctly.", exc_info=True)
        sys.exit(1)


    # --- Initialize RAG components (Query Transformer, Reranker, Context Distiller) ---
    
    # Initialize Query Transformer (for MQR)
    query_transformer_instance: Optional[BaseQueryTransformer] = None
    if args.query_transformer_strategy:
        if args.query_transformer_strategy in QUERY_TRANSFORMER_REGISTRY:
            try:
                mqr_llm = initialize_llm_for_component(
                    "query_transformer", settings, config=full_config,
                    default_model=rag_config.get("query_transformer", {}).get("llm_model", llm_generation_config.get("model", "gpt-3.5-turbo-0125")),
                    default_provider=rag_config.get("query_transformer", {}).get("llm_provider", llm_generation_config.get("provider", "openrouter")),
                    default_temperature=rag_config.get("query_transformer", {}).get("llm_temperature", 0.1)
                )
                if mqr_llm:
                    QueryTransformerClass = QUERY_TRANSFORMER_REGISTRY[args.query_transformer_strategy]
                    query_transformer_instance = QueryTransformerClass(
                        llm=mqr_llm,
                        num_queries=rag_config.get("query_transformer", {}).get("num_queries", 3)
                    )
                    logger.info(f"Initialized query transformer: {args.query_transformer_strategy}.")
                else:
                    logger.warning("Query transformer LLM not initialized. MQR will be disabled.")
            except ImportError as ie:
                logger.error(f"Query Transformer '{args.query_transformer_strategy}' failed to initialize due to missing library: {ie}. Skipping MQR.")
            except Exception as e:
                logger.error(f"Error initializing query transformer '{args.query_transformer_strategy}': {e}. Skipping MQR.", exc_info=True)
        else:
            logger.error(f"Unknown query transformer strategy: '{args.query_transformer_strategy}'. Skipping MQR.")

    # Initialize Reranker
    reranker_instance: Optional[BaseReranker] = None
    if args.reranker_strategy:
        if args.reranker_strategy in RERANKER_REGISTRY:
            try:
                RerankerClass = RERANKER_REGISTRY[args.reranker_strategy]
                reranker_configured_model = rag_config.get("reranker", {}).get("model", "ms-marco-TinyBERT-L-2-v2")
                # top_n-rerank from CLI takes precedence, otherwise use rag_config.reranker.top_n or default
                reranker_top_n_final = args.top_n_rerank if args.top_n_rerank != parser.get_default("top_n_rerank") else rag_config.get("reranker", {}).get("top_n", 5)
                
                reranker_instance = RerankerClass(
                    model_name=reranker_configured_model,
                    top_n=reranker_top_n_final
                )
                logger.info(f"Initialized reranker strategy: {args.reranker_strategy}.")
            except ImportError as ie:
                logger.error(f"Reranker '{args.reranker_strategy}' failed to initialize due to missing library: {ie}. Skipping reranking.")
            except Exception as e:
                logger.error(f"Error initializing reranker '{args.reranker_strategy}': {e}. Skipping reranking.", exc_info=True)
        else:
            logger.error(f"Unknown reranker strategy: '{args.reranker_strategy}'. Skipping reranking.")

    # Initialize Context Distiller
    context_distiller_instance: Optional[BaseContextDistiller] = None
    if args.context_distiller_strategy:
        if args.context_distiller_strategy in CONTEXT_DISTILLER_REGISTRY:
            try:
                distiller_llm = initialize_llm_for_component(
                    "context_distiller", settings, config=full_config,
                    default_model=rag_config.get("context_distiller", {}).get("llm_model", llm_generation_config.get("model", "gpt-3.5-turbo-0125")),
                    default_provider=rag_config.get("context_distiller", {}).get("llm_provider", llm_generation_config.get("provider", "openrouter")),
                    default_temperature=rag_config.get("context_distiller", {}).get("llm_temperature", 0.0)
                )
                if distiller_llm:
                    ContextDistillerClass = CONTEXT_DISTILLER_REGISTRY[args.context_distiller_strategy]
                    context_distiller_instance = ContextDistillerClass(
                        llm=distiller_llm,
                        summary_type=rag_config.get("context_distiller", {}).get("summary_type", "concise")
                    )
                    logger.info(f"Initialized context distiller: {args.context_distiller_strategy}.")
                else:
                    logger.warning("Context distiller LLM not initialized. Context distillation will be disabled.")
            except ImportError as ie:
                logger.error(f"Context Distiller '{args.context_distiller_strategy}' failed to initialize due to missing library: {ie}. Skipping distillation.")
            except Exception as e:
                logger.error(f"Error initializing context distiller '{args.context_distiller_strategy}': {e}. Skipping distillation.", exc_info=True)
        else:
            logger.error(f"Unknown context distiller strategy: '{args.context_distiller_strategy}'. Skipping context distillation.")


    # --- Query Processing Flow ---

    # Step 1: Query Transformation (MQR)
    original_query = args.query
    queries_for_embedding_search = [original_query] # Default to original query

    if query_transformer_instance:
        logger.info(f"Applying query transformation strategy: {args.query_transformer_strategy}")
        transformed_queries = query_transformer_instance.transform_query(original_query)
        # Ensure original query is always part of the search if MQR is applied, and avoid duplicates
        queries_for_embedding_search = list(set(transformed_queries + [original_query]))
        logger.debug(f"Queries for embedding search: {queries_for_embedding_search}")
    
    # Step 2: Embed All Queries
    all_query_embeddings: List[List[float]] = []
    try:
        if queries_for_embedding_search:
            all_query_embeddings = query_embedder_callable(queries_for_embedding_search)
        else:
            raise ValueError("No queries to embed after transformation.")
        logger.info(f"Generated {len(all_query_embeddings)} embeddings for initial search.")
    except SystemExit as e:
        logger.error(f"Failed to embed queries: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error embedding queries: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 3: Initial Retrieval from child chunks store (using all query embeddings from MQR)
    all_retrieved_child_chunks: List[Dict[str, Any]] = []
    retrieved_chunk_ids_set = set() # To store unique IDs and prevent duplicates from MQR

    for q_emb in all_query_embeddings:
        logger.debug(f"Searching child collection with one of {len(all_query_embeddings)} query embeddings for top {args.top_k_chunks} chunks...")
        # Use child_chunks_store.query() instead of direct child_collection.query()
        retrieval_results = child_chunks_store.query(
            query_embeddings=[q_emb],
            n_results=args.top_k_chunks,
            include=['documents', 'metadatas', 'distances']
        )
        if retrieval_results and retrieval_results.get("ids") and retrieval_results["ids"]:
            # retrieval_results['ids'] is a list of lists since query_embeddings is a list
            for i, chunk_id_result in enumerate(retrieval_results["ids"][0]):
                # chunk_id_result might already be prefixed if Phase B did that
                # We need to make sure the ID we use to check uniqueness is consistent.
                processed_chunk_id = chunk_id_result # Assume it's already correctly prefixed from Phase B
                
                if processed_chunk_id not in retrieved_chunk_ids_set:
                    chunk_data = {
                        "id": processed_chunk_id,
                        "text": retrieval_results["documents"][0][i],
                        "metadata": retrieval_results["metadatas"][0][i],
                        "distance": retrieval_results["distances"][0][i]
                    }
                    all_retrieved_child_chunks.append(chunk_data)
                    retrieved_chunk_ids_set.add(processed_chunk_id)

    if not all_retrieved_child_chunks:
        logger.warning(f"No child chunks found for query: '{args.query}'.")
        print("I'm sorry, I couldn't find any relevant information based on the provided data.")
        return
    
    logger.info(f"Initial retrieval (potentially with MQR) yielded {len(all_retrieved_child_chunks)} unique child chunks.")


    # Step 4: Reranking
    reranked_child_chunks = all_retrieved_child_chunks # Default if no reranker
    if reranker_instance:
        logger.info(f"Applying reranking strategy: {args.reranker_strategy} to {len(all_retrieved_child_chunks)} chunks, selecting top {args.top_n_rerank}.")
        try:
            reranked_child_chunks = reranker_instance.rerank(
                query=original_query,
                documents=all_retrieved_child_chunks
            )
            logger.info(f"Reranking complete. New top {len(reranked_child_chunks)} chunks selected.")
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Proceeding with original retrieved chunks (truncated to {args.top_n_rerank}).", exc_info=True)
            reranked_child_chunks = all_retrieved_child_chunks[:args.top_n_rerank]
    else:
        reranked_child_chunks = all_retrieved_child_chunks[:args.top_n_rerank]
        logger.info(f"No reranker configured. Using top {len(reranked_child_chunks)} chunks from initial retrieval (limited to --top-n-rerank).")


    # Step 5: Extract unique parent_chunk_ids from reranked/selected child chunks
    parent_ids_from_children = set()
    for chunk_data in reranked_child_chunks:
        meta = chunk_data.get("metadata", {})
        parent_id_from_meta = meta.get("parent_chunk_id")
        if parent_id_from_meta:
            parent_ids_from_children.add(parent_id_from_meta)
        else:
            logger.debug(f"Child chunk ID '{chunk_data.get('id', 'N/A')}' from reranked results has no 'parent_chunk_id'.")
    
    unique_parent_ids = list(parent_ids_from_children)

    final_context_texts: List[str] = []
    if unique_parent_ids and args.top_k_parents > 0:
        logger.info(f"Retrieving up to {args.top_k_parents} parent documents based on {len(unique_parent_ids)} unique parent IDs identified from child chunks.")
        parents_to_fetch_ids = unique_parent_ids[:args.top_k_parents]

        # Use parent_docs_store.get() instead of direct parent_collection.get()
        parent_docs_results = parent_docs_store.get(ids=parents_to_fetch_ids, include=['documents', 'metadatas'])
        
        if parent_docs_results and parent_docs_results.get("documents"):
            final_context_texts.extend(parent_docs_results["documents"])
        else:
            logger.warning(f"No parent documents found in store '{args.parent_collection_name}' for IDs {parents_to_fetch_ids}. Falling back to reranked child chunk texts for partial context.")
            if reranked_child_chunks:
                final_context_texts.extend([c["text"] for c in reranked_child_chunks])
    
    if not final_context_texts:
        logger.info("No parent context retrieved or requested. Using reranked child chunk texts directly for context.")
        final_context_texts.extend([c["text"] for c in reranked_child_chunks])


    raw_context_for_llm = "\n\n---\n\n".join(final_context_texts)
    
    debug_context_display = raw_context_for_llm[:1000] 
    if len(raw_context_for_llm) > 1000: debug_context_display += "\n... (raw context truncated) ..."
    logger.debug(f"Raw context for LLM:\n{debug_context_display}")


    # Step 6: Context Distillation
    final_context_for_llm = raw_context_for_llm
    if context_distiller_instance:
        logger.info(f"Applying context distillation strategy: {args.context_distiller_strategy}.")
        try:
            final_context_for_llm = context_distiller_instance.distill_context(raw_context_for_llm, original_query)
            logger.info("Context distillation complete.")
        except Exception as e:
            logger.error(f"Error during context distillation: {e}. Proceeding with raw context.", exc_info=True)
    
    debug_final_context_display = final_context_for_llm[:1000]
    if len(final_context_for_llm) > 1000: debug_final_context_display += "\n... (final context truncated) ..."
    logger.debug(f"Final context for LLM:\n{debug_final_context_display}")


    # Step 7: Generate answer using LLM
    llm_instance = initialize_llm_for_component(
        "main_generation", settings, config=full_config,
        default_model=llm_generation_config.get("model", "gpt-4o-mini"),
        default_provider=llm_generation_config.get("provider", "openrouter"),
        default_temperature=llm_generation_config.get("temperature", 0.0)
    )

    if llm_instance and ChatPromptTemplate and RunnablePassthrough and StrOutputParser:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI assistant. Use the following context to answer the user's question. Be concise and accurate. If you don't know the answer based on the provided context, state that you don't have enough information.\n\nContext:\n{context}"),
                ("human", "{question}"),
            ]
        )

        rag_chain = (
            {"context": lambda x: final_context_for_llm, "question": RunnablePassthrough()}
            | prompt_template
            | llm_instance
            | StrOutputParser()
        )

        logger.info(f"Generating answer for query: '{args.query}' using LLM Model='{llm_instance.model_name}' Provider='{llm_instance.client.base_url or 'OpenAI' }'.")
        answer = rag_chain.invoke({"question": original_query})
        print("\n--- Answer ---")
        print(answer)
    else:
        print("\n--- Answer (LLM not fully configured or LangChain components missing) ---")
        print("Main LLM provider or its API keys are not properly configured, or required LangChain components are missing. Cannot generate answer.")
        print(f"\nRetrieved and potentially distilled Context (from Vector Store):\n{final_context_for_llm}")

    logger.info("RAG Application complete.")


if __name__ == "__main__":
    main()