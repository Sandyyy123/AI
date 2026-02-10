# RAG Pipeline Usage Guide

This document provides a comprehensive guide to operating and configuring your RAG (Retrieval-Augmented Generation) pipeline. It details the purpose of each phase, available options, and how to orchestrate the pipeline for various scenarios. This guide covers configuration precedence, dynamic naming conventions, and advanced RAG techniques.

## Table of Contents

1.  [Overview & Key Concepts](#1-overview--key-concepts)
2.  [Setting Up Your Environment](#2-setting-up-your-environment)
3.  [Configuration Files](#3-configuration-files)
    *   [`config.yaml`](#configyaml)
    *   [`data_sources.yaml`](#data_sourcesyaml)
4.  [Phase A: Ingestion & Chunking (`scripts/phase_a_build_chunks.py`)](#4-phase-a-ingestion--chunking-scriptsphase_a_build_chunkspy)
    *   [Purpose & Outputs](#purpose--outputs)
    *   [Key Options](#key-options)
    *   [Example A.1: Ingesting a Mixed Corpus with Parent-Document & Proposition Chunking](#example-a1-ingesting-a-mixed-corpus-with-parent-document--proposition-chunking)
5.  [Phase B: Embedding & Vector Store Population (`scripts/phase_b_embed.py`)](#5-phase-b-embedding--vector-store-population-scriptsphase_b_embedpy)
    *   [Purpose & Outputs](#purpose--outputs-1)
    *   [Key Options](#key-options-1)
    *   [Example B.1: Populating ChromaDB with Google Gemini Embeddings](#example-b1-populating-chromadb-with-google-gemini-embeddings)
    *   [Example B.2: Populating FAISS for the same data, but with a Local HF Embedder](#example-b2-populating-faiss-for-the-same-data-but-with-a-local-hf-embedder)
6.  [Phase C: Querying & Generation (`scripts/phase_c_query.py`)](#6-phase-c-querying--generation-scriptsphase_c_querypy)
    *   [Purpose](#purpose-2)
    *   [Key Options](#key-options-2)
    *   [Example C.1: Basic Query to ChromaDB](#example-c1-basic-query-to-chromadb)
    *   [Example C.2: Querying FAISS with MQR and FlashRank](#example-c2-querying-faiss-with-mqr-and-flashrank)
    *   [Example C.3: Full Advanced RAG Pipeline with Context Distillation](#example-c3-full-advanced-rag-pipeline-with-context-distillation)
7.  [Advanced Recommendations for Diverse Data Types](#7-advanced-recommendations-for-diverse-data-types)
    *   [Academic & Research Papers (PDF, DOCX)](#academic--research-papers-pdf-docx)
    *   [Technical Documentation & Manuals (Markdown, HTML, TXT)](#technical-documentation--manuals-markdown-html-txt)
    *   [Structured/Tabular Data (CSV, TSV, Excel)](#structuredtabular-data-csv-tsv-excel)
    *   [Conversational Logs & User Interactions](#conversational-logs--user-interactions)
    *   [Global Sense-Making & Synthesis](#global-sense-making--synthesis)
8.  [Conclusion & Future Directions](#8-conclusion--future-directions)

---

## 1. Overview & Key Concepts

This RAG system is designed for modularity, extensibility, and reproducibility across its three main phases: Ingestion, Embedding, and Querying.

*   **Configuration Precedence:**
    1.  **Command-Line Arguments (`--arg`):** Highest priority, overrides `config.yaml`. Ideal for experiments or one-off runs.
    2.  **`config.yaml`:** Provides project-wide defaults and structured configuration for RAG components.
    3.  **Hardcoded Defaults:** Fallback if neither CLI nor `config.yaml` provides a value.
*   **Dynamic Naming Consistency:**
    *   `--output-suffix-chunking-strategy` and `--output-suffix-doc-type` are critical for linking phases. They dynamically name output JSONL files (Phase A) and the unique vector store directory (Phase B/C).
    *   **Always ensure these match across phases** when working with a specific dataset.
*   **Traceability & Caching:**
    *   **`pipeline_config_hash`:** A hash of the entire `rag` section from `config.yaml` is embedded in chunk metadata. This tracks the RAG pipeline's overall configuration (MQR, reranker, distiller settings).
    *   **`source_processing_config_hash`:** A hash of how a specific source was loaded and chunked, embedded in chunk metadata.
    *   **`phase_a_manifest.json`:** Records executed Phase A operations to allow skipping reprocessing of unchanged sources.
*   **`--id-prefix`:** An optional prefix applied to all document/chunk IDs in the vector store, useful for multitenancy or versioning.

---

## 2. Setting Up Your Environment

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you have `pypdf`, `python-docx`, `requests`, `beautifulsoup4`, `markdown-it-py`, `mdit-py-plugins`, `pandas`, `openpyxl`, `sentence-transformers`, `faiss-cpu` (or `faiss-gpu`), `chromadb`, `openai`, `google-generativeai`, `litellm`, `voyageai` as needed for your chosen components.)

2.  **Environment Variables (`.env` file):** Create a `.env` file in your project root with your API keys.
    ```
    GOOGLE_API_KEY="your_google_api_key_here"
    OPENAI_API_KEY="your_openai_api_key_here"
    OPENROUTER_API_KEY="your_openrouter_api_key_here"
    OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
    XAI_API_KEY="your_xai_api_key_here"
    XAI_BASE_URL="https://api.x.ai/v1"
    OLLAMA_BASE_URL="http://localhost:11434" # Or your Ollama server address
    VOYAGE_API_KEY="your_voyage_api_key_here"
    APP_URL="https://your-rag-app.com" # For LLM header
    APP_NAME="RAG-Query-App"           # For LLM header
    ```
3.  **Configuration (`config.yaml`):** Customize the main RAG and LLM settings.
    ```yaml
    llm:
      provider: openrouter # Default LLM for final answer generation
      model: gpt-4o-mini
      temperature: 0.3

    rag:
      chunk_size: 500       # Global default for chunk size
      chunk_overlap: 50     # Global default for chunk overlap
      embedding_model: text-embedding-3-small # Global default embedding model
      embedding_provider: openrouter            # Global default embedding provider

      # Optional: Specific config for local_hf embedder
      # local_hf_embed_device: cpu # or cuda
      # local_hf_embed_normalize: true

      # NEW: Reranker Configuration (for Phase C)
      reranker:
        strategy: flashrank                     # Supported: flashrank (or leave out for no reranking)
        model: ms-marco-TinyBERT-L-2-v2         # Model for Flashrank (e.g., 'rank_llm/vbc_rank_model')
        top_n: 5                                # Default final top-N after reranking

      # NEW: Query Transformer Configuration (Multi-Query Retrieval) (for Phase C)
      query_transformer:
        strategy: multi_query                   # Supported: multi_query (or leave out for no MQR)
        num_queries: 3                          # For multi_query: how many variants to generate
        llm_model: gpt-3.5-turbo-0125           # LLM for query rewriting (can be cheaper)
        llm_provider: openrouter                # Provider for MQR LLM

      # NEW: Context Distillation Configuration (for Phase C)
      context_distiller:
        strategy: llm_summarizer                # Supported: llm_summarizer (or leave out for no distillation)
        summary_type: concise                   # e.g., concise, key_facts, verbose
        llm_model: gpt-3.5-turbo-0125           # LLM for context summarization
        llm_provider: openrouter                # Provider for Distiller LLM

      # ChromaDB specific settings (for embedding or querying Chroma)
      chroma:
        hnsw_space: cosine # Default HNSW space for Chroma (can be 'cosine', 'l2', 'ip')

    # NEW: Vector Store Configuration
    vector_store:
      provider: chroma                          # Default provider: 'chroma' or 'faiss'
      base_path: vector_stores                  # Base directory for local vector store persistence

      faiss:                                    # FAISS-specific settings
        embedding_dimension: 1536               # **CRITICAL**: Must match the dimension of your chosen embedding model! (e.g., 1536 for text-embedding-3-small, 384 for all-MiniLM-L6-v2)
    ```
4.  **Data Sources (`data_sources.yaml`):** Define the content you want to ingest.
    ```yaml
    sources:
      # Example: Folder of PDFs (Phase A will auto-discover)
      - id: research_papers
        type: folder
        format: pdf
        path: data/corpus/papers
        glob: "**/*.pdf"
        title: "Academic Research Papers"
        tags: ["pdf", "science"]
        chunking: # Source-specific chunking override (optional)
          strategy: parent_document
          child_chunk_size: 200

      # Example: Specific Markdown documentation file
      - id: project_readme
        type: file
        format: md
        path: data/docs/README.md
        title: "Project Readme File"
        tags: ["docs"]

      # Example: A web URL
      - id: langchain_docs_page
        type: url
        url: https://www.langchain.com/langchain-plus
        title: "Langchain Plus Documentation"
        tags: ["web"]
    ```

---

## 4. Phase A: Ingestion & Chunking (`scripts/phase_a_build_chunks.py`)

This script processes your raw data sources, cleans them, chunks them according to specified strategies, and outputs the result into JSONL files.

**Purpose:** Prepare documents for embedding and retrieval. It effectively creates an *intermediate dataset* of structured small chunks and larger parent documents.

**Outputs:**
*   `outputs/phase_a_processed_docs_{SUFFIX}.jsonl`: Raw extracted text of documents.
*   `outputs/phase_a_processed_chunks_{SUFFIX}.jsonl`: Structured JSONL of child chunks, destined for embedding. Each entry contains `id`, `doc_id`, `text`, and `metadata` (including `parent_chunk_id`, `source_config_hash`, `pipeline_config_hash`).
*   `outputs/phase_a_processed_parents_{SUFFIX}.jsonl`: Structured JSONL of parent documents (full text for context).
*   `outputs/phase_a_manifest.json`: A manifest recording processing details for caching.

**Usage:**

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py [OPTIONS]