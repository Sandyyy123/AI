# RAG Pipeline: Complete Usage Guide

## Table of Contents

1. [Overview & Key Concepts](#overview--key-concepts)
2. [System Architecture](#system-architecture)
3. [Environment Setup](#environment-setup)
4. [Configuration Files](#configuration-files)
5. [Phase A: Ingestion & Chunking](#phase-a-ingestion--chunking)
6. [Phase B: Embedding & Vector Store Population](#phase-b-embedding--vector-store-population)
7. [Phase C: Querying & Generation](#phase-c-querying--generation)
8. [Advanced Recommendations by Data Type](#advanced-recommendations-by-data-type)
9. [Complete Workflow Examples](#complete-workflow-examples)
10. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## Overview & Key Concepts

### What is This Pipeline?

This RAG (Retrieval-Augmented Generation) system is a production-ready, three-phase pipeline designed for:
- **Modularity**: Each phase can be run independently or as part of an integrated workflow
- **Flexibility**: Supports multiple vector stores (ChromaDB, FAISS), embedding providers, and chunking strategies
- **Traceability**: Built-in configuration hashing for reproducibility and cache management
- **Advanced RAG Techniques**: Multi-Query Retrieval (MQR), Reranking, Context Distillation

### Core Principles

#### 1. Configuration Precedence

The system follows a clear hierarchy for configuration:

```
Command-Line Arguments (--arg)
    ↓ (highest priority)
config.yaml settings
    ↓
Hardcoded defaults
    ↓ (lowest priority)
```

**Example:**
```bash
# config.yaml has: embedding_provider: openrouter
# CLI override takes precedence:
--embedder gemini  # This wins!
```

#### 2. Dynamic Naming Convention

Two critical identifiers link all three phases:

- **`--output-suffix-chunking-strategy`**: Names the chunking approach (e.g., `parent_document`, `sentence`)
- **`--output-suffix-doc-type`**: Names the document corpus (e.g., `research_papers`, `legal_docs`)

**These create unique paths:**
```
Phase A Output: phase_a_processed_chunks_<strategy>_<doctype>.jsonl
Phase B Vector Store: vector_stores/<strategy>_<doctype>_<embedder>_<model>/
Phase C Query: Targets the exact same path
```

⚠️ **Critical**: These must match EXACTLY across all phases!

#### 3. Traceability & Hashing

- **`pipeline_config_hash`**: Hash of entire `rag` section from `config.yaml`
- **`source_processing_config_hash`**: Hash of specific source loading/chunking config
- **`phase_a_manifest.json`**: Tracks processed sources to enable smart caching

#### 4. ID Prefixing

Optional `--id-prefix` adds a namespace to all document IDs:
```
Without prefix: chunk_001
With prefix:    v1_project_alpha::chunk_001
```

**Use cases:**
- Multi-tenant systems
- Dataset versioning
- Preventing ID collisions

---

## System Architecture

### Three-Phase Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE A: INGESTION                         │
│  Raw Files → Parsing → Cleaning → Chunking → JSONL Output       │
│                                                                   │
│  Outputs:                                                         │
│  • phase_a_processed_chunks_<suffix>.jsonl (child chunks)        │
│  • phase_a_processed_parents_<suffix>.jsonl (parent docs)        │
│  • phase_a_manifest.json (cache tracker)                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE B: EMBEDDING & STORAGE                   │
│  JSONL Input → Embedding → Vector Store Population              │
│                                                                   │
│  Supported Vector Stores:                                        │
│  • ChromaDB (local/remote, persistent)                           │
│  • FAISS (local, in-memory after load)                           │
│                                                                   │
│  Outputs:                                                         │
│  • Populated vector database at calculated path                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE C: QUERY & GENERATION                    │
│  User Query → Embedding → Retrieval → [MQR] → [Rerank] →        │
│  → [Context Distillation] → LLM Generation → Answer              │
│                                                                   │
│  Advanced Features:                                               │
│  • Multi-Query Retrieval (MQR)                                   │
│  • FlashRank Reranking                                           │
│  • LLM-based Context Distillation                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages:**
- Document parsing: `pypdf`, `python-docx`, `beautifulsoup4`, `markdown-it-py`
- Data handling: `pandas`, `openpyxl`
- Embeddings: `sentence-transformers`, `openai`, `google-generativeai`
- Vector stores: `chromadb`, `faiss-cpu` (or `faiss-gpu`)
- Advanced RAG: `flashrank-rs` (for reranking)
- LLM providers: `litellm`, `voyageai`

### 2. Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY="sk-..."

# Google (for Gemini)
GOOGLE_API_KEY="AIza..."

# OpenRouter (multiple models)
OPENROUTER_API_KEY="sk-or-v1-..."
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"

# X.AI (Grok)
XAI_API_KEY="xai-..."
XAI_BASE_URL="https://api.x.ai/v1"

# Voyage AI (embeddings)
VOYAGE_API_KEY="pa-..."

# Ollama (local models)
OLLAMA_BASE_URL="http://localhost:11434"

# App metadata (for LLM request headers)
APP_URL="https://your-app.com"
APP_NAME="RAG-Pipeline"
```

### 3. Project Structure

```
project_root/
├── config.yaml                    # Main configuration
├── data_sources.yaml              # Data source definitions
├── .env                           # API keys (gitignored)
├── data/
│   └── corpus/
│       ├── papers/                # PDF files
│       ├── docs_md/               # Markdown files
│       ├── html_archive/          # HTML files
│       └── structured/            # CSV, Excel files
├── outputs/
│   ├── phase_a_processed_*.jsonl # Phase A outputs
│   └── phase_a_manifest.json     # Processing cache
├── vector_stores/
│   └── <dynamic_paths>/          # Vector databases
└── scripts/
    ├── phase_a_build_chunks.py
    ├── phase_b_embed.py
    └── phase_c_query.py
```

---

## Configuration Files

### config.yaml

The central configuration hub for your RAG system.

```yaml
# ============================================================================
# LLM CONFIGURATION (for final answer generation)
# ============================================================================
llm:
  provider: openrouter              # openai | openrouter | gemini | xai | ollama
  model: gpt-4o-mini                # Model name
  temperature: 0.3                  # 0.0 (factual) to 1.0 (creative)
  max_tokens: 2000                  # Max response length

# ============================================================================
# RAG CONFIGURATION (core retrieval & chunking settings)
# ============================================================================
rag:
  # ---------------------------------------------------------------------------
  # CHUNKING (Phase A defaults, can be overridden per source)
  # ---------------------------------------------------------------------------
  chunk_size: 500                   # Default chunk size in characters
  chunk_overlap: 50                 # Overlap between consecutive chunks
  
  # ---------------------------------------------------------------------------
  # EMBEDDING (Phase B)
  # ---------------------------------------------------------------------------
  embedding_model: text-embedding-3-small  # Model for vector embeddings
  embedding_provider: openrouter           # openai | openrouter | gemini | local_hf | voyage
  
  # Local HuggingFace embeddings (optional, if embedding_provider: local_hf)
  local_hf_embed_device: cpu        # cpu | cuda
  local_hf_embed_normalize: true    # Normalize embeddings for cosine similarity
  
  # ---------------------------------------------------------------------------
  # ADVANCED RAG: RERANKER (Phase C)
  # ---------------------------------------------------------------------------
  reranker:
    strategy: flashrank             # flashrank | <empty for no reranking>
    model: ms-marco-TinyBERT-L-2-v2 # FlashRank model
    top_n: 5                        # Final number of chunks after reranking
  
  # ---------------------------------------------------------------------------
  # ADVANCED RAG: QUERY TRANSFORMER (Phase C)
  # ---------------------------------------------------------------------------
  query_transformer:
    strategy: multi_query           # multi_query | <empty for no transformation>
    num_queries: 3                  # Number of query variants to generate
    llm_model: gpt-3.5-turbo-0125   # Cheaper/faster model for query rewriting
    llm_provider: openrouter
  
  # ---------------------------------------------------------------------------
  # ADVANCED RAG: CONTEXT DISTILLER (Phase C)
  # ---------------------------------------------------------------------------
  context_distiller:
    strategy: llm_summarizer        # llm_summarizer | <empty for no distillation>
    summary_type: concise           # concise | key_facts | verbose | analysis
    llm_model: gpt-3.5-turbo-0125   # Model for summarization
    llm_provider: openrouter
  
  # ---------------------------------------------------------------------------
  # CHROMADB SPECIFIC SETTINGS
  # ---------------------------------------------------------------------------
  chroma:
    hnsw_space: cosine              # cosine | l2 | ip (inner product)

# ============================================================================
# VECTOR STORE CONFIGURATION (Phase B & C)
# ============================================================================
vector_store:
  provider: chroma                  # chroma | faiss
  base_path: vector_stores          # Base directory for persistence
  
  # FAISS-specific settings
  faiss:
    embedding_dimension: 1536       # ⚠️ MUST match your embedding model!
                                    # text-embedding-3-small: 1536
                                    # all-MiniLM-L6-v2: 384
                                    # gemini-embedding-001: 768
```

### data_sources.yaml

Defines individual data sources with format-specific settings.

```yaml
sources:
  # ==========================================================================
  # FOLDER SOURCES (auto-discovery with glob patterns)
  # ==========================================================================
  
  # ---- PDF Research Papers ----
  - id: research_papers
    type: folder
    format: pdf
    path: data/corpus/papers
    glob: "**/*.pdf"                # Recursive search
    title: "Academic Research Papers"
    tags: ["pdf", "scientific"]
    chunking:                       # Source-specific override
      strategy: parent_document
      parent_chunk_size: 1500       # Larger context units
      child_chunk_size: 200         # Smaller searchable units
      child_chunk_overlap: 50
  
  # ---- DOCX Legal Documents ----
  - id: legal_docs
    type: folder
    format: docx
    path: data/corpus/legal_stuff
    glob: "**/*.docx"
    title: "Legal Memos and Contracts"
    tags: ["legal", "docx"]
    chunking:
      strategy: sentence            # High granularity for legal text
      chunk_overlap: 0
  
  # ---- Markdown Documentation ----
  - id: project_documentation_md
    type: folder
    format: md
    path: data/corpus/markdown
    glob: "**/*.md"
    title: "Project Documentation (Markdown)"
    tags: ["markdown", "docs"]
    chunking:
      strategy: markdown_header     # Respects H1, H2, H3 structure
  
  # ---- HTML Archive ----
  - id: web_archive_html
    type: folder
    format: html
    path: data/corpus/html_archive
    glob: "**/*.html"
    title: "Archived Web Content"
    tags: ["html", "web"]
    chunking:
      strategy: html_section        # Splits by semantic HTML tags
  
  # ---- Plain Text ----
  - id: documentation_txt
    type: folder
    format: txt
    path: data/corpus/txt
    glob: "**/*.txt"
    title: "Plain Text Documents"
    tags: ["txt"]
    # Uses default recursive_character strategy from config.yaml
  
  # ==========================================================================
  # FILE SOURCES (specific single files)
  # ==========================================================================
  
  # ---- Excel Spreadsheet ----
  - id: financial_data
    type: file
    format: xlsx
    path: data/structured/finance_report.xlsx
    title: "Q4 Financial Report"
    tags: ["excel", "finance"]
    loader_options:
      sheet_name: Sales_Overview    # Load specific sheet
      # If omitted, all sheets are loaded
  
  # ---- CSV Data ----
  - id: customer_data
    type: file
    format: csv
    path: data/structured/customers.csv
    title: "Customer Database"
    tags: ["csv", "customers"]
  
  # ==========================================================================
  # URL SOURCES (web pages)
  # ==========================================================================
  
  - id: dspy_tutorial
    type: url
    url: https://dspy.ai/tutorials/rag/
    title: "DSPy RAG Tutorial"
    tags: ["url", "tutorial"]
```

---

## Phase A: Ingestion & Chunking

### Purpose

Transform raw, heterogeneous data into standardized, chunked JSONL files ready for embedding.

### Key Features

- **Multi-format support**: PDF, DOCX, Markdown, HTML, TXT, CSV, Excel, URLs
- **Smart chunking**: Multiple strategies (recursive, sentence, parent-document, proposition, markdown_header, html_section)
- **Metadata enrichment**: Extracts and preserves source metadata (page numbers, headers, etc.)
- **Intelligent caching**: Tracks processed sources via `phase_a_manifest.json`

### Outputs

```
outputs/
├── phase_a_processed_docs_<strategy>_<doctype>.jsonl      # Raw extracted text
├── phase_a_processed_chunks_<strategy>_<doctype>.jsonl    # Child chunks (for embedding)
├── phase_a_processed_parents_<strategy>_<doctype>.jsonl   # Parent documents (for context)
└── phase_a_manifest.json                                   # Processing cache
```

### Usage

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py [OPTIONS]
```

### Key Options

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `--output-suffix-chunking-strategy` | ✅ | Names the chunking strategy used | `parent_document` |
| `--output-suffix-doc-type` | ✅ | Names the document corpus | `research_papers` |
| `--include-sources` | ❌ | Process only specified source IDs | `--include-sources papers docs` |
| `--exclude-sources` | ❌ | Exclude specified source IDs | `--exclude-sources old_data` |
| `--force-reprocess` | ❌ | Ignore cache, reprocess everything | `--force-reprocess` |
| `--data-sources-path` | ❌ | Custom path to data_sources.yaml | `--data-sources-path configs/sources.yaml` |
| `--config-path` | ❌ | Custom path to config.yaml | `--config-path configs/main.yaml` |
| `--timeout` | ❌ | URL request timeout (seconds) | `--timeout 30` |
| `--log-level` | ❌ | Logging verbosity | `--log-level DEBUG` |

### Chunking Strategies

| Strategy | Description | Best For | Configuration |
|----------|-------------|----------|---------------|
| **`recursive_character`** | Splits by delimiters (`\n\n`, `\n`, `.`) recursively | General text, fallback | `chunk_size`, `chunk_overlap` |
| **`sentence`** | Splits into individual sentences | High granularity needs, legal docs | `chunk_overlap` |
| **`markdown_header`** | Splits by markdown headings (H1-H6) | Structured markdown docs | Heading levels preserved |
| **`html_section`** | Splits by semantic HTML tags | Web pages, HTML docs | Respects DOM structure |
| **`proposition`** | LLM extracts atomic facts/propositions | Dense academic text | `llm_model`, `llm_provider` |
| **`parent_document`** | Creates small child chunks + large parent docs | Long-form content, combating "lost in middle" | `child_chunk_size`, `parent_chunk_size`, overlaps |

### Examples

#### Example A.1: Basic Ingestion with Recursive Chunking

```bash
# Process all sources with default recursive character splitting
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --output-suffix-chunking-strategy recursive_character \
  --output-suffix-doc-type my_documents \
  --log-level INFO
```

**Output:**
```
outputs/phase_a_processed_chunks_recursive_character_my_documents.jsonl
outputs/phase_a_processed_parents_recursive_character_my_documents.jsonl
```

#### Example A.2: Parent-Document Strategy for Research Papers

```bash
# Use parent-document chunking specifically for PDFs
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources research_papers \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_corpus \
  --log-level DEBUG
```

**What happens:**
1. Reads PDFs from path defined in `research_papers` source
2. Creates small 200-char child chunks (for embedding precision)
3. Creates large 1500-char parent chunks (for context richness)
4. Links each child to its parent via `parent_chunk_id` in metadata

#### Example A.3: Mixed Corpus with Source-Specific Strategies

```bash
# Process multiple sources, each using its own chunking strategy
# (as defined in data_sources.yaml)
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --output-suffix-chunking-strategy multi_strategy \
  --output-suffix-doc-type mixed_corpus \
  --force-reprocess
```

**Result:** Different sources chunked differently:
- PDFs → `parent_document`
- Markdown → `markdown_header`
- HTML → `html_section`
- All output to same JSONL files with strategy metadata preserved

---

## Phase B: Embedding & Vector Store Population

### Purpose

Convert text chunks into vector embeddings and populate a searchable vector database.

### Key Features

- **Multiple vector stores**: ChromaDB (feature-rich, persistent) or FAISS (fast, lightweight)
- **Flexible embeddings**: OpenAI, Gemini, local HuggingFace, Ollama, Voyage AI
- **ID prefixing**: Namespace isolation for multi-tenant or versioned datasets
- **Batch processing**: Efficient API usage with configurable batch sizes and rate limiting

### Vector Store Comparison

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| **Persistence** | Native (disk-based) | Requires separate metadata file |
| **Metadata filtering** | ✅ Rich querying | ⚠️ Basic (via separate JSON) |
| **Scalability** | Good (with remote mode) | Excellent (GPU-accelerated) |
| **Setup complexity** | Low | Medium |
| **Best for** | Development, moderate scale | Production, large-scale, speed |

### Usage

```bash
PYTHONPATH=. python scripts/phase_b_embed.py [OPTIONS]
```

### Key Options

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `--output-suffix-chunking-strategy` | ✅ | Must match Phase A | `parent_document` |
| `--output-suffix-doc-type` | ✅ | Must match Phase A | `research_papers` |
| `--vector-store-provider` | ⚠️ | Override config.yaml | `--vector-store-provider faiss` |
| `--vector-store-base-path` | ❌ | Base directory for vector stores | `--vector-store-base-path my_indexes` |
| `--embedder` | ⚠️ | Embedding provider | `--embedder gemini` |
| `--model` | ⚠️ | Specific embedding model | `--model text-embedding-3-small` |
| `--faiss-dimension` | ⚠️ | Required if using FAISS | `--faiss-dimension 1536` |
| `--id-prefix` | ❌ | Prepend to all IDs | `--id-prefix v1_` |
| `--overwrite` | ❌ | Delete existing collection/index | `--overwrite` |
| `--batch-size` | ❌ | Chunks per API call | `--batch-size 50` |
| `--sleep` | ❌ | Seconds between batches | `--sleep 0.1` |
| `--chroma-mode` | ❌ | `local` or `http` | `--chroma-mode http` |
| `--chroma-host` | ❌ | Remote ChromaDB host | `--chroma-host localhost` |
| `--chroma-port` | ❌ | Remote ChromaDB port | `--chroma-port 8000` |

### Embedding Model Dimensions

⚠️ **Critical for FAISS**: You must specify the correct dimension!

| Model | Provider | Dimension |
|-------|----------|-----------|
| `text-embedding-3-small` | OpenAI | 1536 |
| `text-embedding-3-large` | OpenAI | 3072 |
| `text-embedding-ada-002` | OpenAI | 1536 |
| `gemini-embedding-001` | Google | 768 |
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace | 384 |
| `sentence-transformers/all-mpnet-base-v2` | HuggingFace | 768 |
| `voyage-2` | Voyage AI | 1024 |

### Examples

#### Example B.1: ChromaDB with OpenAI Embeddings

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy recursive_character \
  --output-suffix-doc-type my_documents \
  --vector-store-provider chroma \
  --embedder openai \
  --model text-embedding-3-small \
  --batch-size 50 \
  --sleep 0.1
```

**Result:**
- Creates ChromaDB at: `vector_stores/recursive_character_my_documents_openai_text_embedding_3_small/`
- Stores both child chunks (with embeddings) and parent documents
- Metadata includes `pipeline_config_hash` and `source_config_hash`

#### Example B.2: FAISS with Local HuggingFace Model

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_corpus \
  --vector-store-provider faiss \
  --faiss-dimension 384 \
  --embedder local_hf \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --id-prefix papers_v1 \
  --overwrite
```

**Result:**
- Creates FAISS index at: `vector_stores/parent_document_academic_corpus_local_hf_sentence_transformers_all_minilm_l6_v2/`
- Files: `index.faiss`, `index_metadata.jsonl`
- All IDs prefixed: `papers_v1::chunk_001`

#### Example B.3: Remote ChromaDB for Team Collaboration

```bash
# Start ChromaDB server first:
# docker run -p 8000:8000 chromadb/chroma

PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy multi_strategy \
  --output-suffix-doc-type mixed_corpus \
  --vector-store-provider chroma \
  --chroma-mode http \
  --chroma-host 10.0.1.50 \
  --chroma-port 8000 \
  --embedder gemini \
  --model gemini-embedding-001 \
  --id-prefix team_shared_v1
```

**Use case:** Team shares a centralized ChromaDB instance for collaborative RAG development.

---

## Phase C: Querying & Generation

### Purpose

Answer user questions by retrieving relevant context and generating informed responses using advanced RAG techniques.

### Advanced RAG Pipeline

```
User Query
    ↓
[1] QUERY TRANSFORMATION (MQR)
    ├─ Original: "What is quantum entanglement?"
    ├─ Variant 1: "How does quantum entanglement work?"
    ├─ Variant 2: "Explain quantum particle correlation"
    └─ Variant 3: "What connects entangled quantum states?"
    ↓
[2] INITIAL RETRIEVAL
    ├─ Embed all queries
    ├─ Search vector store
    └─ Retrieve top-k chunks (e.g., 20)
    ↓
[3] RERANKING (FlashRank)
    ├─ Re-score all 20 chunks
    └─ Select top-n best matches (e.g., 5)
    ↓
[4] PARENT DOCUMENT FETCHING
    └─ Retrieve top-k parent docs for context (e.g., 3)
    ↓
[5] CONTEXT DISTILLATION
    ├─ Summarize/distill combined context
    └─ Reduce token count, highlight relevance
    ↓
[6] FINAL LLM GENERATION
    └─ Generate answer using distilled context
```

### Usage

```bash
PYTHONPATH=. python scripts/phase_c_query.py [OPTIONS]
```

### Key Options

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `--query` | ✅ | The question to answer | `--query "What is quantum computing?"` |
| `--output-suffix-chunking-strategy` | ✅ | Must match Phase A/B | `parent_document` |
| `--output-suffix-doc-type` | ✅ | Must match Phase A/B | `research_papers` |
| `--vector-store-provider` | ⚠️ | Must match Phase B | `--vector-store-provider chroma` |
| `--embedder` | ✅ | Must match Phase B | `--embedder gemini` |
| `--model` | ✅ | Must match Phase B | `--model gemini-embedding-001` |
| `--faiss-dimension` | ⚠️ | Required if FAISS | `--faiss-dimension 384` |
| `--id-prefix` | ⚠️ | Must match Phase B if used | `--id-prefix v1_` |
| `--top-k-chunks` | ❌ | Initial chunks to retrieve | `--top-k-chunks 15` |
| `--top-k-parents` | ❌ | Parent docs to fetch | `--top-k-parents 3` |
| `--query-transformer-strategy` | ❌ | Enable MQR | `--query-transformer-strategy multi_query` |
| `--reranker-strategy` | ❌ | Enable reranking | `--reranker-strategy flashrank` |
| `--top-n-rerank` | ❌ | Chunks after reranking | `--top-n-rerank 5` |
| `--context-distiller-strategy` | ❌ | Enable distillation | `--context-distiller-strategy llm_summarizer` |

### Examples

#### Example C.1: Basic Query (No Advanced Features)

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "What are the main findings on AI safety in the papers?" \
  --output-suffix-chunking-strategy recursive_character \
  --output-suffix-doc-type my_documents \
  --vector-store-provider chroma \
  --embedder openai \
  --model text-embedding-3-small \
  --top-k-chunks 7 \
  --top-k-parents 2
```

**Process:**
1. Embed query with `text-embedding-3-small`
2. Search ChromaDB for top 7 child chunks
3. Fetch 2 parent documents
4. Generate answer with default LLM

#### Example C.2: Multi-Query Retrieval + Reranking

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "Compare machine learning approaches to climate modeling across the research papers" \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_corpus \
  --vector-store-provider faiss \
  --faiss-dimension 384 \
  --embedder local_hf \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --id-prefix papers_v1 \
  --query-transformer-strategy multi_query \
  --reranker-strategy flashrank \
  --top-k-chunks 15 \
  --top-n-rerank 5 \
  --top-k-parents 3 \
  --log-level DEBUG
```

**Process:**
1. **MQR**: Generate 3 query variants (from `config.yaml`)
2. **Retrieve**: Get 15 chunks total across all variants
3. **Rerank**: FlashRank scores all 15, keeps top 5
4. **Parents**: Fetch 3 parent docs linked to top 5 chunks
5. **Generate**: Use parent context for answer

**Debug output shows:**
```
[MQR] Generated queries:
  1. Compare machine learning approaches to climate modeling across the research papers
  2. What ML techniques are used for climate prediction in the papers?
  3. How do different papers approach climate modeling with AI?

[Retrieval] Found 15 chunks (5 per query variant)

[Reranking] FlashRank scores:
  chunk_042: 0.89
  chunk_127: 0.85
  chunk_089: 0.81
  ...
```

#### Example C.3: Full Advanced Pipeline (MQR + Rerank + Distillation)

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "Summarize all policy recommendations for AI regulation found in the legal documents" \
  --output-suffix-chunking-strategy multi_strategy \
  --output-suffix-doc-type mixed_corpus \
  --vector-store-provider chroma \
  --embedder gemini \
  --model gemini-embedding-001 \
  --id-prefix team_shared_v1 \
  --query-transformer-strategy multi_query \
  --reranker-strategy flashrank \
  --top-k-chunks 25 \
  --top-n-rerank 7 \
  --context-distiller-strategy llm_summarizer \
  --top-k-parents 4 \
  --log-level INFO
```

**Process:**
1. **MQR**: Generate query variants
2. **Retrieve**: 25 initial chunks
3. **Rerank**: FlashRank to 7 best chunks
4. **Parents**: Fetch 4 parent documents
5. **Distill**: LLM summarizes the 4 parents into key facts
6. **Generate**: Final LLM uses distilled summary (reduced tokens!)

**Benefits:**
- **Precision**: Reranking ensures relevance
- **Context**: Parent docs provide full context
- **Efficiency**: Distillation reduces token costs by ~65% (per "Scaling RAG" paper)
- **Quality**: Combats "lost in the middle" problem

---

## Advanced Recommendations by Data Type

### 1. Academic & Research Papers (PDF, DOCX)

**Characteristics:**
- Highly structured (abstract, intro, methods, results, discussion)
- Dense technical content
- Long documents (often 20+ pages)
- Prone to "lost in the middle" when context window is large

**Optimal Configuration:**

```yaml
# data_sources.yaml
- id: research_papers
  type: folder
  format: pdf
  path: data/corpus/papers
  chunking:
    strategy: parent_document
    parent_chunk_size: 1500      # Full section/paragraph
    child_chunk_size: 200        # Precise search granularity
    child_chunk_overlap: 50
```

**Phase A:**
```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources research_papers \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_papers
```

**Phase B:**
```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_papers \
  --embedder openai \
  --model text-embedding-3-small \
  --vector-store-provider chroma
```

**Phase C:**
```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "What are the limitations of current quantum computing approaches?" \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type academic_papers \
  --embedder openai \
  --model text-embedding-3-small \
  --vector-store-provider chroma \
  --query-transformer-strategy multi_query \
  --reranker-strategy flashrank \
  --top-k-chunks 20 \
  --top-n-rerank 7 \
  --context-distiller-strategy llm_summarizer \
  --top-k-parents 4
```

**Why this works:**
- **Parent-document chunking**: Balances search precision (small chunks) with context richness (large parents)
- **Reranking**: Academic search needs high precision (12% improvement in Retrieval Recall@5)
- **Context distillation**: Reduces dense content to key facts (65% token reduction)
- **MQR**: Complex academic questions benefit from multiple angles

---

### 2. Technical Documentation & Manuals (Markdown, HTML, TXT)

**Characteristics:**
- Hierarchical structure (headings, subheadings)
- Code blocks, lists, procedures
- Often references other sections
- User seeks specific procedures or definitions

**Optimal Configuration:**

```yaml
# data_sources.yaml
- id: tech_docs
  type: folder
  format: md
  path: data/docs/api
  chunking:
    strategy: markdown_header     # Preserves semantic structure
```

**Phase A:**
```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources tech_docs \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type api_documentation
```

**Phase C:**
```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "How do I authenticate API requests?" \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type api_documentation \
  --embedder local_hf \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --vector-store-provider faiss \
  --faiss-dimension 384 \
  --reranker-strategy flashrank \
  --top-k-chunks 10 \
  --top-n-rerank 3
```

**Why this works:**
- **Markdown header chunking**: Each heading becomes a coherent unit
- **Reranking**: Ensures the right section is prioritized (e.g., "Authentication" over "Authorization")
- **Smaller top-k**: Technical docs are usually more straightforward; fewer chunks needed

---

### 3. Structured/Tabular Data (CSV, Excel)

**Characteristics:**
- Row-based data
- Relationships defined by columns
- Dense with entities and values
- Often better to aggregate/analyze than quote raw rows

**Optimal Configuration:**

```yaml
# data_sources.yaml
- id: sales_data
  type: file
  format: xlsx
  path: data/reports/Q4_sales.xlsx
  loader_options:
    sheet_name: Regional_Sales
```

**Phase A:**
```bash
# Structured data is automatically "chunked" as formatted rows
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources sales_data \
  --output-suffix-chunking-strategy structured_rows \
  --output-suffix-doc-type financial_reports
```

**Each row becomes a chunk:**
```
Region: North America; Product: Widget A; Revenue: $45000; Quarter: Q4
Region: Europe; Product: Widget B; Revenue: $32000; Quarter: Q4
```

**Phase C:**
```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "Which region had the highest revenue for Widget products in Q4?" \
  --output-suffix-chunking-strategy structured_rows \
  --output-suffix-doc-type financial_reports \
  --embedder gemini \
  --model gemini-embedding-001 \
  --vector-store-provider chroma \
  --context-distiller-strategy llm_summarizer \
  --top-k-chunks 15 \
  --top-k-parents 5
```

**Why this works:**
- **Row-as-chunk**: Preserves tabular integrity
- **Context distillation**: LLM aggregates rows (e.g., "North America totals $X across 5 products")
- **Higher top-k**: May need multiple rows to answer comparative questions

---

### 4. Conversational Logs & User Interactions

**Characteristics:**
- Temporal, fragmented
- User-specific context
- Informal language
- Traditional RAG struggles with personalization

**Optimal Configuration:**

```yaml
# data_sources.yaml
- id: customer_chats
  type: folder
  format: jsonl           # One conversation per line
  path: data/logs/support_chats
  chunking:
    strategy: sentence    # Small units for conversational data
```

**Phase C with Mem0 Concepts:**

```bash
# Standard retrieval
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "What issues did user_12345 report last week?" \
  --output-suffix-chunking-strategy sentence \
  --output-suffix-doc-type support_chats \
  --embedder openai \
  --model text-embedding-3-small \
  --vector-store-provider chroma \
  --top-k-chunks 20 \
  --reranker-strategy flashrank \
  --top-n-rerank 7
```

**Advanced: Mem0-like Personalization (Conceptual)**

For true personalization, you'd implement:

1. **User Profile Store**: Separate vector store for user traits/preferences
2. **Query Augmentation**: Before MQR, retrieve user context and augment query
3. **Dynamic Memory Management**: LLM-based agent to add/update/delete memories

```python
# Conceptual implementation in phase_c_query.py

# 1. Retrieve user profile
user_memory = memory_manager.get_user_profile(user_id)
# Returns: {"preferences": ["technical details", "concise answers"], 
#           "past_issues": ["login problems", "API errors"]}

# 2. Augment query
augmented_query = f"{original_query} (User prefers {user_memory['preferences']})"

# 3. Proceed with normal RAG flow
```

**Configuration addition needed:**

```yaml
# config.yaml (future enhancement)
rag:
  memory_layer:
    enabled: true
    user_profile_store:
      provider: chroma
      collection_name: user_profiles
    memory_manager_llm:
      llm_model: gpt-4o-mini
      llm_provider: openrouter
```

---

### 5. Global Sense-Making & Synthesis

**Challenge:** Traditional RAG fails at answering "What are the main themes across all documents?"

**Solution: GraphRAG Approach (Advanced)**

**Conceptual Pipeline:**

```
Phase A/B Extension:
    ↓
[1] ENTITY EXTRACTION
    ├─ Extract (entity, relationship, entity) triples
    └─ Extract claims/facts
    ↓
[2] KNOWLEDGE GRAPH CONSTRUCTION
    └─ Store in Neo4j/NetworkX
    ↓
[3] COMMUNITY DETECTION
    ├─ Apply Leiden algorithm
    └─ Identify semantic communities
    ↓
[4] COMMUNITY SUMMARIZATION
    └─ LLM generates summaries per community
    ↓
Phase C Modification:
    ↓
[5] GLOBAL QUERY RETRIEVAL
    ├─ Search community summaries (not raw chunks)
    └─ Synthesize answer from multiple communities
```

**Implementation Steps:**

1. **Add Phase A.5 (Entity Extraction):**

```bash
# New script: phase_a5_extract_entities.py
PYTHONPATH=. python scripts/phase_a5_extract_entities.py \
  --input-chunks phase_a_processed_parents_<suffix>.jsonl \
  --llm-model gpt-4o-mini \
  --output knowledge_graph_triples.jsonl
```

2. **Build Knowledge Graph:**

```bash
# New script: phase_b5_build_graph.py
PYTHONPATH=. python scripts/phase_b5_build_graph.py \
  --input knowledge_graph_triples.jsonl \
  --graph-db neo4j \
  --detect-communities leiden \
  --output community_summaries.jsonl
```

3. **Query Communities:**

```bash
# Modified phase_c_query.py with --global-query flag
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "What are the main AI safety concerns across all papers?" \
  --global-query \
  --community-summaries community_summaries.jsonl \
  --llm-model gpt-4o-mini
```

---

## Complete Workflow Examples

### Scenario 1: Basic RAG for Internal Documentation

**Goal:** Index company docs, enable Q&A for employees

**Step 1: Setup**

```yaml
# data_sources.yaml
sources:
  - id: company_wiki
    type: folder
    format: md
    path: data/company_docs
    glob: "**/*.md"
    chunking:
      strategy: markdown_header
```

**Step 2: Ingest**

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type company_wiki \
  --log-level INFO
```

**Step 3: Embed (using OpenAI)**

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type company_wiki \
  --vector-store-provider chroma \
  --embedder openai \
  --model text-embedding-3-small \
  --batch-size 50
```

**Step 4: Query**

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "What is our vacation policy for remote employees?" \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type company_wiki \
  --vector-store-provider chroma \
  --embedder openai \
  --model text-embedding-3-small \
  --top-k-chunks 5
```

---

### Scenario 2: Research Paper Analysis with Advanced RAG

**Goal:** Analyze 100+ academic papers, answer complex comparative questions

**Step 1: Setup**

```yaml
# config.yaml
rag:
  reranker:
    strategy: flashrank
    model: ms-marco-TinyBERT-L-2-v2
    top_n: 5
  
  query_transformer:
    strategy: multi_query
    num_queries: 4
    llm_model: gpt-3.5-turbo-0125
    llm_provider: openai
  
  context_distiller:
    strategy: llm_summarizer
    summary_type: key_facts
    llm_model: gpt-3.5-turbo-0125
    llm_provider: openai

# data_sources.yaml
sources:
  - id: ai_papers
    type: folder
    format: pdf
    path: data/papers/ai_safety
    glob: "**/*.pdf"
    chunking:
      strategy: parent_document
      parent_chunk_size: 1500
      child_chunk_size: 200
      child_chunk_overlap: 50
```

**Step 2: Ingest**

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources ai_papers \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type ai_safety_research \
  --log-level INFO
```

**Step 3: Embed (using Gemini for cost efficiency)**

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type ai_safety_research \
  --vector-store-provider chroma \
  --embedder gemini \
  --model gemini-embedding-001 \
  --id-prefix papers_gemini_v1 \
  --batch-size 20 \
  --sleep 0.1 \
  --overwrite
```

**Step 4: Query with full pipeline**

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "Compare the proposed solutions for AI alignment across the recent papers, focusing on their limitations" \
  --output-suffix-chunking-strategy parent_document \
  --output-suffix-doc-type ai_safety_research \
  --vector-store-provider chroma \
  --embedder gemini \
  --model gemini-embedding-001 \
  --id-prefix papers_gemini_v1 \
  --query-transformer-strategy multi_query \
  --reranker-strategy flashrank \
  --top-k-chunks 25 \
  --top-n-rerank 7 \
  --context-distiller-strategy llm_summarizer \
  --top-k-parents 5 \
  --log-level DEBUG
```

**Expected output:**
```
[MQR] Generated 4 query variants
[Retrieval] Retrieved 25 initial chunks
[FlashRank] Reranked to top 7 chunks
[Parents] Fetched 5 parent documents
[Distiller] Summarized context from 5 parents (reduced from 7500 to 1200 tokens)
[LLM] Generating final answer...

Answer: The reviewed papers propose three main approaches to AI alignment:

1. Constitutional AI (Anthropic, 2023): Uses RLHF with explicit principles...
   - Limitation: Requires extensive human feedback and may not scale...

2. Iterated Amplification (Christiano et al., 2023): Decomposes complex tasks...
   - Limitation: Assumes decomposability of all tasks...

3. Value Learning (Russell, 2023): Learns human values through observation...
   - Limitation: Challenge of inferring true values from behavior...

[Sources: paper_042.pdf (p.5-7), paper_089.pdf (p.12-15), paper_127.pdf (p.3-6)]
```

---

### Scenario 3: Multi-Dataset RAG (Different Embedders for Different Data)

**Goal:** Index both text documents AND structured data, optimize embeddings per type

**Step 1: Setup**

```yaml
# data_sources.yaml
sources:
  - id: technical_docs
    type: folder
    format: md
    path: data/docs
    chunking:
      strategy: markdown_header
  
  - id: sales_spreadsheets
    type: folder
    format: xlsx
    path: data/sales
    glob: "**/*.xlsx"
```

**Step 2a: Ingest Text (Markdown)**

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources technical_docs \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type technical_docs
```

**Step 2b: Ingest Structured (Excel)**

```bash
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --include-sources sales_spreadsheets \
  --output-suffix-chunking-strategy structured_rows \
  --output-suffix-doc-type sales_data
```

**Step 3a: Embed Text (with local HF model - fast)**

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type technical_docs \
  --vector-store-provider faiss \
  --faiss-dimension 384 \
  --embedder local_hf \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --id-prefix docs_local
```

**Step 3b: Embed Structured (with OpenAI - better for entities)**

```bash
PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy structured_rows \
  --output-suffix-doc-type sales_data \
  --vector-store-provider faiss \
  --faiss-dimension 1536 \
  --embedder openai \
  --model text-embedding-3-small \
  --id-prefix sales_oai
```

**Step 4a: Query Text Dataset**

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "How do I configure SSL certificates?" \
  --output-suffix-chunking-strategy markdown_header \
  --output-suffix-doc-type technical_docs \
  --vector-store-provider faiss \
  --faiss-dimension 384 \
  --embedder local_hf \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --id-prefix docs_local \
  --top-k-chunks 5
```

**Step 4b: Query Structured Dataset**

```bash
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "Which sales rep had the highest Q4 revenue in the EMEA region?" \
  --output-suffix-chunking-strategy structured_rows \
  --output-suffix-doc-type sales_data \
  --vector-store-provider faiss \
  --faiss-dimension 1536 \
  --embedder openai \
  --model text-embedding-3-small \
  --id-prefix sales_oai \
  --context-distiller-strategy llm_summarizer \
  --top-k-chunks 20
```

---

## Troubleshooting & Best Practices

### Common Issues

#### Issue 1: "Vector store path mismatch"

**Symptom:**
```
Error: Cannot find vector store at path: vector_stores/recursive_character_docs_openai_text_embedding_3_small/
```

**Cause:** Mismatch between Phase B and Phase C arguments

**Solution:** Ensure ALL these match:
```bash
# Phase B
--output-suffix-chunking-strategy recursive_character
--output-suffix-doc-type docs
--embedder openai
--model text-embedding-3-small

# Phase C (must be identical)
--output-suffix-chunking-strategy recursive_character  # ✅ Match
--output-suffix-doc-type docs                          # ✅ Match
--embedder openai                                      # ✅ Match
--model text-embedding-3-small                         # ✅ Match
```

#### Issue 2: "FAISS dimension mismatch"

**Symptom:**
```
Error: Cannot add vectors of dimension 768 to index of dimension 1536
```

**Cause:** Wrong `--faiss-dimension` for your embedding model

**Solution:**
```bash
# Check your model's dimension first:
# gemini-embedding-001 = 768
# text-embedding-3-small = 1536
# all-MiniLM-L6-v2 = 384

# Then specify correctly:
--faiss-dimension 768  # For Gemini
```

#### Issue 3: "ID prefix not found"

**Symptom:**
```
Warning: No chunks found with prefix 'v1_'
```

**Cause:** `--id-prefix` used in Phase B but omitted in Phase C

**Solution:**
```bash
# Phase B
--id-prefix v1_papers

# Phase C (must include same prefix)
--id-prefix v1_papers  # Don't forget!
```

#### Issue 4: "Rate limit errors"

**Symptom:**
```
Error: RateLimitError: Rate limit exceeded
```

**Solution:**
```bash
# Phase B - adjust batch size and sleep
--batch-size 20        # Smaller batches
--sleep 0.5            # Longer pauses (seconds)

# Or use local embeddings:
--embedder local_hf
--model sentence-transformers/all-MiniLM-L6-v2
```

### Best Practices

#### 1. Start Simple, Add Complexity

```bash
# Step 1: Basic pipeline (no advanced features)
PYTHONPATH=. python scripts/phase_c_query.py \
  --query "..." \
  --top-k-chunks 5

# Step 2: Add reranking
--reranker-strategy flashrank \
--top-k-chunks 15 \
--top-n-rerank 5

# Step 3: Add MQR
--query-transformer-strategy multi_query

# Step 4: Add distillation
--context-distiller-strategy llm_summarizer
```

#### 2. Use Logging for Debugging

```bash
# Debug individual phases
--log-level DEBUG

# Trace the full pipeline:
--log-level DEBUG 2>&1 | tee query_debug.log

# Check what MQR generated:
grep "\[MQR\]" query_debug.log

# Check reranking scores:
grep "\[Rerank\]" query_debug.log
```

#### 3. Configuration Management

```bash
# Keep configs version controlled
git add config.yaml data_sources.yaml

# Use environment-specific configs
--config-path configs/production.yaml
--config-path configs/development.yaml

# Document your experiments
echo "v1.2: Added MQR, improved recall by 15%" >> EXPERIMENTS.md
```

#### 4. Cost Optimization

**Embedding costs:**
```bash
# Expensive: OpenAI (but high quality)
--embedder openai --model text-embedding-3-small

# Cheaper: Gemini
--embedder gemini --model gemini-embedding-001

# Free: Local HuggingFace
--embedder local_hf --model sentence-transformers/all-MiniLM-L6-v2
```

**LLM costs:**
```yaml
# config.yaml
llm:
  model: gpt-4o-mini  # Cheap for main generation

rag:
  query_transformer:
    llm_model: gpt-3.5-turbo-0125  # Even cheaper for MQR
  
  context_distiller:
    llm_model: gpt-3.5-turbo-0125  # Even cheaper for summarization
```

#### 5. Evaluation & Iteration

```bash
# Run evaluation suite (Phase D - future)
PYTHONPATH=. python scripts/phase_d_evaluate.py \
  --test-queries test_queries.jsonl \
  --vector-store-path <...> \
  --metrics ragas_all \
  --output evaluation_results.json

# Compare configurations
python scripts/compare_configs.py \
  --config1 results_baseline.json \
  --config2 results_with_mqr.json
```

---

## Appendix: Quick Reference

### Phase Linking Checklist

Use this checklist to ensure phases are properly linked:

```markdown
☐ Phase A complete: JSONL files exist in outputs/
☐ Note exact values:
   - output-suffix-chunking-strategy: _______________
   - output-suffix-doc-type: _______________

☐ Phase B: Use SAME values as Phase A
   - output-suffix-chunking-strategy: ☐ Match
   - output-suffix-doc-type: ☐ Match
   - Note embedder: _______________
   - Note model: _______________
   - Note id-prefix (if used): _______________
   - Note vector-store-provider: _______________
   - Note faiss-dimension (if FAISS): _______________

☐ Phase C: Use SAME values as Phase B
   - output-suffix-chunking-strategy: ☐ Match
   - output-suffix-doc-type: ☐ Match
   - embedder: ☐ Match
   - model: ☐ Match
   - id-prefix: ☐ Match (if used in B)
   - vector-store-provider: ☐ Match
   - faiss-dimension: ☐ Match (if FAISS)
```

### Command Templates

**Basic pipeline:**
```bash
# A → B → C
PYTHONPATH=. python scripts/phase_a_build_chunks.py \
  --output-suffix-chunking-strategy <STRATEGY> \
  --output-suffix-doc-type <DOCTYPE>

PYTHONPATH=. python scripts/phase_b_embed.py \
  --output-suffix-chunking-strategy <STRATEGY> \
  --output-suffix-doc-type <DOCTYPE> \
  --embedder <EMBEDDER> \
  --model <MODEL> \
  --vector-store-provider <PROVIDER>

PYTHONPATH=. python scripts/phase_c_query.py \
  --query "<QUESTION>" \
  --output-suffix-chunking-strategy <STRATEGY> \
  --output-suffix-doc-type <DOCTYPE> \
  --embedder <EMBEDDER> \
  --model <MODEL> \
  --vector-store-provider <PROVIDER>
```

**Advanced pipeline:**
```bash
# Add to Phase C:
  --query-transformer-strategy multi_query \
  --reranker-strategy flashrank \
  --top-k-chunks 20 \
  --top-n-rerank 7 \
  --context-distiller-strategy llm_summarizer \
  --top-k-parents 4
```

---

## Next Steps

1. **Implement Phase D (Evaluation)**
   - Use Ragas for automated evaluation
   - Measure: Context Precision, Context Recall, Faithfulness, Answer Relevance
   - Compare configurations systematically

2. **Add GraphRAG for Global Queries**
   - Entity extraction (Phase A.5)
   - Knowledge graph construction (Phase B.5)
   - Community detection & summarization
   - Modified Phase C for global sense-making

3. **Implement Mem0 Personalization**
   - User profile store
   - Dynamic memory management
   - Query augmentation with user context

4. **Production Deployment**
   - Containerize with Docker
   - API wrapper for Phase C (FastAPI/Flask)
   - Monitoring & logging
   - CI/CD pipeline for vector store updates

---

## Support & Contributing

- **Issues**: Open GitHub issues for bugs or feature requests
- **Documentation**: Keep this guide updated with new features
- **Experiments**: Document findings in `EXPERIMENTS.md`
- **Best Practices**: Share successful configurations in team wiki

---

*Last Updated: February 10, 2026*
*Version: 2.0*