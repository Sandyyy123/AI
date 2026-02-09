"""
Phase A: YAML-driven ingestion -> cleaning -> chunking -> JSONL outputs.

What this version adds (Option 1):
✅ type: folder expansion (glob -> many file sources)
✅ real PDF extraction (pypdf)
✅ real DOCX extraction (python-docx)
✅ CSV/TSV extraction (each row becomes a text chunk with headers).
✅ Excel (.xlsx, .xls) extraction (each row/sheet becomes a text chunk with headers).
✅ Markdown and HTML extraction
✅ Chunking strategies:
    # RecursiveCharacterTextSplitter for intelligent splitting based on configurable separators.
    # Sentence-based pre-processing for better contextual chunks.
    # Metadata enrichment: Page numbers, section hints, filename, etc., stored with chunks.
    # Error handling and warnings for missing dependencies.


You can now add PDFs/DOCX as folder entries in data_sources.yaml and run downstream.
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib

import yaml
import warnings 
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Optional dependencies (graceful)
# -----------------------------
# PDF
try:
    from pypdf import PdfReader  # type: ignore
except ImportError:
    PdfReader = None
    logger.warning("pypdf not installed. PDF loading disabled. Install: pip install pypdf")

# DOCX
try:
    from docx import Document as DocxDocument  # type: ignore
except ImportError:
    DocxDocument = None
    logger.warning("python-docx not installed. DOCX loading disabled. Install: pip install python-docx")

# Requests
try:
    import requests  # type: ignore
except ImportError:
    requests = None
    logger.warning("requests not installed. URL loading disabled. Install: pip install requests")

# HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
except ImportError:
    BeautifulSoup = None
    logger.warning("beautifulsoup4 not installed. HTML parsing will be raw. Install: pip install beautifulsoup4")

# Markdown parsing (better)
try:
    from markdown_it import MarkdownIt  # type: ignore
    from mdit_py_plugins.front_matter import front_matter_plugin  # type: ignore
except ImportError:
    MarkdownIt = None
    front_matter_plugin = None
    logger.warning("markdown-it-py / mdit-py-plugins not installed. Markdown parsing will be raw. "
                   "Install: pip install markdown-it-py mdit-py-plugins")

# CSV / Excel
try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None
    logger.warning("pandas not installed. CSV/Excel loading disabled. Install: pip install pandas")

try:
    import openpyxl  # type: ignore
except ImportError:
    openpyxl = None
    # (pandas can still read xls via xlrd etc., but we keep it simple)
    logger.warning("openpyxl not installed. Excel (.xlsx) loading may fail. Install: pip install openpyxl")

# LangChain splitter
# --- optional recursive chunking (LangChain splitters) ---
RecursiveCharacterTextSplitter = None
try:
    # Newer / recommended package
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        # Older LangChain path (some versions)
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None
        logger.warning(
            "Recursive chunking disabled (missing langchain_text_splitters). "
            "Install: pip install langchain-text-splitters"
        )


# NLTK sentence tokenization
try:
    import nltk  # type: ignore
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)
    sentence_tokenizer = nltk.sent_tokenize
except Exception:
    nltk = None
    sentence_tokenizer = lambda text: text.split(". ")
    logger.warning("NLTK not available or punkt missing. Sentence chunking will be basic.")



# -----------------------------
# Data models
# -----------------------------
@dataclass
class Document:
    id: str            # Unique identifier for the document
    title: str         # Human-readable title
    source_type: str   # file/url
    source: str        # abs path or url
    text: str          # Full document text
    meta: Dict[str, Any] = field(default_factory=dict) # Other metadata (tags, etc.)


@dataclass
class Chunk:
    doc_id: str     # ID of the parent document
    chunk_id: str   # Unique identifier for the chunk (e.g., doc_id_chunk_001)
    text: str       # The actual chunk text
    meta: Dict[str, Any] = field(default_factory=dict) # Metadata specific to the chunk (page, section, etc.)


# -----------------------------
# Cleaning
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Normalizes text by replacing various whitespace characters,
    removing excessive blank lines, and stripping leading/trailing whitespace.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Common zero-width and non-breaking spaces
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = text.replace("\t", " ")  # Replace tabs with a single space
    text = re.sub(r"[^\S\n]+", " ", text)  # Replace multiple spaces/tabs with a single space

    lines = [ln.strip() for ln in text.split("\n")]
    out: List[str] = []
    blank = 0  # Check for empty string after stripping
    for ln in lines:
        if ln == "":
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln)

    return "\n".join(out).strip()


# -----------------------------
# Folder expansion
# -----------------------------
def slugify(s: str) -> str:
    """Converts a string to a slug (lowercase, alphanumeric, hyphen-separated)."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)  # Remove non-alphanumeric chars (except spaces/hyphens)
    s = re.sub(r"[\s_-]+", "_", s)
    return s.strip("_")


def expand_folder_sources(project_root: Path, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded: List[Dict[str, Any]] = []

    for s in sources:
        if s.get("type") != "folder":
            expanded.append(s)
            continue

        folder_path = s.get("path")
        pattern = s.get("glob")
        fmt = s.get("format")

        if not folder_path or not pattern or not fmt:
            logger.error(f"Folder source '{s.get('id','N/A')}' must include path, glob, format. Skipping.")
            continue

        folder_abs = (project_root / folder_path).resolve()
        if not folder_abs.exists() or not folder_abs.is_dir():
            logger.warning(f"Folder path invalid for '{s.get('id')}': {folder_abs}. Skipping.")
            continue

        matches = sorted(folder_abs.glob(pattern))
        if not matches:
            logger.info(f"No files matched for folder source '{s.get('id')}' with glob '{pattern}'")
            continue

        base_id = s.get("id", slugify(folder_abs.name))
        base_title = s.get("title", base_id.replace("_", " ").title())
        tags = s.get("tags", [])
        loader_options = s.get("loader_options", {})
        meta_extra = {k: v for k, v in s.items() if k not in {"id","type","path","glob","format","title","tags","loader_options"}}

        for fp in matches:
            if fp.is_dir():
                continue

            rel = str(fp.relative_to(project_root))
            h = hashlib.md5(rel.encode()).hexdigest()[:8]  # stable per relative path
            derived_id = f"{base_id}__{h}"

            expanded.append({
                **meta_extra,
                "id": derived_id,
                "type": "file",
                "format": fmt,
                "path": rel,
                "title": f"[{base_title}] {fp.name}",
                "tags": tags,
                "loader_options": loader_options,
            })

    return expanded

# -----------------------------
# File loaders by format
# -----------------------------
# -----------------------------
# Loaders
# -----------------------------
def load_txt_like(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.error(f"Error reading text file {path}: {e}")
        return ""


def load_pdf(path: Path) -> str:
    if PdfReader is None:
        raise ImportError("pypdf missing")
    reader = PdfReader(str(path))
    parts: List[str] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text()
        if txt:
            parts.append(f"[PAGE {i}]\n{txt}")
    return "\n\n".join(parts)


def load_docx(path: Path) -> str:
    if DocxDocument is None:
        raise ImportError("python-docx missing")
    doc = DocxDocument(str(path))
    paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(paras)


def load_html(path: Path) -> str:
    raw = load_txt_like(path)
    if BeautifulSoup is None:
        logger.warning(f"bs4 not installed. Loading HTML raw: {path}")
        return raw
    try:
        soup = BeautifulSoup(raw, "html.parser")
        for t in soup(["script", "style"]):
            t.extract()
        return normalize_text(soup.get_text())
    except Exception as e:
        logger.error(f"HTML parse error {path}: {e}")
        return raw


def load_markdown(path: Path) -> str:
    raw = load_txt_like(path)
    if MarkdownIt and front_matter_plugin:
        try:
            md = MarkdownIt().use(front_matter_plugin)
            tokens = md.parse(raw)
            # Simple text extraction from tokens; fallback to raw if empty
            text = "".join(t.content for t in tokens if getattr(t, "content", None))
            return normalize_text(text or raw)
        except Exception as e:
            logger.warning(f"Markdown parse failed for {path}: {e}. Using raw.")
    return raw


def _df_rows_to_text(df: Any, table_name: str, row_limit: Optional[int] = None) -> List[str]:
    if df is None or df.empty:
        return []
    headers = df.columns.tolist()
    rows = df.head(row_limit) if row_limit else df

    out: List[str] = []
    for idx, row in rows.iterrows():
        parts = [f"Contents of {table_name}, row {idx+1}."]
        for h in headers:
            val = row[h]
            if pd is not None and pd.isna(val):
                val = "N/A"
            parts.append(f"{h}: {val}.")
        out.append(normalize_text(" ".join(parts)))
    return out


def load_csv(path: Path, loader_options: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    if pd is None:
        raise ImportError("pandas missing")
    loader_options = loader_options or {}
    df = pd.read_csv(path, **loader_options)
    table_name = path.stem.replace("_", " ").title()
    rows = _df_rows_to_text(df, table_name=table_name)
    full = f"Table from {path.name}:\n" + "\n---\n".join(rows)
    return normalize_text(full), rows


def load_tsv(path: Path, loader_options: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    if pd is None:
        raise ImportError("pandas missing")
    loader_options = loader_options or {}
    if "sep" not in loader_options:
        loader_options["sep"] = "\t"
    df = pd.read_csv(path, **loader_options)
    table_name = path.stem.replace("_", " ").title()
    rows = _df_rows_to_text(df, table_name=table_name)
    full = f"Table from {path.name}:\n" + "\n---\n".join(rows)
    return normalize_text(full), rows


def load_excel(path: Path, loader_options: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    if pd is None:
        raise ImportError("pandas missing")

    loader_options = loader_options or {}
    all_rows: List[str] = []
    combined: List[str] = []

    suffix = path.suffix.lower()

    # Force correct engine depending on file type
    engine = None

    if suffix == ".xlsx":
        if openpyxl is None:
            raise ImportError("openpyxl required for .xlsx files. Install: pip install openpyxl")
        engine = "openpyxl"

    elif suffix == ".xls":
        try:
            import xlrd  # type: ignore
            engine = "xlrd"
        except ImportError:
            raise ImportError("xlrd required for .xls files. Install: pip install xlrd")

    try:
        xls = pd.ExcelFile(path, engine=engine)
    except Exception as e:
        logger.error(f"Failed to open Excel file {path}: {e}")
        return "", []

    for sheet in xls.sheet_names:
        try:
            df = xls.parse(sheet, **loader_options)

            name = f"{path.stem.replace('_',' ').title()} - Sheet: {sheet}"
            rows = _df_rows_to_text(df, table_name=name)

            if rows:
                combined.append(f"--- Sheet: {sheet} ---\n" + "\n---\n".join(rows))
                all_rows.extend(rows)

        except Exception as e:
            logger.warning(f"Failed reading sheet '{sheet}' in {path}: {e}")

    return normalize_text("\n\n".join(combined)), all_rows



def load_url(url: str, timeout: int = 30) -> str:
    if requests is None:
        raise ImportError("requests missing")

    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                )
            }
        )

        r.raise_for_status()

    except Exception as e:
        logger.error(f"URL fetch failed: {url} -> {e}")
        return ""

    ctype = (r.headers.get("Content-Type") or "").lower()

    try:
        if "text/html" in ctype and BeautifulSoup is not None:
            soup = BeautifulSoup(r.text, "html.parser")

            for t in soup(["script", "style", "noscript"]):
                t.extract()

            return normalize_text(soup.get_text())

        return normalize_text(r.text)

    except Exception as e:
        logger.error(f"URL parsing failed: {url} -> {e}")
        return ""


def load_file_by_format(
    project_root: Path, rel_path: str, fmt: str, loader_options: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str], str]:
    """
    Returns:
      full_text, pre_chunks, source_abs_path
    pre_chunks is used for CSV/TSV/Excel row-level chunks.
    """
    p = (project_root / rel_path).resolve()
    if not p.exists():
        logger.error(f"File not found: {p}")
        return "", [], str(p)

    fmt = (fmt or "").lower()
    pre_chunks: List[str] = []

    try:
        if fmt in {"txt"}:
            return load_txt_like(p), [], str(p)
        if fmt in {"md", "markdown"}:
            return load_markdown(p), [], str(p)
        if fmt == "html":
            return load_html(p), [], str(p)
        if fmt == "pdf":
            return load_pdf(p), [], str(p)
        if fmt == "docx":
            return load_docx(p), [], str(p)
        if fmt == "csv":
            full, rows = load_csv(p, loader_options)
            return full, rows, str(p)
        if fmt == "tsv":
            full, rows = load_tsv(p, loader_options)
            return full, rows, str(p)
        if fmt in {"xlsx", "xls"}:
            full, rows = load_excel(p, loader_options)
            return full, rows, str(p)

        logger.warning(f"Unsupported format '{fmt}' for {p}. Loading as text.")
        return load_txt_like(p), [], str(p)

    except ImportError as e:
        logger.error(f"Missing dependency for {fmt} ({p}): {e}")
        return "", [], str(p)
    except Exception as e:
        logger.error(f"Error loading {p} as {fmt}: {e}")
        return "", [], str(p)

# -----------------------------
# Chunkers (keep simple for Phase A; you can expand)
# -----------------------------
def chunk_char_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return [c for c in out if c]


def chunk_recursive(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], str]:
    if RecursiveCharacterTextSplitter is None:
        return chunk_char_window(text, chunk_size, overlap), "fallback_char_window:no_langchain"
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text), "used_langchain_recursive"


def chunk_sentence_level(text: str, chunk_size: int, overlap: int) -> Tuple[List[str], str]:
    if not text:
        return [], "empty"
    sents = sentence_tokenizer(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0

    for s in sents:
        sl = len(s) + 1
        if cur and cur_len + sl > chunk_size:
            chunks.append(" ".join(cur).strip())
            # overlap: keep tail sentences whose total length <= overlap
            tail: List[str] = []
            tail_len = 0
            for t in reversed(cur):
                tl = len(t) + 1
                if tail_len + tl <= overlap:
                    tail.insert(0, t)
                    tail_len += tl
                else:
                    break
            cur = tail[:]
            cur_len = sum(len(x) + 1 for x in cur)

        cur.append(s)
        cur_len += sl

    if cur:
        chunks.append(" ".join(cur).strip())

    return [c for c in chunks if c], ("used_nltk_sentence" if nltk is not None else "fallback_naive_sentence")


# -----------------------------
# Selection helper
# -----------------------------
def select_sources(sources: List[Dict[str, Any]], ids: Optional[List[str]], exclude: Optional[List[str]]) -> List[Dict[str, Any]]:
    wanted = set(ids) if ids else None
    excluded = set(exclude) if exclude else set()

    selected: List[Dict[str, Any]] = []
    for s in sources:
        sid = s.get("id")
        if not sid:
            raise SystemExit("A source is missing required field: id")
        if wanted is not None and sid not in wanted:
            continue
        if sid in excluded:
            continue
        selected.append(s)
    return selected

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A: YAML ingestion -> clean -> chunk -> JSONL")
    parser.add_argument("--ids", nargs="*", default=None, help="Process only these IDs")
    parser.add_argument("--exclude", nargs="*", default=None, help="Exclude these IDs")
    parser.add_argument("--strategy", choices=["char", "recursive", "sentence"], default="recursive")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=30, help="URL timeout (sec)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--data_sources", type=str, default="data_sources.yaml")
    args = parser.parse_args()

    root = Path(".").resolve()
    manifest_path = (root / args.data_sources).resolve()
    if not manifest_path.exists():
        raise SystemExit(f"{args.data_sources} not found in project root.")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources", [])
    if not sources:
        raise SystemExit("No sources found under 'sources:' in YAML.")

    # Expand folder sources BEFORE selection
    sources = expand_folder_sources(root, sources)

    # Apply selection
    selected = select_sources(sources, args.ids, args.exclude)
    if not selected:
        available = [s.get("id") for s in sources]
        raise SystemExit(f"No sources selected. Available IDs: {available}")

    out_dir = (root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = "all" if not args.ids else "_".join(sorted(args.ids))
    out_docs = out_dir / f"documents_{label}.jsonl"
    out_chunks = out_dir / f"chunks_{label}_{args.strategy}.jsonl"

    documents: List[Document] = []

    # -----------------------------
    # Load + normalize documents
    # -----------------------------
    for s in selected:
        sid = s["id"]
        stype = s.get("type")
        title = s.get("title", sid)
        fmt = s.get("format")
        loader_options = s.get("loader_options", {}) or {}

        meta = {k: v for k, v in s.items() if k not in {"path", "url", "type", "title"}}
        meta["title"] = title
        meta["format"] = fmt

        full_text = ""
        pre_chunks: List[str] = []
        src = ""

        if stype == "file":
            if "path" not in s or not fmt:
                raise SystemExit(f"File source '{sid}' must include 'path' and 'format'.")
            logger.info(f"Loading file {s['path']} (format: {fmt})")
            full_text, pre_chunks, src = load_file_by_format(root, s["path"], fmt, loader_options)

        elif stype == "url":
            if "url" not in s:
                raise SystemExit(f"URL source '{sid}' missing 'url'.")
            logger.info(f"Loading URL {s['url']}")
            full_text = load_url(s["url"], timeout=args.timeout)
            src = s["url"]

        else:
            raise SystemExit(f"Unsupported type '{stype}' for source '{sid}' (use file/url/folder).")

        full_text = normalize_text(full_text)
        if not full_text and not pre_chunks:
            logger.warning(f"No text extracted for {sid}. Skipping.")
            continue

        doc = Document(
            id=sid,
            title=title,
            source_type=stype,
            source=src,
            text=full_text,
            meta=meta
        )

        # Attach pre-chunks so chunking stage can pick them up (CSV/TSV/Excel)
        if pre_chunks:
            doc.meta["_pre_chunks"] = pre_chunks
            doc.meta["_pre_chunks_count"] = len(pre_chunks)
            doc.meta["_structured_rows"] = True

        documents.append(doc)

    # -----------------------------
    # Write documents JSONL
    # -----------------------------
    with out_docs.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    # -----------------------------
    # Chunk and write chunks JSONL
    # -----------------------------
    with out_chunks.open("w", encoding="utf-8") as f:
        for d in documents:
            # 1) Structured data path (CSV/TSV/Excel rows), if present
            pre_chunks = d.meta.get("_pre_chunks")
            if pre_chunks:
                parts = pre_chunks
                strategy_note = "structured_rows"
            else:
                # 2) Unstructured text path (PDF/DOCX/MD/HTML/TXT/URL)
                if args.strategy == "char":
                    parts = chunk_char_window(d.text, args.chunk_size, args.overlap)
                    strategy_note = "char_window"
                elif args.strategy == "sentence":
                    parts, strategy_note = chunk_sentence_level(d.text, args.chunk_size, args.overlap)
                else:
                    # Use the local function defined in THIS file
                    parts, strategy_note = chunk_recursive(d.text, args.chunk_size, args.overlap)

            for i, txt in enumerate(parts, start=1):
                txt = normalize_text(txt)
                if not txt:
                    continue

                # Stable chunk id with hash
                h = hashlib.md5(txt.encode()).hexdigest()[:8]
                chunk_id = f"{d.id}::c{i:04d}::{h}"
                
                clean_meta = {k: v for k, v in d.meta.items() if not k.startswith("_")}

                meta_out = {
                    **clean_meta,
                    "source_type": d.source_type,
                    "source": d.source,
                    "chunk_index": i,
                    "n_chars": len(txt),
                    "strategy": args.strategy,
                    "strategy_note": strategy_note,
                    "chunk_size": args.chunk_size,
                    "overlap": args.overlap,
                }

                f.write(
                    json.dumps(
                        asdict(Chunk(doc_id=d.id, chunk_id=chunk_id, text=txt, meta=meta_out)),
                        ensure_ascii=False
                    ) + "\n"
                )

    print("✅ Selected IDs (final):", [d.id for d in documents])
    print("✅ Wrote:")
    print(" -", out_docs)
    print(" -", out_chunks)
    print("✅ Folder sources were expanded BEFORE selection (PDF/DOCX now supported).")



if __name__ == "__main__":
    main()
