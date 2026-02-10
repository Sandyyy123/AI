# src/rag/components/chunkers.py 

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type, Tuple
import sys
from pathlib import Path

# --- Conditional imports for LangChain text splitters ---
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLSectionSplitter
    from langchain_core.documents import Document as LCDocument # LangChain's Document type for some splitters
# Sentence splitter: use NLTKTextSplitter (available in your install)
    from langchain_text_splitters import NLTKTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None
    NLTKTextSplitter = None
    MarkdownHeaderTextSplitter = None
    HTMLSectionSplitter = None
    LCDocument = None
    logging.warning(
        "LangChain text splitters not installed. "
        "Install with `pip install langchain-text-splitters` for advanced chunking strategies."
    )

# --- Conditional imports for LLM-based chunking ---
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import Runnable
    from langchain_core.output_parsers import StrOutputParser # Crucial for PropositionChunking LLM output
except ImportError:
    PromptTemplate = None
    Runnable = None
    StrOutputParser = None
    logging.warning("LangChain Core components not installed. LLM-based chunkers may not be available. Install `pip install langchain-core`.")

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:
    ChatOpenAI = None
    BaseChatModel = None
    logging.warning(
        "LangChain OpenAI not installed. OpenAI/OpenRouter LLMs for chunking will not be available. "
        "Install `pip install langchain-openai`."
    )

# Fix 2: Define logger at the top, right after imports
logger = logging.getLogger(__name__)

# --- Import Settings from project root config.py ---
try:
    from config import Settings, load_settings
except (ImportError, ValueError) as e:
    logger.error(f"Failed to import Settings due to: {e}. Attempting sys.path adjustment.")
    _proj_root = Path(__file__).resolve().parents[3]
    if str(_proj_root) not in sys.path:
        sys.path.insert(0, str(_proj_root))
        logger.info(f"Added project root '{_proj_root}' to sys.path for config import.")
    try:
        from config import Settings, load_settings
        logger.info(f"Successfully imported config from {_proj_root} after sys.path adjustment.")
    except Exception as e_after_syspath:
        logger.error(f"Still failed to import Settings from config.py after sys.path adjustment: {e_after_syspath}. "
                     "Ensure config.py is in the project root and PYTHON_PATH is set correctly or adjust import path.")
        Settings = None
        load_settings = None


# --- Component Registry Setup ---
# This registry will hold all chunking strategy classes, mapped by a string name
CHUNK_STRATEGY_REGISTRY: Dict[str, Type['BaseChunkingStrategy']] = {}

def register_chunk_strategy(name: str):
    """
    Decorator to register chunking strategies dynamically.
    Enables selection of chunker based on configuration name.
    """
    def decorator(cls: Type['BaseChunkingStrategy']):
        if name in CHUNK_STRATEGY_REGISTRY:
            logger.warning(f"Chunking strategy '{name}' already registered. Overwriting.")
        CHUNK_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

# --- Data Models ---
@dataclass
class Chunk:
    """
    Represents a single chunk of text with its associated metadata.
    id: Unique identifier for the chunk (e.g., doc_id_chunk_001).
    doc_id: Identifier of the parent document.
    text: The actual text content of the chunk.
    metadata: Dictionary for additional information (e.g., page_number, section_title).
    """
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Abstract Base Class for Chunking Strategies ---
class BaseChunkingStrategy(ABC):
    """
    Abstract Base Class for all chunking strategies.
    Defines the interface that all concrete chunking strategies must implement.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        """
        Initializes the chunking strategy.
        Args:
            chunk_size (int): The maximum size of each chunk. (May be advisory for some types like LLM-based)
            chunk_overlap (int): The number of characters/tokens to overlap between chunks.
            **kwargs: Additional configuration parameters specific to the strategy (e.g., separators, LLM).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with config: "
                    f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                    f"kwargs={self.config}")

    @abstractmethod
    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Abstract method to split a given text into a list of Chunk objects.
        Args:
            text (str): The input text to be split.
            document_id (str): The ID of the parent document.
            document_metadata (Dict[str, Any]): Metadata from the parent document.
        Returns:
            List[Chunk]: A list of Chunk objects.
        """
        pass

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Helper to generate a unique chunk ID."""
        return f"{doc_id}_chunk_{chunk_index:04d}"

# --- Concrete Chunking Strategies ---

@register_chunk_strategy("recursive_character")
class RecursiveCharacterChunking(BaseChunkingStrategy):
    """
    Chunking strategy using LangChain's RecursiveCharacterTextSplitter.
    Splits text by a list of characters, attempting to keep paragraphs and sentences together.
    Configurable via 'separators', 'keep_separator', 'length_function'.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "RecursiveCharacterTextSplitter not available. "
                "Please install `langchain-text-splitters` to use this chunking strategy."
            )
        
        separators = self.config.get("separators", ["\n\n", "\n", " ", ""])
        keep_separator = self.config.get("keep_separator", False)
        length_function_str = self.config.get("length_function", "len")
        
        length_function_map = {"len": len} # Can extend this for token length functions if needed
        length_function: Callable[[str], int] = length_function_map.get(length_function_str, len)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=length_function,
        )
        logger.info(f"RecursiveCharacterTextSplitter initialized with "
                    f"separators={separators}, keep_separator={keep_separator}, "
                    f"length_function={length_function_str}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document_id, i)
            chunk_metadata = {
                **document_metadata,
                "chunk_index": i,
                "strategy": "recursive_character",
                "chunk_size_actual": len(chunk_text)
            }
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=chunk_text, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy("sentence")
class SentenceChunking(BaseChunkingStrategy):
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if NLTKTextSplitter is None:
            raise ImportError("NLTKTextSplitter not available. Install: pip install langchain-text-splitters nltk")

        # NLTKTextSplitter doesn't support chunk_overlap/keep_separator
        self.splitter = NLTKTextSplitter(chunk_size=self.chunk_size)

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(Chunk(
                id=self._generate_chunk_id(document_id, i),
                doc_id=document_id,
                text=chunk_text,
                metadata={**document_metadata, "chunk_index": i, "strategy": "sentence", "chunk_size_actual": len(chunk_text)},
            ))
        return chunks

@register_chunk_strategy("markdown_header")
class MarkdownHeaderChunking(BaseChunkingStrategy):
    """
    Chunking strategy for Markdown files using LangChain's MarkdownHeaderTextSplitter.
    Splits text based on specified headers, preserving the hierarchical structure.
    Configurable via 'headers_to_split_on'.
    NOTE: chunk_size and chunk_overlap are less relevant here as splitting is structural.
          The split_text will produce self-contained sections.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if MarkdownHeaderTextSplitter is None or LCDocument is None:
            raise ImportError(
                "MarkdownHeaderTextSplitter not available. "
                "Please install `langchain-text-splitters` to use this chunking strategy."
            )

        headers_to_split_on_config = self.config.get("headers_to_split_on", [])
        if not headers_to_split_on_config:
            logger.warning("No 'headers_to_split_on' provided for MarkdownHeaderChunking. Using default common headers.")
            headers_to_split_on_config = [
                {"_level": 1, "name": "Header 1"},
                {"_level": 2, "name": "Header 2"},
                {"_level": 3, "name": "Header 3"},
            ]
        
        converted_headers: List[Tuple[str, str]] = []
        for h_dict in headers_to_split_on_config:
            # Fix 1: Ensure '_level' is checked, not 'level'
            if '_level' in h_dict and 'name' in h_dict:
                level_prefix = "#" * h_dict['_level']
                converted_headers.append((level_prefix, h_dict['name']))
            else:
                logger.warning(f"Invalid header format in config for MarkdownHeaderChunking: {h_dict}. Expected '_level' and 'name'. Skipping.")
        
        if not converted_headers:
            raise ValueError("No valid headers to split on found for MarkdownHeaderChunking after parsing config.")

        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=converted_headers
        )
        logger.info(f"MarkdownHeaderTextSplitter initialized with headers_to_split_on={converted_headers}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_lc_docs: List[LCDocument] = self.splitter.split_text(text)
        chunks = []
        for i, lc_doc in enumerate(raw_lc_docs):
            chunk_id = self._generate_chunk_id(document_id, i)
            chunk_metadata = {
                **document_metadata,
                **lc_doc.metadata,
                "chunk_index": i,
                "strategy": "markdown_header",
                "chunk_size_actual": len(lc_doc.page_content)
            }
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=lc_doc.page_content, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy("html_section")
class HTMLSectionChunking(BaseChunkingStrategy):
    """
    Chunking strategy for HTML files using LangChain's HTMLSectionSplitter.
    Splits text based on specified HTML tags, attempting to preserve section structure.
    Configurable via 'tags_to_split_on', 'keep_separator'.
    NOTE: Similar to MarkdownHeaderChunking, chunk_size and chunk_overlap are less dominant.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if HTMLSectionSplitter is None or LCDocument is None:
            raise ImportError(
                "HTMLSectionSplitter not available. "
                "Please install `langchain-text-splitters` (`pip install 'langchain-text-splitters[html]'`) "
                "to use this chunking strategy."
            )

        tags_to_split_on_config = self.config.get("tags_to_split_on", [])
        if not tags_to_split_on_config:
            logger.warning("No 'tags_to_split_on' provided for HTMLSectionChunking. Using default common tags.")
            tags_to_split_on_config = ["h1", "h2", "h3", "p", "div", "ul", "ol"]

        # Fix 3: Stable de-duplication for converted_tags
        seen = set()
        converted_tags: List[Tuple[str, str]] = []
        for tag in tags_to_split_on_config:
            t_tuple = (tag, tag) # LangChain HTMLSectionSplitter prefers tuple (tag, metadata_key)
            if t_tuple not in seen:
                seen.add(t_tuple)
                converted_tags.append(t_tuple)

        self.splitter = HTMLSectionSplitter(
            tags_to_split_on=converted_tags,
            keep_separator=self.config.get("keep_separator", False)
        )
        logger.info(f"HTMLSectionSplitter initialized with tags_to_split_on={tags_to_split_on_config}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_lc_docs: List[LCDocument] = self.splitter.split_text(text)
        chunks = []
        for i, lc_doc in enumerate(raw_lc_docs):
            chunk_id = self._generate_chunk_id(document_id, i)
            chunk_metadata = {
                **document_metadata,
                **lc_doc.metadata, # Includes metadata like 'header_level'
                "chunk_index": i,
                "strategy": "html_section",
                "chunk_size_actual": len(lc_doc.page_content)
            }
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=lc_doc.page_content, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy("proposition")
class PropositionChunking(BaseChunkingStrategy):
    """
    Chunking strategy that uses an LLM to break down text into atomic propositions (facts).
    Inspired by: https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/text_splitter.py
    """
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 0, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)

        if PromptTemplate is None or Runnable is None or BaseChatModel is None or StrOutputParser is None:
            raise ImportError(
                "LangChain core components (PromptTemplate, Runnable, StrOutputParser) or LLM (BaseChatModel) not available. "
                "Please install `langchain-core` and an LLM provider like `langchain-openai`."
            )
        if Settings is None or load_settings is None:
            raise ImportError(
                "Project settings (`Settings`, `load_settings`) could not be imported. "
                "Ensure config.py path is correct and accessible."
            )

        self._llm: Optional[BaseChatModel] = kwargs.pop("llm", None)
        if not self._llm:
            logger.warning("No LLM instance provided to PropositionChunking. Attempting to create one from loaded settings.")
            
            app_settings = load_settings()

            if app_settings.OPENROUTER_API_KEY and app_settings.OPENROUTER_BASE_URL:
                if ChatOpenAI is None:
                    raise ImportError("ChatOpenAI not available for LLM initialization. Please install `langchain_openai`.")
                
                self._llm = ChatOpenAI(
                    openai_api_key=app_settings.OPENROUTER_API_KEY,
                    openai_api_base=app_settings.OPENROUTER_BASE_URL,
                    # Model name and temperature can be configured via kwargs or fall back to defaults
                    model_name=self.config.get("model_name", "gpt-4o-mini"),
                    temperature=self.config.get("temperature", 0.0),
                    default_headers={
                        "HTTP-Referer": self.config.get("app_url", "https://your-rag-app.com"),
                        "X-Title": self.config.get("app_name", "RAG-System-Proposition-Chunker"),
                    }
                )
            else:
                raise ValueError(
                    "LLM required for PropositionChunking. "
                    "Provide an 'llm' instance in kwargs at initialization, "
                    "or ensure OPENROUTER_API_KEY and OPENROUTER_BASE_URL are set in environment "
                    "and visible to config.py."
                )

        self.PROPOSITION_PROMPT = PromptTemplate.from_template(
            """
            You are an expert data engineer building a RAG system.
            Your task is to decompose the "Current Text" into simple, atomic propositions (facts).

            ### INPUT DATA
            1. **Document Title**: {title} (Use this for global context)
            2. **Previous Context**: {previous_window} (READ-ONLY. Use this ONLY to resolve pronouns like 'he', 'it', 'they' in the current text.)
            3. **Current Text**: {current_chunk} (EXTRACT facts from this text only.)

            ### RULES
            - **Atomic Facts**: Each sentence must be a standalone fact.
            - **Coreference Resolution**: If 'Current Text' says "He decided...", and 'Previous Context' identifies him as "Elon Musk", write "Elon Musk decided...".
            - **Isolation**: DO NOT create propositions from the 'Previous Context'. Only the 'Current Text'.

            ### OUTPUT
            Return a list of sentences separated by newlines.
            """
        )
        self.proposition_chain: Runnable = self.PROPOSITION_PROMPT | self._llm | StrOutputParser()

        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "!", "?", " "],
            keep_separator=False
        )

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits a document's text into propositions using an LLM.
        The document is first pre-split into manageable chunks to stay within LLM context limits.
        """
        doc_title = document_metadata.get("title", "Untitled Document")
        
        # HTML/MD raw text is expected for *structural* chunkers. If the input text here
        # is already cleaned/extracted, it's fine for recursive/sentence/proposition.
        # This function operates on the 'text' property of the Document object,
        # which will be pre-cleaned by phase_a_build_chunks.py
        
        pre_chunks = self.pre_splitter.split_text(text)

        all_propositions: List[str] = []
        previous_window = ""

        for i, small_chunk in enumerate(pre_chunks):
            try:
                response_content = self.proposition_chain.invoke({
                    "title": doc_title,
                    "previous_window": previous_window,
                    "current_chunk": small_chunk
                })
                
                propositions = [p.strip() for p in response_content.split("\n") if p.strip()]
                all_propositions.extend(propositions)

                previous_window = small_chunk
            except Exception as e:
                logger.error(f"Error during proposition generation for document {document_id}, pre-chunk {i}: {e}", exc_info=True)
                all_propositions.append(small_chunk) # Fallback to original small chunk if LLM fails
            
        final_chunks = []
        for i, prop_text in enumerate(all_propositions):
            chunk_meta = {
                **document_metadata,
                "chunk_index": i,
                "strategy": "proposition",
                "original_doc_title": doc_title,
                "chunk_size_actual": len(prop_text)
            }
            final_chunks.append(
                Chunk(
                    id=self._generate_chunk_id(document_id, i),
                    doc_id=document_id,
                    text=prop_text,
                    metadata=chunk_meta
                )
            )
        return final_chunks