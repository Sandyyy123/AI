# src/rag/components/chunking.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable

# Conditional imports for text splitters
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTextSplitter
    from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLSectionSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None
    SentenceTextSplitter = None
    MarkdownHeaderTextSplitter = None
    HTMLSectionSplitter = None
    logging.warning(
        class="tok-str">"LangChain text splitters not installed. "
        class="tok-str">"Install with `pip install langchain-text-splitters` for advanced chunking."
    )

logger = logging.getLogger(__name__)

# --- Component Registry Setup ---
CHUNK_STRATEGY_REGISTRY = {}

def register_chunk_strategy(name: str):
    class="tok-str">"""
    Decorator to register chunking strategies dynamically.
    class="tok-str">"""
    def decorator(cls):
        if name in CHUNK_STRATEGY_REGISTRY:
            logger.warning(class="tok-str">f"Chunking strategy '{name}' already registered. Overwriting.")
        CHUNK_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator

# --- Data Models ---
@dataclass
class Chunk:
    class="tok-str">"""
    Represents a single chunk of text with its associated metadata.
    id: Unique identifier for the chunk (e.g., doc_id_chunk_001)
    doc_id: Identifier of the parent document
    text: The actual text content of the chunk
    metadata: Dictionary for additional information (e.g., page_number, section_title)
    class="tok-str">"""
    id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- Abstract Base Class for Chunking Strategies ---
class BaseChunkingStrategy(ABC):
    class="tok-str">"""
    Abstract Base Class for all chunking strategies.
    Defines the interface that all concrete chunking strategies must implement.
    class="tok-str">"""
    @abstractmethod
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        class="tok-str">"""
        Initializes the chunking strategy.
        Args:
            chunk_size (int): The maximum size of each chunk.
            chunk_overlap (int): The number of characters/tokens to overlap between chunks.
            **kwargs: Additional configuration parameters specific to the strategy.
        class="tok-str">"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.config = kwargs
        logger.info(class="tok-str">f"Initializing {self.__class__.__name__} with config: "
                    class="tok-str">f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
                    class="tok-str">f"kwargs={self.config}")

    @abstractmethod
    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        class="tok-str">"""
        Abstract method to split a given text into a list of Chunk objects.
        Args:
            text (str): The input text to be split.
            document_id (str): The ID of the parent document.
            document_metadata (Dict[str, Any]): Metadata from the parent document.
        Returns:
            List[Chunk]: A list of Chunk objects.
        class="tok-str">"""
        pass

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        class="tok-str">""class="tok-str">"Helper to generate a unique chunk ID."class="tok-str">""
        return class="tok-str">f"{doc_id}_chunk_{chunk_index:03d}"

# --- Concrete Chunking Strategies ---

@register_chunk_strategy(class="tok-str">"recursive_character")
class RecursiveCharacterChunking(BaseChunkingStrategy):
    class="tok-str">"""
    Chunking strategy using LangChain's RecursiveCharacterTextSplitter.
    Splits text by a list of characters, attempting to keep paragraphs and sentences together.
    Configurable via class="tok-str">'separators', class="tok-str">'keep_separator', class="tok-str">'length_function'.
    class="tok-str">"""
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                class="tok-str">"RecursiveCharacterTextSplitter not available. "
                class="tok-str">"Please install `langchain-text-splitters` to use this chunking strategy."
            )
        
        separators = self.config.get(class="tok-str">"separators", [class="tok-str">"\n\n", class="tok-str">"\n", class="tok-str">" ", class="tok-str">""])
        keep_separator = self.config.get(class="tok-str">"keep_separator", False)
        length_function_str = self.config.get(class="tok-str">"length_function", class="tok-str">"len")
        
        # Map string to actual function for length_function
        length_function_map = {class="tok-str">"len": len} class=class="tok-str">"tok-cmt"># Add more if needed, e.g., tokenizers
        length_function: Callable[[str], int] = length_function_map.get(length_function_str, len)

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            keep_separator=keep_separator,
            length_function=length_function,
        )
        logger.info(class="tok-str">f"RecursiveCharacterTextSplitter initialized with "
                    class="tok-str">f"separators={separators}, keep_separator={keep_separator}, "
                    class="tok-str">f"length_function={length_function_str}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document_id, i)
            # Combine document metadata with chunk-specific metadata (if any from splitter, though RCTS typically doesn't add much)
            chunk_metadata = {**document_metadata, class="tok-str">"chunk_index": i}
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=chunk_text, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy(class="tok-str">"sentence")
class SentenceChunking(BaseChunkingStrategy):
    class="tok-str">"""
    Chunking strategy using LangChain's SentenceTextSplitter.
    Ensures that chunks respect sentence boundaries and can be configured to merge sentences
    up to a maximum chunk size.
    class="tok-str">"""
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if SentenceTextSplitter is None:
            raise ImportError(
                class="tok-str">"SentenceTextSplitter not available. "
                class="tok-str">"Please install `langchain-text-splitters` to use this chunking strategy."
            )

        keep_separator = self.config.get(class="tok-str">"keep_separator", False)
        
        self.splitter = SentenceTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            keep_separator=keep_separator,
        )
        logger.info(class="tok-str">f"SentenceTextSplitter initialized with keep_separator={keep_separator}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document_id, i)
            chunk_metadata = {**document_metadata, class="tok-str">"chunk_index": i}
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=chunk_text, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy(class="tok-str">"markdown_header")
class MarkdownHeaderChunking(BaseChunkingStrategy):
    class="tok-str">"""
    Chunking strategy for Markdown files using LangChain's MarkdownHeaderTextSplitter.
    Splits text based on specified headers, preserving the hierarchical structure.
    Configurable via class="tok-str">'headers_to_split_on', class="tok-str">'keep_separator'.
    NOTE: chunk_size and chunk_overlap are less relevant here as splitting is structural.
          The split_text will produce self-contained sections.
    class="tok-str">"""
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if MarkdownHeaderTextSplitter is None:
            raise ImportError(
                class="tok-str">"MarkdownHeaderTextSplitter not available. "
                class="tok-str">"Please install `langchain-text-splitters` to use this chunking strategy."
            )

        headers_to_split_on = self.config.get(class="tok-str">"headers_to_split_on", [])
        if not headers_to_split_on:
            logger.warning(class="tok-str">"No headers_to_split_on provided for MarkdownHeaderChunking. "
                           class="tok-str">"This may result in no splitting or a single large chunk.")
            # Default to some common headers if not provided
            headers_to_split_on = [
                (class="tok-str">"class="tok-cmtclass="tok-str">">#", class="tok-str">"Header class="tok-num">1"),
                (class="tok-str">"class="tok-cmtclass="tok-str">">##", class="tok-str">"Header class="tok-num">2"),
                (class="tok-str">"class="tok-cmtclass="tok-str">">###", class="tok-str">"Header class="tok-num">3"),
            ]
        
        # MarkdownHeaderTextSplitter expects tuples or list of tuples
        # Our config uses dicts for better readability in YAML
        if isinstance(headers_to_split_on[class="tok-num">0], dict):
            # Convert list of dicts to list of tuples for the splitter
            converted_headers = []
            for h_dict in headers_to_split_on:
                if class="tok-str">'_level' in h_dict and class="tok-str">'name' in h_dict:
                    # Using Markdown level (e.g., #, ##) as header, and name as metadata key
                    level_prefix = class="tok-str">"class="tok-cmtclass="tok-str">">#" * h_dict[class="tok-str">'_level']
                    converted_headers.append((level_prefix, h_dict[class="tok-str">'name']))
                else:
                    logger.warning(class="tok-str">f"Invalid header format in config: {h_dict}. Expected '_level' and 'name'.")
            if not converted_headers:
                raise ValueError(class="tok-str">"No valid headers to split on found for MarkdownHeaderChunking.")
            headers_to_split_on = converted_headers

        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            # LangChain's MarkdownHeaderTextSplitter doesn't directly support `keep_separator` boolean
            # It inherently keeps the header as part of the metadata.
            # We'll need to decide how to handle this if true `keep_separator` is desired.
            # For now, we rely on its default behavior which includes the header in metadata.
        )
        logger.info(class="tok-str">f"MarkdownHeaderTextSplitter initialized with headers_to_split_on={headers_to_split_on}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_lc_docs = self.splitter.split_text(text)
        chunks = []
        for i, lc_doc in enumerate(raw_lc_docs):
            chunk_id = self._generate_chunk_id(document_id, i)
            # lc_doc.metadata contains the extracted header data
            # Add other document metadata and chunk index
            chunk_metadata = {**document_metadata, **lc_doc.metadata, class="tok-str">"chunk_index": i}
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=lc_doc.page_content, metadata=chunk_metadata))
        return chunks

@register_chunk_strategy(class="tok-str">"html_section")
class HTMLSectionChunking(BaseChunkingStrategy):
    class="tok-str">"""
    Chunking strategy for HTML files using LangChain's HTMLSectionSplitter.
    Splits text based on specified HTML tags, attempting to preserve section structure.
    Configurable via class="tok-str">'tags_to_split_on', class="tok-str">'keep_separator'.
    NOTE: Similar to MarkdownHeaderChunking, chunk_size and chunk_overlap are less dominant.
    class="tok-str">"""
    def __init__(self, chunk_size: int, chunk_overlap: int, **kwargs):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        if HTMLSectionSplitter is None:
            raise ImportError(
                class="tok-str">"HTMLSectionSplitter not available. "
                class="tok-str">"Please install `langchain-text-splitters` (`pip install 'langchain-text-splitters[html]'`) "
                class="tok-str">"to use this chunking strategy."
            )

        tags_to_split_on = self.config.get(class="tok-str">"tags_to_split_on", [])
        if not tags_to_split_on:
            logger.warning(class="tok-str">"No tags_to_split_on provided for HTMLSectionChunking. "
                           class="tok-str">"This may result in no splitting or a single large chunk.")
            tags_to_split_on = [class="tok-str">"h1", class="tok-str">"h2", class="tok-str">"h3", class="tok-str">"p", class="tok-str">"div"] class=class="tok-str">"tok-cmt"># Default common tags

        # HTMLSectionSplitter expects a list of tuples (tag, metadata_key) or just tags
        # We'll use just tags for simplicity, metadata_key can be added if needed
        converted_tags = [(tag, tag) for tag in tags_to_split_on] class=class="tok-str">"tok-cmt"># Map tag to itself as metadata key if not specified
        
        self.splitter = HTMLSectionSplitter(
            tags_to_split_on=converted_tags,
            keep_separator=self.config.get(class="tok-str">"keep_separator", False)
        )
        logger.info(class="tok-str">f"HTMLSectionSplitter initialized with tags_to_split_on={tags_to_split_on}")

    def split_text(self, text: str, document_id: str, document_metadata: Dict[str, Any]) -> List[Chunk]:
        raw_lc_docs = self.splitter.split_text(text)
        chunks = []
        for i, lc_doc in enumerate(raw_lc_docs):
            chunk_id = self._generate_chunk_id(document_id, i)
            # lc_doc.metadata contains information about the tags, which can be useful
            chunk_metadata = {**document_metadata, **lc_doc.metadata, class="tok-str">"chunk_index": i}
            chunks.append(Chunk(id=chunk_id, doc_id=document_id, text=lc_doc.page_content, metadata=chunk_metadata))
        return chunks

# --- Chunking Factory ---
class ChunkingStrategyFactory:
    class="tok-str">"""
    Factory class to create chunking strategy instances based on a configuration.
    class="tok-str">"""
    @staticmethod
    def get_strategy(strategy_name: str, chunk_size: int, chunk_overlap: int, strategy_config: Dict[str, Any]) -> BaseChunkingStrategy:
        class="tok-str">"""
        Retrieves and initializes a chunking strategy.
        Args:
            strategy_name (str): The name of the chunking strategy (e.g., class="tok-str">"recursive_character").
            chunk_size (int): Global chunk size.
            chunk_overlap (int): Global chunk overlap.
            strategy_config (Dict[str, Any]): Configuration specific to this chunking strategy.
        Returns:
            BaseChunkingStrategy: An instance of the requested chunking strategy.
        Raises:
            ValueError: If the strategy is not registered or cannot be initialized.
        class="tok-str">"""
        strategy_cls = CHUNK_STRATEGY_REGISTRY.get(strategy_name)
        if not strategy_cls:
            raise ValueError(class="tok-str">f"Unknown chunking strategy: {strategy_name}. "
                             class="tok-str">f"Available strategies: {list(CHUNK_STRATEGY_REGISTRY.keys())}")
        
        try:
            return strategy_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **strategy_config)
        except ImportError as ie:
            logger.error(class="tok-str">f"Failed to initialize chunking strategy '{strategy_name}' due to missing dependency: {ie}")
            raise
        except Exception as e:
            logger.error(class="tok-str">f"Failed to initialize chunking strategy '{strategy_name}' with config {strategy_config}: {e}")
            raise