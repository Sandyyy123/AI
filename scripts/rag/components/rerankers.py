# scripts/rag/components/rerankers.py (CORRECTED PATH)

import logging
from typing import List, Dict, Any, Optional, Type

# --- Corrected Import: Assuming 'config' is directly importable from project root ---
try:
    from config import Settings, load_settings
    logger.info("Successfully imported Settings, load_settings directly from config.py module for rerankers.")
except ImportError as e:
    logger.critical(f"FATAL: Failed to import Settings from config.py in rerankers module: {e}. "
                    "Ensure config.py is directly importable (e.g., project root in PYTHONPATH). "
                    "Configuration for rerankers will be limited.")
    Settings = None
    load_settings = None
# --- End Corrected Import ---


try:
    from flashrank import Ranker # type: ignore
    # This block for Flashrank_rerank import fix might no longer be needed with newer FlashRank versions,
    # but keeping it for robustness if you're using an older version or specific LangChain integration.
    # If Flashrank version is 0.1.0+, you might not need this.
    try:
        import langchain_community.document_compressors.flashrank_rerank as fr_module
        from flashrank import RerankRequest # type: ignore
        fr_module.RerankRequest = RerankRequest # Patch the RerankRequest if used by older langchain_community
    except ImportError:
        logger.debug("langchain_community.document_compressors.flashrank_rerank module not found, or Flashrank version is recent.")
    
    logger.info("Flashrank library found and imported.")

except ImportError:
    Ranker = None
    logger.warning("Flashrank not installed. Reranking functionality will be disabled. Install with `pip install flashrank`.")

# --- Logging Setup ---
logger = logging.getLogger(__name__) # Define logger at the top after initial imports

# --- Component Registry ---
RERANKER_REGISTRY: Dict[str, Type['BaseReranker']] = {} # Using specific type hint now

def register_reranker_strategy(name: str):
    """
    Decorator to register reranker strategy classes dynamically.
    """
    def decorator(cls: Type['BaseReranker']):
        if name in RERANKER_REGISTRY:
            logger.warning(f"Reranker strategy '{name}' already registered. Overwriting.")
        RERANKER_REGISTRY[name] = cls
        return cls
    return decorator

# --- Base Reranker Class ---
class BaseReranker:
    """
    Abstract Base Class for all reranking strategies.
    Defines the interface that all concrete reranker strategies must implement.
    """
    def __init__(self, model_name: str, top_n: int = 5, **kwargs):
        self.model_name = model_name
        self.top_n = top_n
        self.config = kwargs
        logger.info(f"Initializing {self.__class__.__name__} with model: '{model_name}', top_n={top_n}, config={kwargs}")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Abstract method to rerank a list of documents based on the query.
        Args:
            query (str): The user's query.
            documents (List[Dict[str, Any]]): A list of documents, where each document is a dictionary
                                               expected to have at least a 'text' key.
        Returns:
            List[Dict[str, Any]]: A re-ordered list of reranked documents,
                                  each augmented with a 'relevance_score'.
        """
        raise NotImplementedError

# --- Concrete Reranker Strategies ---

@register_reranker_strategy("flashrank")
class FlashrankReranker(BaseReranker):
    """
    Reranker strategy using Flashrank for efficient cross-encoder reranking.
    """
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2", top_n: int = 5, **kwargs):
        super().__init__(model_name, top_n, **kwargs)
        if Ranker is None:
            raise ImportError(
                "Flashrank library not available. "
                "Please install `flashrank` to use this reranking strategy."
            )
        self.ranker = Ranker(model=self.model_name) # Initialize Flashrank's Ranker
        logger.info(f"Flashrank Reranker initialized with model '{self.model_name}'.")

    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not documents:
            return []

        # Prepare documents for Flashrank: list of dicts with a 'text' key
        # Flashrank can also take 'meta' but we don't strictly need it for reranking score calculation
        flashrank_passages = [{"text": doc["text"]} for doc in documents]
        
        rerank_results = self.ranker.rank(query=query, passages=flashrank_passages, top_n=self.top_n)

        # Flashrank returns an ordered list of dictionaries, each with 'text' and 'score'.
        # We need to re-associate these scores with the *original* documents and their full metadata.
        # This requires matching based on text or a unique ID. Since texts can be long and truncated by Flashrank,
        # it's better to rely on order if rerank_results preserves the relative order of matched passages.

        # The simplest way is to assume Flashrank returns the passages in the same order they were passed,
        # just with scores and potentially truncated. More robust is to use original IDs.
        # For our use case, `documents` will be an iterable of CHROMA_RESULTS where original_index can be used.
        
        # Let's rebuild the original documents, adding the 'relevance_score'
        # We assume `rerank_results` is effectively a re-ordered subset of `documents`
        # and has a 'score' key.
        
        # Create a mapping from text content (or ideally ID) to original full document
        # For simplicity, if unique text is available, we use that. Else, re-build.
        
        reranked_docs_with_scores = []
        # Flashrank returns the passages re-ordered with scores.
        # So we iterate through Flashrank's results and find the matching original document.
        # This is inefficient but works if texts are unique. A better way involves unique IDs.
        
        # A more practical approach:
        # 1. Pass a more robust structure to `documents`, e.g., [{"id": ..., "text": ..., "original_doc_data": original_chunk_dict}]
        # 2. Reconstruct by matching `text` field (if unique enough).
        # We'll augment the input `documents` with scores.
        
        # Flashrank's rank method keeps the 'text' key but reorders and adds 'score'.
        # We need to map these scores back to the original full documents.
        
        # Create a temporary list of dicts to preserve original indices
        passages_for_rerank = []
        for i, doc in enumerate(documents):
            passages_for_rerank.append({"id": i, "text": doc["text"]}) # Use index as a temporary ID

        reranked_with_scores = self.ranker.rank(query=query, passages=passages_for_rerank, top_n=self.top_n)
        
        final_reranked_documents = []
        for ranked_item in reranked_with_scores:
            original_index = ranked_item["id"] # Use our temporary ID
            original_doc = documents[original_index]
            original_doc["relevance_score"] = ranked_item["score"] # Add the score
            final_reranked_documents.append(original_doc)

        return final_reranked_documents
