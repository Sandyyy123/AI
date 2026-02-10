# scripts/rag/components/vector_stores.py (from our previous discussion - ensure this is in place)

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Union, Type

# Conditional imports for Chroma and FAISS
try:
    import chromadb
except ImportError:
    chromadb = None
    logging.warning("ChromaDB not installed. ChromaVectorStore will be unavailable.")

try:
    # We will need numpy for FAISS index creation from List[List[float]]
    import numpy as np
    from faiss import IndexFlatL2, write_index, read_index # type: ignore
except ImportError:
    np = None
    IndexFlatL2 = None
    write_index = None
    read_index = None
    logging.warning("FAISS not installed. FAISSVectorStore will be unavailable. Install with `pip install faiss-cpu` or `faiss-gpu`.")


logger = logging.getLogger(__name__)

# --- Base Vector Store Protocol ---
class BaseVectorStore(Protocol):
    """
    Protocol for a generic vector store.
    Defines the interface that all concrete vector store strategies must implement.
    """
    config: Dict[str, Any] # Add this to the protocol if all implementations take it

    def __init__(self, config: Dict[str, Any]):
        ... # Implementations should call super().__init__(config)

    def add(self, ids: List[str], embeddings: Optional[List[List[float]]], documents: List[str], metadatas: List[Dict[str, Any]]):
        """Adds documents, embeddings, and metadata to the vector store."""
        ...

    def query(self, query_embeddings: List[List[float]], n_results: int, include: List[str]) -> Dict[str, Any]:
        """
        Queries the vector store for nearest neighbors.
        Returns a dict in a consistent format (e.g., {'ids': [[...]], 'distances': [[...]], 'documents': [[...]], 'metadatas': [[...]]}).
        """
        ...
    
    def get(self, ids: List[str], include: List[str]) -> Dict[str, Any]:
        """Retrieves documents and metadata by ID."""
        ...

    def count(self) -> int:
        """Returns the number of embeddings in the store."""
        ...
    
    def persist(self):
        """Saves the vector store data to disk if applicable."""
        pass # Optional method for non-persistent stores


# --- Component Registry ---
# Using Type[Any] because Type[BaseVectorStore] causes issues in python <3.9 with Protocol
VECTOR_STORE_REGISTRY: Dict[str, Any] = {} 

def register_vector_store(name: str):
    """Decorator to register vector store implementations dynamically."""
    def decorator(cls: Type[BaseVectorStore]):
        if name in VECTOR_STORE_REGISTRY:
            logger.warning(f"Vector store '{name}' already registered. Overwriting.")
        VECTOR_STORE_REGISTRY[name] = cls
        return cls
    return decorator


# --- ChromaDB Implementation ---
@register_vector_store("chroma")
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        self.config = config # Set config attribute
        if chromadb is None:
            raise ImportError("ChromaDB library not found. Please install `chromadb`.")
        
        self.mode = config.get("mode", "local")
        self.collection_name = config.get("collection_name")
        self.persist_directory = config.get("persist_directory")
        self.host = config.get("host")
        self.port = config.get("port")
        self.collection_metadata = config.get("collection_metadata", {})
        self.overwrite = config.get("overwrite", False)

        if not self.collection_name:
            raise ValueError("ChromaVectorStore requires a 'collection_name'.")

        self.client: chromadb.ClientAPI = self._init_client()
        self.collection: chromadb.Collection = self._get_or_create_collection()
        logger.info(f"ChromaVectorStore '{self.collection_name}' initialized, count: {self.count()} (mode: {self.mode}).")

    def _init_client(self) -> chromadb.ClientAPI:
        if self.mode == "local":
            if not self.persist_directory:
                raise ValueError("Chroma mode 'local' requires 'persist_directory'.")
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            return chromadb.PersistentClient(path=str(self.persist_directory))
        elif self.mode == "http":
            if not self.host or not self.port:
                raise ValueError("Chroma mode 'http' requires 'host' and 'port'.")
            return chromadb.HttpClient(host=self.host, port=self.port)
        else:
            raise ValueError(f"Unknown Chroma mode: {self.mode}. Supported: 'local', 'http'.")

    def _get_or_create_collection(self) -> chromadb.Collection:
        if self.overwrite:
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.warning(f"Existing Chroma collection '{self.collection_name}' deleted as overwrite=True.")
            except Exception:
                pass # Collection might not exist
        
        return self.client.get_or_create_collection(name=self.collection_name, metadata=self.collection_metadata)

    def add(self, ids: List[str], embeddings: Optional[List[List[float]]], documents: List[str], metadatas: List[Dict[str, Any]]):
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("Lengths of ids, documents, and metadatas must match.")
        if embeddings is not None and len(ids) != len(embeddings):
             raise ValueError("Lengths of ids and embeddings must match if embeddings are provided.")
        
        # Chroma requires embeddings list to be provided for add, even if None elements.
        # If we have documents but no embeddings (e.g., for parent store), create dummy embeddings or rely on Chroma behavior.
        # Chroma's `add` method can handle `None` for embeddings if a default embedding function is set (not here).
        # We explicitly pass embeddings if given, else pass an empty array, which add() handles IF it's a metadata-only collection.
        # For our child collection, embeddings are GUARANTEED to be passed.
        # For our parent collection, embeddings are explicitly None to signify storage only.
        
        # If no embeddings provided for a collection that requires them (e.g., for vector search on child_chunks_store), this will fail.
        # Our BaseVectorStore.add method requires embeddings as Optional. Let's send it to Chroma as is.
        self.collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        logger.debug(f"Added {len(ids)} items to Chroma collection '{self.collection_name}'.")


    def query(self, query_embeddings: List[List[float]], n_results: int, include: List[str]) -> Dict[str, Any]:
        # Chroma's query method returns distance, ID, document text, and metadata directly
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include # e.g., ['documents', 'metadatas', 'distances']
        )
    
    def get(self, ids: List[str], include: List[str]) -> Dict[str, Any]:
        # Chroma's get method allows retrieving by IDs
        return self.collection.get(ids=ids, include=include)

    def count(self) -> int:
        return self.collection.count()
    
    def persist(self):
        # Only persistent clients support this method
        if self.mode == "local" and hasattr(self.client, 'persist'):
            self.client.persist()
            logger.info(f"Chroma collection '{self.collection_name}' persisted.")


# --- FAISS Implementation ---
@register_vector_store("faiss")
class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: Dict[str, Any]):
        self.config = config # Set config attribute
        if IndexFlatL2 is None or np is None:
            raise ImportError("FAISS library or NumPy not found. Please install `faiss-cpu` (or `faiss-gpu`) and `numpy`.")
        
        self.persist_directory = config.get("persist_directory")
        self.dimension = config.get("dimension")
        self.collection_name = config.get("collection_name")
        self.overwrite = config.get("overwrite", False)
        # For FAISS, a "collection" in our abstract sense will be a file pair (.faiss and _metadata.jsonl)
        # OR it can store only metadata (no .faiss file)
        self.is_vector_indexed = config.get("is_vector_indexed", True) # Default to true, false for parent docs
        
        if not self.persist_directory: raise ValueError("FAISSVectorStore requires a 'persist_directory'.")
        if self.is_vector_indexed and not self.dimension:
            raise ValueError("FAISSVectorStore with vector indexing requires 'dimension' of embeddings.")
        if not self.collection_name: raise ValueError("FAISSVectorStore requires a 'collection_name'.")

        self.persist_directory = Path(self.persist_directory)
        self.faiss_index_path = self.persist_directory / f"{self.collection_name}.faiss"
        self.metadata_path = self.persist_directory / f"{self.collection_name}_metadata.jsonl"
        
        self.index: Optional[Any] = None # Will be IndexFlatL2 for child embeddings, None for parent.
        self.id_to_internal_idx: Dict[str, int] = {} # Maps external ID (str) to internal FAISS index (int)
        self.internal_idx_to_id: List[str] = []      # Maps internal FAISS index (int) back to external ID (str)
        self.all_data: Dict[str, Dict[str, Any]] = {} # Stores full item data (text, meta, original_faiss_idx) by external ID

        self._load_or_init()
        logger.info(f"FAISSVectorStore '{self.collection_name}' initialized (indexed: {self.is_vector_indexed}, dimension: {self.dimension if self.is_vector_indexed else 'N/A'}).")

    def _load_or_init(self):
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        if self.overwrite:
            if self.faiss_index_path.exists(): self.faiss_index_path.unlink()
            if self.metadata_path.exists(): self.metadata_path.unlink()
            logger.warning(f"Overwriting FAISS index/metadata for '{self.collection_name}'.")
        
        # Load metadata first
        if self.metadata_path.exists():
            logger.info(f"Loading existing metadata for FAISS store '{self.collection_name}' from {self.metadata_path}")
            current_idx = 0
            with self.metadata_path.open("r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    orig_id = entry["id"]
                    self.all_data[orig_id] = entry
                    if self.is_vector_indexed: # Only build internal mappings if expects vectors
                        self.id_to_internal_idx[orig_id] = entry.get("internal_faiss_idx", current_idx) # Use stored internal_idx if present
                        self.internal_idx_to_id.insert(entry.get("internal_faiss_idx", current_idx), orig_id) # Ensure correct position
                        current_idx += 1
            logger.info(f"Loaded {len(self.all_data)} items into FAISS metadata store for '{self.collection_name}'.")

        if self.is_vector_indexed and self.faiss_index_path.exists():
            logger.info(f"Loading existing FAISS index for '{self.collection_name}' from {self.faiss_index_path}")
            self.index = read_index(str(self.faiss_index_path))
            
            # Additional check:
            if self.index.ntotal != len(self.internal_idx_to_id): # OR len(self.all_data) if internal_idx_to_id isn't fully rebuilt
                logger.warning(f"FAISS index count ({self.index.ntotal}) differs from metadata ID mapping count ({len(self.internal_idx_to_id)}). "
                               "This may indicate a corrupted index or out-of-sync files. Re-indexing might be needed.")
        elif self.is_vector_indexed:
            logger.info(f"Initializing new empty FAISS index for '{self.collection_name}'.")
            self.index = IndexFlatL2(self.dimension)
        # If not self.is_vector_indexed, self.index remains None.

    def add(self, ids: List[str], embeddings: Optional[List[List[float]]], documents: List[str], metadatas: List[Dict[str, Any]]):
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError("Lengths of ids, documents, and metadatas must match.")
        if embeddings is not None and len(ids) != len(embeddings):
             raise ValueError("Lengths of ids and embeddings must match if embeddings are provided.")

        entries_to_add_to_faiss: List[List[float]] = []
        entries_to_add_to_metadata: List[Dict[str, Any]] = []

        if self.is_vector_indexed:
            if embeddings is None:
                raise ValueError("FAISSVectorStore (vector indexed) requires 'embeddings' for add operation.")
            embeddings_np = np.array(embeddings).astype('float32') # FAISS expects float32
            if embeddings_np.shape[1] != self.dimension:
                raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings_np.shape[1]}.")
            entries_to_add_to_faiss = embeddings_np
            
        
        for i, item_id in enumerate(ids):
            # For each item, prepare the full data entry for metadata store
            entry_to_store = {"id": item_id, "text": documents[i], "metadata": metadatas[i]}
            entries_to_add_to_metadata.append(entry_to_store)
            
            # Update internal mappings if vector indexed
            if self.is_vector_indexed:
                if item_id in self.id_to_internal_idx:
                    logger.warning(f"Duplicate ID '{item_id}' detected. FAISS index does not support in-place update by ID. "
                                   "Adding as new entry. Metadata will reflect the latest addition for this ID.")
                    # In FAISS, if you `add` an existing ID, it adds a new vector. We map `id_to_internal_idx` to the _latest_ one.
                    # For robust updates/deletes in FAISS, a "wrapper" index like IndexIDMap would be needed.
                    # For simplicity, we assume IDs are unique for the metadata JSONL.
                
                # Assign internal FAISS index and update mappings
                if self.index:
                    internal_idx = self.index.ntotal + len(entries_to_add_to_faiss) - len(ids) + i # Current ntotal + how many we are about to add (minus those already added in this batch) + current item in batch
                    # This logic for internal_idx is tricky. Simpler: add vectors to FAISS, then iterate and assign _newly_ added indices.
                    # Or, just append to mappings assuming `add` works sequentially.
                    entry_to_store["internal_faiss_idx"] = self.index.ntotal + i if self.index else 0 # Temp mapping, will be correct after add
                    # A more robust solution for internal_faiss_idx tracking is outside this simple FlatL2 index example.
                    # For now, let's just ensure it's recorded for future retrieval.
                    
                self.id_to_internal_idx[item_id] = len(self.internal_idx_to_id) # Assign next available internal index
                self.internal_idx_to_id.append(item_id) # Map that internal index back to the ID
        
        # Add vectors to FAISS index if this store is vector indexed
        if self.is_vector_indexed and self.index and len(entries_to_add_to_faiss) > 0:
            original_ntotal = self.index.ntotal # Store current count before adding
            self.index.add(entries_to_add_to_faiss)
            # Update internal_faiss_idx in entries_to_add_to_metadata for newly added items
            for i, entry in enumerate(entries_to_add_to_metadata):
                entry["internal_faiss_idx"] = original_ntotal + i

        # Append to metadata store
        with self.metadata_path.open("a", encoding="utf-8") as f:
            for entry in entries_to_add_to_metadata:
                self.all_data[entry["id"]] = entry # Update all_data with newly constructed dict
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        logger.debug(f"Added {len(ids)} items to FAISS store '{self.collection_name}'.")


    def query(self, query_embeddings: List[List[float]], n_results: int, include: List[str]) -> Dict[str, Any]:
        if self.index is None:
            raise RuntimeError("FAISS index not initialized for queries.")
        if not query_embeddings:
            return {'ids': [], 'distances': [], 'documents': [], 'metadatas': []}

        query_embeddings_np = np.array(query_embeddings).astype('float32')
        if query_embeddings_np.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.dimension}, got {query_embeddings_np.shape[1]}.")

        D, I = self.index.search(query_embeddings_np, n_results)

        all_query_ids: List[List[str]] = []
        all_query_distances: List[List[float]] = []
        all_query_documents: List[List[str]] = []
        all_query_metadatas: List[List[Dict[str, Any]]] = []

        for i_query in range(query_embeddings_np.shape[0]):
            query_ids: List[str] = []
            query_distances: List[float] = []
            query_documents: List[str] = []
            query_metadatas: List[Dict[str, Any]] = []

            for i_res in range(n_results):
                faiss_internal_idx = I[i_query, i_res]
                distance = float(D[i_query, i_res])

                if faiss_internal_idx == -1 or faiss_internal_idx >= len(self.internal_idx_to_id):
                    continue
                
                orig_id = self.internal_idx_to_id[faiss_internal_idx] # Map internal index to original ID
                full_item = self.all_data.get(orig_id)
                if full_item:
                    query_ids.append(orig_id)
                    query_distances.append(distance)
                    if 'documents' in include: query_documents.append(full_item.get('text', ''))
                    if 'metadatas' in include: query_metadatas.append(full_item.get('metadata', {}))
                else: logger.warning(f"FAISS index returned internal index {faiss_internal_idx} for ID {orig_id} but it's not in metadata store.")

            all_query_ids.append(query_ids)
            all_query_distances.append(query_distances)
            all_query_documents.append(query_documents)
            all_query_metadatas.append(query_metadatas)
        
        return {
            'ids': all_query_ids, 'distances': all_query_distances,
            'documents': all_query_documents, 'metadatas': all_query_metadatas
        }
    
    def get(self, ids: List[str], include: List[str]) -> Dict[str, Any]:
        retrieved_ids: List[str] = []
        retrieved_documents: List[str] = []
        retrieved_metadatas: List[Dict[str, Any]] = []

        for item_id in ids:
            full_item = self.all_data.get(item_id)
            if full_item:
                retrieved_ids.append(item_id)
                if 'documents' in include: retrieved_documents.append(full_item.get('text', ''))
                if 'metadatas' in include: retrieved_metadatas.append(full_item.get('metadata', {}))
            else: logger.warning(f"ID '{item_id}' requested but not found in FAISS metadata store '{self.collection_name}'.")
        
        return {'ids': retrieved_ids, 'documents': retrieved_documents, 'metadatas': retrieved_metadatas}


    def count(self) -> int:
        return self.index.ntotal if self.index else 0
    
    def persist(self):
        if self.is_vector_indexed and self.index is not None:
             if self.persist_directory:
                 self.persist_directory.mkdir(parents=True, exist_ok=True)
                 write_index(self.index, str(self.faiss_index_path))
                 logger.info(f"FAISS index '{self.collection_name}' saved to {self.faiss_index_path}")
             else:
                 logger.warning(f"Cannot persist FAISS index for '{self.collection_name}': persist_directory not set.")
        # Metadata is appended on add, no specific 'persist' needed for it.
