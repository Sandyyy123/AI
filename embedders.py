# src/embedders.py
from typing import Protocol, List
import numpy as np

class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        ...

# ---- SentenceTransformers ----
class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self.model.encode(texts, normalize_embeddings=False)
        return np.asarray(vecs)

# ---- Gemini ----
class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        # Gemini returns one embedding per content item
        out = []
        for t in texts:
            res = self.client.models.embed_content(
                model=self.model,
                contents=t,
            )
            out.append(res.embeddings[0].values)
        return np.asarray(out)
