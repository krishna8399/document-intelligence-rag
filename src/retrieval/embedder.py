"""
Embedding engine — converts text to vectors for similarity search.

Why sentence-transformers (all-MiniLM-L6-v2):
- Runs locally (no API key, no cost, no data leaving your machine)
- 384-dimensional vectors (compact, fast similarity computation)
- Trained on 1B+ sentence pairs for semantic similarity
- Good enough for most RAG use cases
- 80MB model size (vs 1.3GB for larger models)

How embeddings work:
- Input: "The patient has a fever" (text)
- Output: [0.023, -0.156, 0.891, ...] (384 numbers)
- Similar texts have similar vectors (high cosine similarity)
- "The patient has a fever" and "Person is sick with high temperature"
  will have cosine similarity > 0.8, even though they share few words
"""

from typing import List

from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class EmbeddingEngine:
    """Wraps sentence-transformers for text embedding."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()
        print(f"  Embedding model: {model_name} ({self.dimension}-dim)")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def get_embedding_function(self):
        """Return ChromaDB-compatible embedding function."""
        return SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
        )
