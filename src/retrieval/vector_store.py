"""
ChromaDB vector store — persistent storage for document embeddings.

Why ChromaDB:
- Runs locally (no cloud dependency, no cost)
- Persistent storage (survives restarts)
- Built-in embedding support
- Simple API, great for prototypes and small-medium datasets
- Metadata filtering (search within specific documents)

For production at scale, you'd switch to Pinecone, Weaviate, or Qdrant.
But ChromaDB is perfect for a portfolio project.
"""

from typing import List, Optional

import chromadb

from src.ingestion.chunker import TextChunk


class VectorStoreManager:
    """Manages ChromaDB collection for document chunks."""

    def __init__(
        self,
        persist_dir: str = "data/chroma_db",
        collection_name: str = "documents",
        embedding_function=None,
    ):
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  Vector store: {persist_dir} ({self.collection.count()} existing chunks)")

    def add_chunks(self, chunks: List[TextChunk]):
        """Add text chunks to the vector store."""
        if not chunks:
            return

        self.collection.add(
            ids=[f"{c.source}__chunk_{c.chunk_id}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "source": c.source,
                    "chunk_id": c.chunk_id,
                    "char_count": len(c.text),
                    "total_chunks": c.metadata.get("total_chunks", 0),
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for relevant chunks.

        Returns:
            List of dicts with 'text', 'source', 'score', 'metadata'
        """
        where_filter = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        return [
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,  # cosine distance → similarity
                "source": meta.get("source", "unknown"),
            }
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
        }

    def delete_source(self, source: str):
        """Delete all chunks from a specific source document."""
        self.collection.delete(where={"source": source})

    def clear(self):
        """Delete all chunks."""
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
