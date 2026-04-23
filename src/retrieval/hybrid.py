"""
Hybrid search — combines semantic (vector) search with BM25 keyword search.

Default: 70% semantic + 30% BM25

Why this ratio:
- Most questions are semantic ("What are the side effects?" → matches
  "adverse reactions include...")
- But some need exact matching ("What does Section 4.2 say?" → needs
  keyword match on "Section 4.2")
- 70/30 handles both cases well
"""

from typing import List, Optional

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.bm25 import BM25Search


class HybridRetriever:
    """Combines semantic and BM25 search with weighted scoring."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        self.vector_store = vector_store
        self.bm25 = BM25Search()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self._bm25_indexed = False

    def build_bm25_index(self):
        """Build BM25 index from all chunks in vector store."""
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"],
        )

        # Always mark as indexed so search() doesn't retry on empty store
        self._bm25_indexed = True

        if not all_data["ids"]:
            return

        chunks = [
            {
                "text": doc,
                **meta,
            }
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]

        self.bm25.index(chunks)
        print(f"  BM25 index built: {len(chunks)} chunks")

    def reset_bm25(self):
        """Clear the BM25 index (call after vector store is cleared)."""
        self.bm25 = BM25Search()
        self._bm25_indexed = False

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Hybrid search: semantic + BM25.

        Returns:
            List of dicts sorted by hybrid score, with source citations
        """
        if not self._bm25_indexed:
            self.build_bm25_index()

        semantic_results = self.vector_store.search(
            query, top_k=top_k * 2, source_filter=source_filter,
        )
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        merged = self._merge_results(semantic_results, bm25_results)
        return merged[:top_k]

    def _merge_results(
        self,
        semantic_results: List[dict],
        bm25_results: List[dict],
    ) -> List[dict]:
        """
        Merge results from both search methods using reciprocal rank fusion.

        Why RRF instead of raw score combination:
        - Semantic scores are cosine similarity [0, 1]
        - BM25 scores are unbounded [0, infinity)
        - They're not on the same scale, so raw combination is biased
        - RRF uses rank position instead of raw scores, making them comparable
        """
        combined = {}

        for rank, result in enumerate(semantic_results):
            key = result.get("id", result["text"][:100])
            combined[key] = {
                "text": result["text"],
                "source": result["source"],
                "metadata": result.get("metadata", {}),
                "semantic_score": result.get("score", 0),
                "bm25_score": 0,
                "semantic_rank": rank + 1,
                "bm25_rank": 999,  # sentinel: not found in BM25 results
            }

        for rank, result in enumerate(bm25_results):
            key = result.get("id", result["text"][:100])
            if key not in combined:
                combined[key] = {
                    "text": result["text"],
                    "source": result["source"],
                    "metadata": result.get("metadata", {}),
                    "semantic_score": 0,
                    "bm25_score": result.get("score", 0),
                    "semantic_rank": 999,
                    "bm25_rank": rank + 1,
                }
            else:
                combined[key]["bm25_rank"] = rank + 1
                combined[key]["bm25_score"] = result.get("score", 0)

        k = 60  # RRF constant (standard value)
        for result in combined.values():
            result["hybrid_score"] = (
                self.semantic_weight / (k + result["semantic_rank"]) +
                self.bm25_weight / (k + result["bm25_rank"])
            )

        return sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
