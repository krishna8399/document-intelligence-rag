"""
BM25 keyword search — complements semantic search.

Why BM25 alongside vector search:
- Vector search finds semantically similar text ("fever" matches "high temperature")
- BM25 finds exact keyword matches ("HIPAA" matches "HIPAA" — vector search might miss this)
- Hybrid = best of both worlds

When BM25 wins over semantic search:
- Acronyms and technical terms (API, GDPR, HIPAA)
- Proper nouns (company names, product names)
- Code snippets and variable names
- Numbers and dates
"""

from typing import List

from rank_bm25 import BM25Okapi


class BM25Search:
    """BM25 keyword search over document chunks."""

    def __init__(self):
        self.corpus = []        # list of chunk texts
        self.metadata = []      # parallel list of metadata
        self.bm25 = None

    def index(self, chunks: List[dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: list of dicts with 'text', 'source', 'chunk_id'
        """
        self.corpus = [c["text"] for c in chunks]
        self.metadata = chunks

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search using BM25 keyword matching.

        Returns:
            List of dicts with 'text', 'source', 'score'
        """
        if self.bm25 is None or len(self.corpus) == 0:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include matches with non-zero score
                results.append({
                    "text": self.corpus[idx],
                    "metadata": self.metadata[idx],
                    "score": float(scores[idx]),
                    "source": self.metadata[idx].get("source", "unknown"),
                })

        return results
