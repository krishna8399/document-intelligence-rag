"""Tests for BM25 and hybrid retrieval."""

import pytest
from src.retrieval.bm25 import BM25Search


SAMPLE_CHUNKS = [
    {"text": "The patient was diagnosed with diabetes mellitus type 2.", "source": "medical.txt", "chunk_id": 0},
    {"text": "HIPAA regulations require strict data privacy for health records.", "source": "legal.txt", "chunk_id": 0},
    {"text": "Machine learning models can predict disease outcomes accurately.", "source": "ai.txt", "chunk_id": 0},
    {"text": "The quarterly revenue exceeded expectations by 15 percent.", "source": "finance.txt", "chunk_id": 0},
    {"text": "Deep learning architectures use multiple hidden layers for feature extraction.", "source": "ai.txt", "chunk_id": 1},
]


class TestBM25Search:
    def setup_method(self):
        self.bm25 = BM25Search()
        self.bm25.index(SAMPLE_CHUNKS)

    def test_returns_results_for_matching_query(self):
        results = self.bm25.search("HIPAA privacy regulations", top_k=3)
        assert len(results) > 0

    def test_top_result_is_most_relevant(self):
        results = self.bm25.search("HIPAA privacy regulations", top_k=5)
        assert results[0]["source"] == "legal.txt"

    def test_no_results_for_unrelated_query(self):
        results = self.bm25.search("xyzzy completely unrelated nonsense", top_k=3)
        assert len(results) == 0

    def test_top_k_limits_results(self):
        results = self.bm25.search("machine learning deep learning", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self):
        results = self.bm25.search("diabetes patient diagnosis", top_k=3)
        assert all(r["score"] > 0 for r in results)

    def test_exact_keyword_match_scores_high(self):
        results = self.bm25.search("HIPAA", top_k=5)
        top = results[0]
        assert "HIPAA" in top["text"]

    def test_empty_index_returns_no_results(self):
        empty_bm25 = BM25Search()
        results = empty_bm25.search("anything", top_k=3)
        assert results == []

    def test_result_contains_required_keys(self):
        results = self.bm25.search("machine learning", top_k=1)
        assert len(results) > 0
        for key in ("text", "source", "score", "metadata"):
            assert key in results[0]
