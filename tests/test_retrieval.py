"""
Comprehensive tests for BM25, VectorStoreManager, and HybridRetriever.

VectorStoreManager tests use a mocked ChromaDB client — no real DB or
embedding model is needed. HybridRetriever merge tests call _merge_results
directly with hand-crafted fixtures to verify RRF arithmetic and deduplication.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from src.retrieval.bm25 import BM25Search
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid import HybridRetriever
from src.ingestion.chunker import TextChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    {"text": "The patient was diagnosed with diabetes mellitus type 2.", "source": "medical.txt", "chunk_id": 0},
    {"text": "HIPAA regulations require strict data privacy for health records.", "source": "legal.txt", "chunk_id": 0},
    {"text": "Machine learning models can predict disease outcomes accurately.", "source": "ai.txt", "chunk_id": 0},
    {"text": "The quarterly revenue exceeded expectations by 15 percent.", "source": "finance.txt", "chunk_id": 0},
    {"text": "Deep learning architectures use multiple hidden layers for feature extraction.", "source": "ai.txt", "chunk_id": 1},
]


def make_chunk(text: str, source: str = "doc.txt", chunk_id: int = 0) -> TextChunk:
    return TextChunk(
        text=text,
        chunk_id=chunk_id,
        source=source,
        metadata={"total_chunks": 1},
    )


@pytest.fixture
def vs():
    """VectorStoreManager backed by a mocked ChromaDB client."""
    mock_collection = MagicMock()
    mock_collection.name = "test_docs"
    mock_collection.count.return_value = 0

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection

    with patch("chromadb.PersistentClient") as MockClient:
        MockClient.return_value = mock_client
        store = VectorStoreManager(
            persist_dir="/tmp/test_chroma",
            collection_name="test_docs",
            embedding_function="fake_ef",
        )
        yield store, mock_collection, mock_client


@pytest.fixture
def retriever():
    """HybridRetriever with a mocked vector store."""
    mock_vs = MagicMock()
    mock_vs.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    mock_vs.search.return_value = []
    return HybridRetriever(vector_store=mock_vs, semantic_weight=0.7, bm25_weight=0.3)


def make_merger(semantic_weight: float = 0.7, bm25_weight: float = 0.3) -> HybridRetriever:
    """Lightweight HybridRetriever for testing _merge_results only."""
    r = HybridRetriever.__new__(HybridRetriever)
    r.semantic_weight = semantic_weight
    r.bm25_weight = bm25_weight
    return r


# ---------------------------------------------------------------------------
# BM25Search
# ---------------------------------------------------------------------------

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

    def test_no_results_for_completely_unrelated_query(self):
        results = self.bm25.search("xyzzy completely unrelated nonsense foobar", top_k=3)
        assert len(results) == 0

    def test_top_k_caps_number_of_results(self):
        results = self.bm25.search("machine learning deep learning", top_k=2)
        assert len(results) <= 2

    def test_scores_are_positive(self):
        results = self.bm25.search("diabetes patient diagnosis", top_k=3)
        assert all(r["score"] > 0 for r in results)

    def test_scores_are_in_descending_order(self):
        results = self.bm25.search("machine learning", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_keyword_match_scores_highest(self):
        results = self.bm25.search("HIPAA", top_k=5)
        assert "HIPAA" in results[0]["text"]

    def test_empty_index_returns_empty_list(self):
        empty = BM25Search()
        assert empty.search("anything", top_k=3) == []

    def test_result_contains_required_keys(self):
        results = self.bm25.search("machine learning", top_k=1)
        assert len(results) > 0
        for key in ("text", "source", "score", "metadata"):
            assert key in results[0], f"Missing key: {key}"

    def test_reindex_replaces_previous_corpus(self):
        # Need ≥ 3 docs so query terms appear in < half the corpus,
        # giving positive BM25 IDF scores (rank_bm25's Okapi formula
        # returns negative IDF when df == N)
        new_chunks = [
            {"text": "Quantum computing is revolutionary.", "source": "quantum.txt", "chunk_id": 0},
            {"text": "Classical computing uses transistors.", "source": "classical.txt", "chunk_id": 0},
            {"text": "Optical computing uses photons and light.", "source": "optical.txt", "chunk_id": 0},
        ]
        self.bm25.index(new_chunks)
        # Old HIPAA chunk should no longer be indexed
        assert all("HIPAA" not in r["text"] for r in self.bm25.search("HIPAA", top_k=3))
        # New corpus should surface the quantum doc
        quantum_results = self.bm25.search("quantum", top_k=3)
        assert len(quantum_results) >= 1
        assert any("quantum.txt" in r["source"] for r in quantum_results)


# ---------------------------------------------------------------------------
# VectorStoreManager
# ---------------------------------------------------------------------------

class TestVectorStoreManager:
    def test_add_chunks_builds_correct_ids_and_documents(self, vs):
        store, mock_col, _ = vs
        chunks = [
            make_chunk("Hello world", source="doc.txt", chunk_id=0),
            make_chunk("Second chunk", source="doc.txt", chunk_id=1),
        ]
        store.add_chunks(chunks)
        args = mock_col.add.call_args.kwargs
        assert args["ids"] == ["doc.txt__chunk_0", "doc.txt__chunk_1"]
        assert args["documents"] == ["Hello world", "Second chunk"]

    def test_add_chunks_sets_correct_metadata(self, vs):
        store, mock_col, _ = vs
        chunk = make_chunk("Test text", source="report.pdf", chunk_id=3)
        store.add_chunks([chunk])
        meta = mock_col.add.call_args.kwargs["metadatas"][0]
        assert meta["source"] == "report.pdf"
        assert meta["chunk_id"] == 3
        assert meta["char_count"] == len("Test text")

    def test_add_empty_chunks_list_does_not_call_collection(self, vs):
        store, mock_col, _ = vs
        store.add_chunks([])
        mock_col.add.assert_not_called()

    def test_search_converts_distance_to_similarity_score(self, vs):
        store, mock_col, _ = vs
        mock_col.query.return_value = {
            "ids": [["doc.txt__chunk_0"]],
            "documents": [["Some content."]],
            "metadatas": [[{"source": "doc.txt", "chunk_id": 0}]],
            "distances": [[0.2]],   # similarity = 1 - 0.2 = 0.8
        }
        results = store.search("test query")
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8)

    def test_search_returns_correct_fields(self, vs):
        store, mock_col, _ = vs
        mock_col.query.return_value = {
            "ids": [["doc.txt__chunk_0"]],
            "documents": [["Chunk text here."]],
            "metadatas": [[{"source": "doc.txt", "chunk_id": 0}]],
            "distances": [[0.1]],
        }
        result = store.search("query")[0]
        assert result["id"] == "doc.txt__chunk_0"
        assert result["text"] == "Chunk text here."
        assert result["source"] == "doc.txt"
        assert "score" in result
        assert "metadata" in result

    def test_search_with_source_filter_passes_where_clause(self, vs):
        store, mock_col, _ = vs
        mock_col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        store.search("query", source_filter="specific.txt")
        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs["where"] == {"source": "specific.txt"}

    def test_search_without_filter_omits_where_clause(self, vs):
        store, mock_col, _ = vs
        mock_col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        store.search("query")
        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs["where"] is None

    def test_delete_source_passes_correct_where_filter(self, vs):
        store, mock_col, _ = vs
        store.delete_source("old_report.pdf")
        mock_col.delete.assert_called_once_with(where={"source": "old_report.pdf"})

    def test_clear_deletes_and_recreates_collection_with_embedding_function(self, vs):
        store, mock_col, mock_client = vs
        store.clear()
        mock_client.delete_collection.assert_called_once_with("test_docs")
        mock_client.create_collection.assert_called_once_with(
            name="test_docs",
            embedding_function="fake_ef",
            metadata={"hnsw:space": "cosine"},
        )

    def test_get_stats_returns_count_and_collection_name(self, vs):
        store, mock_col, _ = vs
        mock_col.count.return_value = 42
        mock_col.name = "test_docs"
        stats = store.get_stats()
        assert stats["total_chunks"] == 42
        assert stats["collection_name"] == "test_docs"


# ---------------------------------------------------------------------------
# HybridRetriever — _merge_results (pure logic, no I/O)
# ---------------------------------------------------------------------------

# Helpers: build result dicts without "id" so both sides use text[:100] as key

def sem(text: str, source: str = "a.txt", rank_score: float = 0.9) -> dict:
    return {"text": text, "source": source, "score": rank_score, "metadata": {}}


def bm25r(text: str, source: str = "a.txt", score: float = 3.0) -> dict:
    return {"text": text, "source": source, "score": score, "metadata": {}}


class TestHybridRetrieverMerge:
    TEXT_A = "Machine learning models learn from data automatically."
    TEXT_B = "Deep learning uses many-layer neural networks."
    TEXT_C = "Natural language processing handles text understanding."

    def test_result_in_both_sources_ranked_above_semantic_only(self):
        merger = make_merger()
        # TEXT_A: semantic rank 1 only
        # TEXT_B: semantic rank 2 AND bm25 rank 1 → should overtake TEXT_A
        semantic = [sem(self.TEXT_A), sem(self.TEXT_B)]
        bm25 = [bm25r(self.TEXT_B), bm25r(self.TEXT_C)]
        merged = merger._merge_results(semantic, bm25)
        assert merged[0]["text"] == self.TEXT_B

    def test_semantic_only_result_gets_bm25_sentinel_rank(self):
        merger = make_merger()
        semantic = [sem(self.TEXT_A)]
        merged = merger._merge_results(semantic, [])
        assert merged[0]["bm25_rank"] == 999

    def test_bm25_only_result_gets_semantic_sentinel_rank(self):
        merger = make_merger()
        bm25 = [bm25r(self.TEXT_C)]
        merged = merger._merge_results([], bm25)
        assert merged[0]["semantic_rank"] == 999

    def test_merged_results_sorted_by_hybrid_score_descending(self):
        merger = make_merger()
        semantic = [sem(self.TEXT_A), sem(self.TEXT_B), sem(self.TEXT_C)]
        bm25 = [bm25r(self.TEXT_C), bm25r(self.TEXT_A)]
        merged = merger._merge_results(semantic, bm25)
        scores = [r["hybrid_score"] for r in merged]
        assert scores == sorted(scores, reverse=True)

    def test_merge_deduplicates_shared_text(self):
        merger = make_merger()
        semantic = [sem(self.TEXT_A)]
        bm25 = [bm25r(self.TEXT_A)]   # same text → same key
        merged = merger._merge_results(semantic, bm25)
        # Exactly one result, not two
        assert len(merged) == 1
        assert merged[0]["semantic_rank"] == 1
        assert merged[0]["bm25_rank"] == 1

    def test_higher_semantic_weight_favours_semantic_rank(self):
        # With very high semantic weight, a semantic-rank-1 result
        # should beat a bm25-rank-1 result that has no semantic rank
        merger = make_merger(semantic_weight=0.95, bm25_weight=0.05)
        semantic = [sem(self.TEXT_A)]
        bm25 = [bm25r(self.TEXT_B)]
        merged = merger._merge_results(semantic, bm25)
        # TEXT_A (semantic rank 1) should beat TEXT_B (bm25 rank 1)
        assert merged[0]["text"] == self.TEXT_A

    def test_hybrid_score_uses_rrf_formula(self):
        merger = make_merger(semantic_weight=0.7, bm25_weight=0.3)
        semantic = [sem(self.TEXT_A)]
        merged = merger._merge_results(semantic, [])
        k = 60
        expected = 0.7 / (k + 1) + 0.3 / (k + 999)
        assert merged[0]["hybrid_score"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# HybridRetriever — state management
# ---------------------------------------------------------------------------

class TestHybridRetrieverState:
    def test_reset_bm25_clears_index_and_sets_flag_false(self, retriever):
        retriever._bm25_indexed = True
        retriever.reset_bm25()
        assert retriever._bm25_indexed is False
        assert retriever.bm25.bm25 is None   # BM25Search recreated without index

    def test_build_bm25_on_empty_store_still_sets_indexed_flag(self, retriever):
        # Empty store: no documents → build_bm25_index should set flag anyway
        retriever.vector_store.collection.get.return_value = {
            "ids": [], "documents": [], "metadatas": []
        }
        retriever.build_bm25_index()
        assert retriever._bm25_indexed is True

    def test_search_triggers_bm25_build_if_not_yet_indexed(self, retriever):
        retriever._bm25_indexed = False
        retriever.vector_store.search.return_value = []
        retriever.search("query", top_k=3)
        # build_bm25_index was called → flag should now be True
        assert retriever._bm25_indexed is True
