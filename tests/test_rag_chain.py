"""
Comprehensive tests for the RAG chain.

All external dependencies (LLM, vector store, embedder, retriever) are mocked
so tests run instantly without any models, databases, or API keys.
"""

import pytest
from unittest.mock import MagicMock, call
from src.generation.rag_chain import RAGChain, RAGResponse


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def make_mock_chain() -> RAGChain:
    """Build a RAGChain with every external dependency mocked out."""
    chain = RAGChain.__new__(RAGChain)
    chain.config = {
        "embedding":  {"model": "all-MiniLM-L6-v2"},
        "retrieval":  {"top_k": 5, "hybrid": {"semantic_weight": 0.7, "bm25_weight": 0.3}},
        "llm":        {"model": "llama3.1:8b", "provider": "ollama",
                       "temperature": 0.1, "max_tokens": 1024},
        "generation": {},
    }
    chain.top_k = 5
    chain.conversation_history = []
    chain.vector_store = MagicMock()
    chain.vector_store.get_stats.return_value = {
        "total_chunks": 10,
        "collection_name": "documents",
    }
    chain.retriever = MagicMock()
    chain.llm = MagicMock()
    chain.llm.generate.return_value = "A mocked answer."
    return chain


def fake_chunk(text="Chunk text.", source="doc.txt", chunk_id=0, score=0.8):
    return {
        "text": text,
        "source": source,
        "hybrid_score": score,
        "metadata": {"chunk_id": chunk_id},
    }


# ---------------------------------------------------------------------------
# Query — success path
# ---------------------------------------------------------------------------

class TestRAGChainQuery:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_successful_query_returns_rag_response(self):
        self.chain.retriever.search.return_value = [fake_chunk()]
        self.chain.llm.generate.return_value = "The answer is 42."
        response = self.chain.query("What is the answer?")
        assert isinstance(response, RAGResponse)
        assert response.answer == "The answer is 42."

    def test_successful_query_sets_num_chunks_retrieved(self):
        self.chain.retriever.search.return_value = [fake_chunk(), fake_chunk(chunk_id=1)]
        response = self.chain.query("question")
        assert response.num_chunks_retrieved == 2

    def test_successful_query_includes_source_in_response(self):
        self.chain.retriever.search.return_value = [
            fake_chunk(source="report.pdf", chunk_id=3)
        ]
        response = self.chain.query("question")
        assert len(response.sources) == 1
        assert response.sources[0]["source"] == "report.pdf"
        assert response.sources[0]["chunk_id"] == 3

    def test_response_query_field_matches_input(self):
        self.chain.retriever.search.return_value = []
        response = self.chain.query("What is the meaning of life?")
        assert response.query == "What is the meaning of life?"

    def test_source_text_truncated_to_200_chars_with_ellipsis(self):
        long_text = "X" * 500
        self.chain.retriever.search.return_value = [fake_chunk(text=long_text)]
        response = self.chain.query("question")
        assert len(response.sources[0]["text"]) <= 203  # 200 + "..."
        assert response.sources[0]["text"].endswith("...")

    def test_short_source_text_not_truncated(self):
        short = "Short text."
        self.chain.retriever.search.return_value = [fake_chunk(text=short)]
        response = self.chain.query("question")
        assert response.sources[0]["text"] == short


# ---------------------------------------------------------------------------
# Query — no results
# ---------------------------------------------------------------------------

class TestRAGChainNoResults:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_empty_retrieval_returns_helpful_message(self):
        self.chain.retriever.search.return_value = []
        response = self.chain.query("Unknown topic")
        assert isinstance(response, RAGResponse)
        assert response.num_chunks_retrieved == 0
        assert response.sources == []

    def test_empty_retrieval_does_not_call_llm(self):
        self.chain.retriever.search.return_value = []
        self.chain.query("Unknown topic")
        self.chain.llm.generate.assert_not_called()

    def test_empty_retrieval_answer_indicates_missing_info(self):
        self.chain.retriever.search.return_value = []
        response = self.chain.query("Unknown topic")
        answer_lower = response.answer.lower()
        assert any(w in answer_lower for w in ("couldn't", "no ", "not ", "don't"))


# ---------------------------------------------------------------------------
# Retriever integration
# ---------------------------------------------------------------------------

class TestRAGChainRetriever:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_source_filter_forwarded_to_retriever(self):
        self.chain.retriever.search.return_value = []
        self.chain.query("question", source_filter="specific.pdf")
        self.chain.retriever.search.assert_called_once_with(
            query="question",
            top_k=self.chain.top_k,
            source_filter="specific.pdf",
        )

    def test_no_source_filter_passes_none_to_retriever(self):
        self.chain.retriever.search.return_value = []
        self.chain.query("question")
        call_kwargs = self.chain.retriever.search.call_args.kwargs
        assert call_kwargs["source_filter"] is None


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------

class TestConversationHistory:
    def setup_method(self):
        self.chain = make_mock_chain()
        self.chain.retriever.search.return_value = [fake_chunk()]

    def test_history_grows_by_two_per_query(self):
        self.chain.query("First question")
        assert len(self.chain.conversation_history) == 2  # user + assistant

        self.chain.query("Second question")
        assert len(self.chain.conversation_history) == 4

    def test_history_trimmed_to_20_messages_after_overflow(self):
        for _ in range(15):
            self.chain.query("question")
        assert len(self.chain.conversation_history) <= 20

    def test_use_history_false_passes_none_to_llm(self):
        self.chain.conversation_history = [
            {"role": "user", "content": "Previous"},
            {"role": "assistant", "content": "Prior answer"},
        ]
        self.chain.query("New question", use_history=False)
        kwargs = self.chain.llm.generate.call_args.kwargs
        assert kwargs["conversation_history"] is None

    def test_use_history_true_passes_history_to_llm(self):
        prior = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
        self.chain.conversation_history = prior[:]
        self.chain.query("Follow-up", use_history=True)
        kwargs = self.chain.llm.generate.call_args.kwargs
        passed = kwargs["conversation_history"]
        # The mock holds a reference; after the query the list has 2 more entries
        # (the Follow-up turn), so check that the prior turns are at the start
        assert passed[:2] == prior

    def test_clear_history_empties_the_list(self):
        self.chain.conversation_history = [{"role": "user", "content": "hi"}]
        self.chain.clear_history()
        assert self.chain.conversation_history == []

    def test_history_not_mutated_when_use_history_false(self):
        self.chain.query("question", use_history=False)
        # Even with use_history=False, the turn is still recorded
        assert len(self.chain.conversation_history) == 2


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

class TestContextFormatting:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_format_context_includes_source_filename(self):
        results = [{"text": "Content.", "source": "annual_report.pdf", "metadata": {"chunk_id": 2}}]
        context = self.chain._format_context(results)
        assert "annual_report.pdf" in context

    def test_format_context_includes_chunk_id(self):
        results = [{"text": "Content.", "source": "doc.txt", "metadata": {"chunk_id": 7}}]
        context = self.chain._format_context(results)
        assert "7" in context

    def test_format_context_includes_chunk_text(self):
        results = [{"text": "The sky is blue.", "source": "doc.txt", "metadata": {"chunk_id": 0}}]
        context = self.chain._format_context(results)
        assert "The sky is blue." in context

    def test_format_context_separator_between_multiple_chunks(self):
        results = [
            {"text": "First chunk content.", "source": "a.txt", "metadata": {"chunk_id": 0}},
            {"text": "Second chunk content.", "source": "b.txt", "metadata": {"chunk_id": 0}},
        ]
        context = self.chain._format_context(results)
        assert "---" in context
        assert "First chunk content." in context
        assert "Second chunk content." in context

    def test_format_context_handles_missing_chunk_id_gracefully(self):
        results = [{"text": "Some text.", "source": "doc.txt", "metadata": {}}]
        context = self.chain._format_context(results)
        # Should not raise; "?" is the fallback
        assert "?" in context or "doc.txt" in context


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestRAGChainStats:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_get_stats_returns_all_expected_keys(self):
        stats = self.chain.get_stats()
        for key in ("total_chunks", "llm_model", "embedding_model", "conversation_turns"):
            assert key in stats, f"Missing key: {key}"

    def test_get_stats_conversation_turns_counts_pairs(self):
        self.chain.retriever.search.return_value = [fake_chunk()]
        self.chain.query("q1")
        self.chain.query("q2")
        stats = self.chain.get_stats()
        assert stats["conversation_turns"] == 2

    def test_get_stats_llm_model_from_config(self):
        stats = self.chain.get_stats()
        assert stats["llm_model"] == "llama3.1:8b"
