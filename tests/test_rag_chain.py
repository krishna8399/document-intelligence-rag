"""
Tests for the RAG chain — uses mocks so no LLM/vector DB needed.

These tests verify the pipeline logic (formatting, history trimming,
no-results handling) without hitting real models.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.generation.rag_chain import RAGChain, RAGResponse


def make_mock_chain(config_path: str = "configs/local.yaml") -> RAGChain:
    """Build a RAGChain with all external dependencies mocked."""
    with (
        patch("src.generation.rag_chain.EmbeddingEngine"),
        patch("src.generation.rag_chain.VectorStoreManager"),
        patch("src.generation.rag_chain.HybridRetriever"),
        patch("src.generation.rag_chain.LLMEngine"),
        patch("builtins.open", create=True),
        patch("yaml.safe_load", return_value={
            "embedding": {"model": "all-MiniLM-L6-v2"},
            "vector_store": {"persist_directory": "data/chroma_db", "collection_name": "documents"},
            "retrieval": {"top_k": 5, "hybrid": {"semantic_weight": 0.7, "bm25_weight": 0.3}},
            "llm": {"model": "llama3.1:8b", "provider": "ollama"},
            "generation": {"system_prompt": "You are a helpful assistant."},
        }),
    ):
        chain = RAGChain.__new__(RAGChain)
        chain.config = {
            "embedding": {"model": "all-MiniLM-L6-v2"},
            "retrieval": {"top_k": 5, "hybrid": {"semantic_weight": 0.7, "bm25_weight": 0.3}},
            "llm": {"model": "llama3.1:8b"},
        }
        chain.top_k = 5
        chain.conversation_history = []
        chain.embedder = MagicMock()
        chain.vector_store = MagicMock()
        chain.retriever = MagicMock()
        chain.llm = MagicMock()
        return chain


class TestRAGChain:
    def setup_method(self):
        self.chain = make_mock_chain()

    def test_empty_retrieval_returns_no_info_response(self):
        self.chain.retriever.search.return_value = []
        response = self.chain.query("What is AI?")
        assert isinstance(response, RAGResponse)
        assert response.num_chunks_retrieved == 0
        assert "couldn't find" in response.answer.lower() or "no" in response.answer.lower()

    def test_successful_query_returns_answer_and_sources(self):
        fake_chunks = [
            {
                "text": "AI is the simulation of human intelligence.",
                "source": "ai_doc.txt",
                "hybrid_score": 0.9,
                "metadata": {"chunk_id": 0},
            }
        ]
        self.chain.retriever.search.return_value = fake_chunks
        self.chain.llm.generate.return_value = "AI simulates human intelligence [ai_doc.txt, Chunk 0]."

        response = self.chain.query("What is AI?")

        assert response.answer == "AI simulates human intelligence [ai_doc.txt, Chunk 0]."
        assert response.num_chunks_retrieved == 1
        assert response.sources[0]["source"] == "ai_doc.txt"

    def test_conversation_history_updated(self):
        fake_chunk = [{"text": "Some text.", "source": "doc.txt", "hybrid_score": 0.8, "metadata": {"chunk_id": 0}}]
        self.chain.retriever.search.return_value = fake_chunk
        self.chain.llm.generate.return_value = "An answer."

        self.chain.query("First question")
        self.chain.query("Second question")

        assert len(self.chain.conversation_history) == 4  # 2 questions + 2 answers

    def test_history_trimmed_to_20_messages(self):
        fake_chunk = [{"text": "text", "source": "doc.txt", "hybrid_score": 0.5, "metadata": {"chunk_id": 0}}]
        self.chain.retriever.search.return_value = fake_chunk
        self.chain.llm.generate.return_value = "answer"

        for _ in range(15):
            self.chain.query("question")

        assert len(self.chain.conversation_history) <= 20

    def test_clear_history(self):
        self.chain.conversation_history = [{"role": "user", "content": "hi"}]
        self.chain.clear_history()
        assert self.chain.conversation_history == []

    def test_source_text_truncated_to_200_chars(self):
        long_text = "X" * 500
        fake_chunk = [{"text": long_text, "source": "doc.txt", "hybrid_score": 0.8, "metadata": {"chunk_id": 0}}]
        self.chain.retriever.search.return_value = fake_chunk
        self.chain.llm.generate.return_value = "answer"

        response = self.chain.query("question")
        assert len(response.sources[0]["text"]) <= 203  # 200 + "..."

    def test_format_context_includes_source_and_chunk(self):
        results = [
            {"text": "Some content here.", "source": "report.pdf", "metadata": {"chunk_id": 3}},
        ]
        context = self.chain._format_context(results)
        assert "report.pdf" in context
        assert "3" in context
        assert "Some content here." in context
