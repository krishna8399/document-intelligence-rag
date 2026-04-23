"""
RAG Chain — the full retrieval-augmented generation pipeline.

This is the core of the entire project. It ties together:
1. Retrieval (hybrid search) → find relevant chunks
2. Context assembly → format chunks into a prompt
3. Generation (LLM) → produce an answer with citations

Why RAG instead of just asking the LLM directly:
- LLMs hallucinate. They'll confidently make up facts.
- RAG grounds the answer in YOUR documents. The LLM can only use
  what we give it in the context.
- Source citations let users verify the answer.
"""

from typing import List, Optional
from dataclasses import dataclass, field

import yaml

from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.embedder import EmbeddingEngine
from src.retrieval.hybrid import HybridRetriever
from src.generation.llm import LLMEngine


@dataclass
class RAGResponse:
    """A complete RAG response with sources."""
    answer: str
    sources: List[dict]
    query: str
    num_chunks_retrieved: int
    metadata: dict = field(default_factory=dict)


class RAGChain:
    """
    Full RAG pipeline: question → retrieve → generate → answer with sources.
    """

    def __init__(self, config_path: str = "configs/local.yaml", embedding_function=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        print("Initializing RAG chain...")

        if embedding_function is None:
            embedder = EmbeddingEngine(model_name=self.config["embedding"]["model"])
            embedding_function = embedder.get_embedding_function()

        self.vector_store = VectorStoreManager(
            persist_dir=self.config["vector_store"]["persist_directory"],
            collection_name=self.config["vector_store"]["collection_name"],
            embedding_function=embedding_function,
        )

        # Hybrid retriever
        retrieval_cfg = self.config["retrieval"]
        hybrid_cfg = retrieval_cfg.get("hybrid", {})
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            semantic_weight=hybrid_cfg.get("semantic_weight", 0.7),
            bm25_weight=hybrid_cfg.get("bm25_weight", 0.3),
        )

        # LLM
        self.llm = LLMEngine(config_path)

        # Settings
        self.top_k = retrieval_cfg.get("top_k", 5)

        # Conversation memory
        self.conversation_history: List[dict] = []

        print("RAG chain ready!")

    def query(
        self,
        question: str,
        source_filter: Optional[str] = None,
        use_history: bool = True,
    ) -> RAGResponse:
        """
        Answer a question using RAG.

        Args:
            question: user's question
            source_filter: optional filename to restrict search
            use_history: whether to include conversation history

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # 1. Retrieve relevant chunks
        results = self.retriever.search(
            query=question,
            top_k=self.top_k,
            source_filter=source_filter,
        )

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
                sources=[],
                query=question,
                num_chunks_retrieved=0,
            )

        # 2. Format context with source attribution
        context = self._format_context(results)

        # 3. Generate answer
        history = self.conversation_history if use_history else None
        answer = self.llm.generate(
            query=question,
            context=context,
            conversation_history=history,
        )

        # 4. Update conversation memory
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Keep last 20 messages (10 user+assistant exchanges)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # 5. Format sources for display
        sources = [
            {
                "source": r["source"],
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "score": r.get("hybrid_score", r.get("score", 0)),
                "chunk_id": r.get("metadata", {}).get("chunk_id", "?"),
            }
            for r in results
        ]

        return RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            num_chunks_retrieved=len(results),
        )

    def _format_context(self, results: List[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        context_parts = []
        for i, result in enumerate(results):
            source = result["source"]
            chunk_id = result.get("metadata", {}).get("chunk_id", "?")
            text = result["text"]
            context_parts.append(
                f"[Source: {source}, Chunk {chunk_id}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def clear_history(self):
        """Clear conversation memory."""
        self.conversation_history = []

    def get_stats(self) -> dict:
        """Get system statistics."""
        vs_stats = self.vector_store.get_stats()
        return {
            **vs_stats,
            "conversation_turns": len(self.conversation_history) // 2,
            "llm_model": self.config["llm"]["model"],
            "embedding_model": self.config["embedding"]["model"],
        }
