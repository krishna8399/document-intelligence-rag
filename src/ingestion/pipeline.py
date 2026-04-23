"""
Full ingestion pipeline: file → parse → chunk → embed → store.

This is the "write" path of the RAG system.
The "read" path is in retrieval/.
"""

from pathlib import Path

import yaml

from src.ingestion.parser import parse_document
from src.ingestion.chunker import DocumentChunker
from src.retrieval.embedder import EmbeddingEngine
from src.retrieval.vector_store import VectorStoreManager


class IngestionPipeline:
    """
    End-to-end document ingestion.

    Flow: files → parse → chunk → embed → store in ChromaDB
    """

    def __init__(self, config_path: str = "configs/local.yaml", embedding_function=None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        chunk_cfg = self.config["chunking"]
        self.chunker = DocumentChunker(
            chunk_size=chunk_cfg["chunk_size"],
            chunk_overlap=chunk_cfg["chunk_overlap"],
            separators=chunk_cfg.get("separators"),
        )

        if embedding_function is None:
            embedder = EmbeddingEngine(model_name=self.config["embedding"]["model"])
            embedding_function = embedder.get_embedding_function()

        self.vector_store = VectorStoreManager(
            persist_dir=self.config["vector_store"]["persist_directory"],
            collection_name=self.config["vector_store"]["collection_name"],
            embedding_function=embedding_function,
        )

    def ingest_file(self, file_path: str) -> int:
        """
        Ingest a single file into the vector store.

        Returns:
            Number of chunks created
        """
        print(f"\nIngesting: {file_path}")

        doc = parse_document(file_path)

        chunks = self.chunker.chunk_document(doc)
        print(f"  Chunked: {len(chunks)} chunks")

        self.vector_store.delete_source(doc.filename)
        self.vector_store.add_chunks(chunks)
        print(f"  Stored in vector DB")

        return len(chunks)

    def ingest_directory(self, dir_path: str) -> dict:
        """
        Ingest all supported files in a directory.

        Returns:
            Summary dict with file counts and chunk counts
        """
        dir_path = Path(dir_path)
        supported = {".pdf", ".docx", ".txt", ".md"}

        files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        ]

        if not files:
            print(f"No supported files found in {dir_path}")
            return {"files": 0, "chunks": 0}

        print(f"Found {len(files)} files to ingest")

        total_chunks = 0
        results = []
        for file_path in sorted(files):
            n_chunks = self.ingest_file(str(file_path))
            total_chunks += n_chunks
            results.append({"file": file_path.name, "chunks": n_chunks})

        print(f"\nIngestion complete: {len(files)} files, {total_chunks} chunks")
        return {
            "files": len(files),
            "chunks": total_chunks,
            "details": results,
        }

    def get_stats(self) -> dict:
        """Get vector store statistics."""
        return self.vector_store.get_stats()
