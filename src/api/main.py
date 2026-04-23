"""
FastAPI application for the RAG system.

Endpoints:
    POST /upload     — Upload documents for ingestion
    POST /query      — Ask a question, get RAG answer with sources
    GET  /stats      — System statistics
    GET  /health     — Health check
    DELETE /clear    — Clear all documents
"""

import os
import shutil
import tempfile
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.embedder import EmbeddingEngine
from src.ingestion.pipeline import IngestionPipeline
from src.generation.rag_chain import RAGChain
from src.api.schemas import QueryRequest, QueryResponse, UploadResponse, StatsResponse

config_path = os.environ.get("RAG_CONFIG", "configs/local.yaml")
ingestion: IngestionPipeline = None
rag_chain: RAGChain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ingestion, rag_chain
    print("Starting RAG system...")
    embedder = EmbeddingEngine(
        model_name=_read_config()["embedding"]["model"]
    )
    ef = embedder.get_embedding_function()
    ingestion = IngestionPipeline(config_path, embedding_function=ef)
    rag_chain = RAGChain(config_path, embedding_function=ef)
    print("RAG system ready!")
    yield


def _read_config() -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


app = FastAPI(
    title="Document Intelligence RAG API",
    description="Upload documents and ask questions. Answers are grounded in your documents with source citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "healthy", "config": config_path}


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload one or more documents for ingestion."""
    tmp_dir = tempfile.mkdtemp()

    try:
        total_chunks = 0
        details = []

        for file in files:
            suffix = Path(file.filename).suffix.lower()
            if suffix not in {".pdf", ".docx", ".txt", ".md"}:
                raise HTTPException(400, f"Unsupported file: {file.filename}")

            tmp_path = os.path.join(tmp_dir, file.filename)
            with open(tmp_path, "wb") as fp:
                shutil.copyfileobj(file.file, fp)

            n_chunks = ingestion.ingest_file(tmp_path)
            total_chunks += n_chunks
            details.append({"file": file.filename, "chunks": n_chunks})

        rag_chain.retriever.build_bm25_index()

        return UploadResponse(
            files_processed=len(files),
            total_chunks=total_chunks,
            details=details,
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question about uploaded documents."""
    if rag_chain.vector_store.collection.count() == 0:
        raise HTTPException(400, "No documents uploaded yet. Upload documents first.")

    response = rag_chain.query(
        question=request.question,
        source_filter=request.source_filter,
    )

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        num_chunks=response.num_chunks_retrieved,
    )


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    stats = rag_chain.get_stats()
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        llm_model=stats["llm_model"],
        embedding_model=stats["embedding_model"],
    )


@app.delete("/clear")
def clear_documents():
    """Delete all documents from the vector store."""
    rag_chain.vector_store.clear()
    rag_chain.retriever.reset_bm25()
    rag_chain.clear_history()
    return {"status": "cleared"}
