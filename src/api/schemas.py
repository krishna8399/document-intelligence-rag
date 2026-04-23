"""Pydantic request/response schemas for the RAG API."""

from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    source_filter: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    num_chunks: int


class UploadResponse(BaseModel):
    files_processed: int
    total_chunks: int
    details: List[dict]


class StatsResponse(BaseModel):
    total_chunks: int
    llm_model: str
    embedding_model: str
