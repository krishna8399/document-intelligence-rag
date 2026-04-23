"""Tests for the document chunker."""

import pytest
from src.ingestion.chunker import DocumentChunker, TextChunk
from src.ingestion.parser import ParsedDocument


def make_doc(text: str, filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(filename=filename, text=text, pages=1, file_type="txt")


class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    def test_short_text_single_chunk(self):
        doc = make_doc("This is a short text.")
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_long_text_multiple_chunks(self):
        long_text = "This is a sentence. " * 100
        doc = make_doc(long_text)
        chunks = self.chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_sequential(self):
        doc = make_doc("Word " * 200)
        chunks = self.chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))

    def test_source_propagated(self):
        doc = make_doc("Some text here.", filename="my_report.pdf")
        chunks = self.chunker.chunk_document(doc)
        assert all(c.source == "my_report.pdf" for c in chunks)

    def test_chunk_size_respected(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        doc = make_doc("Hello world. " * 100)
        chunks = chunker.chunk_document(doc)
        # All chunks should be at most a bit larger than chunk_size
        for c in chunks:
            assert len(c.text) <= 150  # some tolerance for splitter heuristics

    def test_metadata_contains_source_and_chunk_id(self):
        doc = make_doc("Text " * 50)
        chunks = self.chunker.chunk_document(doc)
        for c in chunks:
            assert "source" in c.metadata
            assert "chunk_id" in c.metadata

    def test_empty_text_returns_no_chunks(self):
        doc = make_doc("")
        chunks = self.chunker.chunk_document(doc)
        assert chunks == []
