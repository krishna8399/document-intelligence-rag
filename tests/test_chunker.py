"""
Comprehensive tests for the document chunker.

Covers: chunk size limits, overlap, empty/whitespace text, very long text,
unicode/special characters, metadata correctness, and ParsedDocument properties.
"""

import pytest
from src.ingestion.chunker import DocumentChunker, TextChunk
from src.ingestion.parser import ParsedDocument


def make_doc(text: str, filename: str = "test.txt") -> ParsedDocument:
    return ParsedDocument(filename=filename, text=text, pages=1, file_type="txt")


# ---------------------------------------------------------------------------
# ParsedDocument
# ---------------------------------------------------------------------------

class TestParsedDocument:
    def test_total_chars_property_reflects_text_length(self):
        doc = make_doc("Hello world")
        assert doc.total_chars == 11

    def test_total_chars_empty_text_is_zero(self):
        doc = make_doc("")
        assert doc.total_chars == 0

    def test_total_chars_updates_when_text_changes(self):
        doc = make_doc("abc")
        doc.text = "abcdef"
        assert doc.total_chars == 6


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    def setup_method(self):
        self.chunker = DocumentChunker(chunk_size=100, chunk_overlap=10)

    # ── basic correctness ─────────────────────────────────────────────────

    def test_short_text_produces_single_chunk(self):
        chunks = self.chunker.chunk_text("This is a short text.", source="t.txt")
        assert len(chunks) == 1
        assert isinstance(chunks[0], TextChunk)

    def test_long_text_produces_multiple_chunks(self):
        chunks = self.chunker.chunk_text("This is a sentence. " * 100, source="t.txt")
        assert len(chunks) > 1

    def test_chunk_ids_are_zero_based_and_sequential(self):
        chunks = self.chunker.chunk_text("Word " * 200, source="t.txt")
        assert [c.chunk_id for c in chunks] == list(range(len(chunks)))

    def test_source_propagated_to_every_chunk(self):
        chunks = self.chunker.chunk_text("Some text here.", source="my_report.pdf")
        assert all(c.source == "my_report.pdf" for c in chunks)

    def test_chunk_document_matches_chunk_text(self):
        doc = make_doc("Alpha beta gamma. " * 50, filename="doc.txt")
        by_doc = self.chunker.chunk_document(doc)
        by_text = self.chunker.chunk_text(doc.text, source=doc.filename)
        assert len(by_doc) == len(by_text)
        assert [c.text for c in by_doc] == [c.text for c in by_text]

    # ── size limits ───────────────────────────────────────────────────────

    def test_chunk_size_respected_with_small_limit(self):
        # With chunk_size=50, no chunk should exceed 2× the limit
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=5)
        chunks = chunker.chunk_text("Hello world. " * 100, source="t.txt")
        for c in chunks:
            assert len(c.text) <= 100, f"Chunk too large: {len(c.text)} chars"

    def test_chunk_size_respected_with_standard_size(self):
        # chunk_size=100: expect most chunks well under 150 chars
        chunks = self.chunker.chunk_text("Sentence number one. " * 200, source="t.txt")
        oversized = [c for c in chunks if len(c.text) > 150]
        assert len(oversized) == 0, f"{len(oversized)} chunks exceeded 1.5× chunk_size"

    def test_very_long_text_produces_many_chunks(self):
        text = "This is a test sentence with several words. " * 500  # ~22 000 chars
        chunks = self.chunker.chunk_text(text, source="long.txt")
        assert len(chunks) > 50
        assert all(len(c.text) > 0 for c in chunks)

    # ── overlap ───────────────────────────────────────────────────────────

    def test_overlap_creates_shared_content_between_adjacent_chunks(self):
        # Repetitive words → predictable split; some words at end of chunk N
        # must reappear at start of chunk N+1 due to the 10-char overlap
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = " ".join([f"word{i}" for i in range(200)])  # ~1 400 chars
        chunks = chunker.chunk_text(text, source="t.txt")
        assert len(chunks) >= 2

        words_end_of_first = set(chunks[0].text.split()[-4:])
        words_start_of_second = set(chunks[1].text.split()[:8])
        shared = words_end_of_first & words_start_of_second
        assert shared, (
            f"No shared words between chunk boundaries.\n"
            f"  End of chunk 0: {chunks[0].text[-60:]!r}\n"
            f"  Start of chunk 1: {chunks[1].text[:60:]!r}"
        )

    # ── edge cases ────────────────────────────────────────────────────────

    def test_empty_text_returns_no_chunks(self):
        assert self.chunker.chunk_text("", source="t.txt") == []

    def test_empty_document_returns_no_chunks(self):
        assert self.chunker.chunk_document(make_doc("")) == []

    def test_whitespace_only_text_handled_gracefully(self):
        chunks = self.chunker.chunk_text("   \n\n\t  \n   ", source="t.txt")
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, TextChunk)

    def test_single_word_returns_one_chunk(self):
        chunks = self.chunker.chunk_text("Hello", source="t.txt")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello"

    # ── unicode & special characters ─────────────────────────────────────

    def test_unicode_characters_round_trip_intact(self):
        text = "Héllo Wörld. Привет мир. 你好世界. " * 20
        chunks = self.chunker.chunk_text(text, source="unicode.txt")
        combined = " ".join(c.text for c in chunks)
        assert "Привет" in combined
        assert "你好世界" in combined
        assert "Héllo" in combined

    def test_emoji_preserved(self):
        text = ("AI is transforming healthcare 🏥. "
                "Robots are amazing 🤖. ") * 20
        chunks = self.chunker.chunk_text(text, source="emoji.txt")
        combined = "".join(c.text for c in chunks)
        assert "🏥" in combined
        assert "🤖" in combined

    def test_code_snippet_characters_preserved(self):
        text = "Function: def foo(x): return x**2 + 3. " * 30
        chunks = self.chunker.chunk_text(text, source="code.txt")
        combined = "".join(c.text for c in chunks)
        assert "def foo" in combined
        assert "x**2" in combined

    def test_special_punctuation_preserved(self):
        text = "Price: $1,000.00 — a 50% discount! (Was: $2,000.) " * 20
        chunks = self.chunker.chunk_text(text, source="prices.txt")
        combined = "".join(c.text for c in chunks)
        assert "$1,000.00" in combined
        assert "50%" in combined

    # ── metadata ─────────────────────────────────────────────────────────

    def test_metadata_contains_source_chunk_id_and_total_chunks(self):
        chunks = self.chunker.chunk_text("Word " * 100, source="doc.txt")
        for c in chunks:
            assert c.metadata["source"] == "doc.txt"
            assert c.metadata["chunk_id"] == c.chunk_id
            assert "total_chunks" in c.metadata

    def test_metadata_total_chunks_matches_actual_count(self):
        chunks = self.chunker.chunk_text("Word " * 100, source="doc.txt")
        expected_total = len(chunks)
        for c in chunks:
            assert c.metadata["total_chunks"] == expected_total

    def test_metadata_char_count_matches_text_length(self):
        chunks = self.chunker.chunk_text("Hello world sentence. " * 50, source="t.txt")
        for c in chunks:
            assert c.metadata["char_count"] == len(c.text)
