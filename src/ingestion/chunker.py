"""
Text chunking — splits documents into retrieval-friendly chunks.

Why chunking matters:
- LLMs have context limits (4K-128K tokens). We can't feed entire documents.
- Embedding models work best on short passages (< 512 tokens).
- Smaller chunks = more precise retrieval (less noise in each chunk).
- BUT too small = loses context. A sentence alone might be meaningless.

Chunk size = 500 tokens with 50 overlap is the sweet spot for most use cases.

Why overlap:
- Without overlap, a sentence split across two chunks loses meaning in both.
- 50 token overlap means the end of chunk N appears at the start of chunk N+1.
- This ensures no information is lost at chunk boundaries.
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class TextChunk:
    """A single chunk of text with metadata."""
    text: str
    chunk_id: int
    source: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentChunker:
    """
    Splits documents into chunks using recursive character splitting.

    Why RecursiveCharacterTextSplitter:
    - Tries to split on natural boundaries first (paragraphs, then sentences,
      then words, then characters)
    - Preserves semantic coherence within chunks
    - Better than fixed-size splitting which cuts mid-sentence

    The separator hierarchy: ["\\n\\n", "\\n", ". ", " ", ""]
    - First tries to split on paragraph breaks
    - If a paragraph is too long, splits on newlines
    - Then on sentence boundaries (". ")
    - Then on word boundaries (" ")
    - Last resort: character-level split
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: List[str] = None,
    ):
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str, source: str = "unknown") -> List[TextChunk]:
        splits = self.splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source}],
        )

        chunks = []
        for i, split in enumerate(splits):
            chunks.append(TextChunk(
                text=split.page_content,
                chunk_id=i,
                source=source,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(splits),
                    "char_count": len(split.page_content),
                },
            ))

        return chunks

    def chunk_document(self, parsed_doc) -> List[TextChunk]:
        """Chunk a ParsedDocument object."""
        return self.chunk_text(parsed_doc.text, source=parsed_doc.filename)
