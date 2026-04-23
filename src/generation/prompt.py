"""
Prompt templates for the RAG chain.

Keeping prompts here (not hardcoded in rag_chain.py) makes it easy to
iterate on instructions without touching pipeline logic.
"""

RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Answer using ONLY the information in the context below. Do not use prior knowledge.
- For every claim, cite the source document and chunk number, e.g. [ai_healthcare_report.txt, Chunk 3].
- If the context does not contain enough information to answer, say: "I don't have enough information in the uploaded documents to answer this."
- Be concise and direct. Avoid repeating the question back.
"""

RAG_USER_TEMPLATE = """Context from documents:
---
{context}
---

Question: {question}

Answer the question based ONLY on the context above. Cite the source document for each claim."""
