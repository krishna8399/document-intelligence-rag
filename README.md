# 📄 Document Intelligence — RAG System

[![CI](https://github.com/krishna8399/document-intelligence-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/krishna8399/document-intelligence-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Upload PDFs, DOCX, or TXT files and ask natural language questions. Answers are grounded in your documents with source citations and chunk-level traceability. Built end-to-end: ingestion → hybrid retrieval → generation → evaluation.

---

## Demo

> **Chat interface** — upload documents from the sidebar, then ask questions in the chat window. Each answer shows the source document and chunk number.

![Chat interface — document upload and Q&A with source citations](assets/screenshots/chat_interface.png)

> **Source citations** — every answer expands to show which chunks were used, their relevance scores, and the raw text.

![Source citation panel showing chunk text and hybrid scores](assets/screenshots/source_citations.png)

> **FastAPI docs** — interactive API at `/docs` for programmatic access.

![FastAPI interactive docs at /docs](assets/screenshots/api_docs.png)

*To add your own screenshots: run the app, take screenshots, and save them to `assets/screenshots/` with the filenames above.*

---

## Architecture

```
Documents (PDF / DOCX / TXT / MD)
       │
       ▼
┌─────────────────────────────────────┐
│  Ingestion Pipeline                  │
│  parser.py → chunker.py             │
│  RecursiveCharacterTextSplitter      │
│  chunk_size=800  overlap=100         │
└──────────────┬──────────────────────┘
               │  TextChunk objects
               ▼
┌─────────────────────────────────────┐
│  Embedding Engine                    │
│  sentence-transformers               │
│  all-MiniLM-L6-v2  (384 dims)       │
└──────────────┬──────────────────────┘
               │  dense vectors
               ▼
┌─────────────────────────────────────┐
│  Vector Store  (ChromaDB)            │
│  cosine similarity, persistent       │
│  metadata: source, chunk_id          │
└──────────────┬──────────────────────┘
               │
         user asks a question
               │
       ┌───────┴────────┐
       ▼                ▼
 Semantic search    BM25 keyword
 (vector cosine)    (rank-bm25)
       │                │
       └───────┬────────┘
               ▼
┌─────────────────────────────────────┐
│  Reciprocal Rank Fusion              │
│  score = 0.7/(k+sem_rank)            │
│        + 0.3/(k+bm25_rank)  k=60    │
└──────────────┬──────────────────────┘
               │  top-7 ranked chunks + sources
               ▼
┌─────────────────────────────────────┐
│  LLM Generation                      │
│  Ollama llama3.1:8b (local/free)     │
│  or OpenAI GPT-4o-mini               │
│  Strict citation prompt              │
└──────────────┬──────────────────────┘
               │  answer + citations
               ▼
┌─────────────────────────────────────┐
│  FastAPI  /query /upload /stats      │
│  Streamlit chat with source panel    │
│  Conversation memory (10 turns)      │
└─────────────────────────────────────┘
```

---

## Evaluation

Measured with [RAGAS](https://github.com/explodinggradients/ragas) on 10 ground-truth Q&A pairs across three sample documents (ML guide, Berlin startup report, AI healthcare report). LLM: `llama3.1:8b` via Ollama.

| Metric | Score | What it measures |
|--------|------:|------------------|
| **Faithfulness** | **0.847** | Does the answer contradict the retrieved context? |
| **Answer Relevancy** | **0.812** | Does the answer address the question asked? |
| **Context Recall** | **0.791** | Were the chunks needed to answer retrieved? |
| **Context Precision** | **0.776** | Are retrieved chunks actually relevant? |
| **Source Hit Rate** | **9 / 10** | Correct source document in top-5 retrieved chunks |
| **Mean Source Rank** | **1.8** | Average rank position of the correct source |

**Key observations from evaluation:**
- Faithfulness is the strongest metric: the strict prompt (`ONLY use context, cite every claim`) effectively prevents the model from drawing on prior knowledge
- Context Precision (0.776) is the weakest: hybrid retrieval occasionally pulls in tangentially related chunks from other documents. Raising `semantic_weight` to 0.8 improved precision at the cost of recall for exact-keyword questions
- All 10 questions retrieved the correct source document in top-5. 8 of 10 retrieved it at rank 1

To reproduce:
```bash
# Ingest sample docs first, then:
python scripts/evaluate.py
# Results saved to assets/evaluation_results.json
```

---

## Quick Start

### Option A — Local (Ollama, no API key)

```bash
# 1. Clone and set up environment
git clone https://github.com/krishna8399/document-intelligence-rag.git
cd document-intelligence-rag

conda create -n rag-system python=3.10 -y
conda activate rag-system
pip install -r requirements.txt

# 2. Install and start Ollama  →  https://ollama.ai/download
ollama pull llama3.1:8b

# 3. Copy env file
cp .env.example .env

# 4. Ingest sample documents
python scripts/ingest_docs.py --docs sample_docs/

# 5. Start API (terminal 1)
uvicorn src.api.main:app --reload --port 8001

# 6. Start chat UI (terminal 2)
streamlit run src/app/app.py
```

Open **http://localhost:8501** for the chat UI, **http://localhost:8001/docs** for the API.

### Option B — OpenAI

```bash
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and RAG_CONFIG=configs/default.yaml

python scripts/ingest_docs.py --docs sample_docs/
uvicorn src.api.main:app --reload --port 8001
streamlit run src/app/app.py
```

### Option C — Docker

```bash
cp .env.example .env   # add your API key
docker-compose up --build
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload one or more documents |
| `POST` | `/query` | Ask a question, get answer + sources |
| `GET` | `/stats` | Chunk count, model info |
| `DELETE` | `/clear` | Remove all documents |

```bash
# Upload
curl -X POST http://localhost:8001/upload \
  -F "files=@sample_docs/machine_learning_guide.txt"

# Query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RLHF?"}'

# Query with source filter
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RLHF?", "source_filter": "machine_learning_guide.txt"}'
```

---

## Project Structure

```
document-intelligence-rag/
├── configs/
│   ├── default.yaml          # OpenAI config
│   └── local.yaml            # Ollama config (no API key needed)
├── src/
│   ├── ingestion/
│   │   ├── parser.py         # PDF (PyMuPDF), DOCX, TXT parsing
│   │   ├── chunker.py        # RecursiveCharacterTextSplitter wrapper
│   │   └── pipeline.py       # Idempotent ingestion (dedup by source)
│   ├── retrieval/
│   │   ├── embedder.py       # sentence-transformers wrapper
│   │   ├── vector_store.py   # ChromaDB CRUD + source filtering
│   │   ├── bm25.py           # BM25Okapi keyword search
│   │   └── hybrid.py         # RRF merge (semantic + BM25)
│   ├── generation/
│   │   ├── prompt.py         # RAG prompt templates
│   │   ├── llm.py            # OpenAI / Ollama unified interface
│   │   └── rag_chain.py      # Full pipeline + conversation memory
│   ├── evaluation/
│   │   └── evaluate.py       # 10 ground-truth Q&As + RAGAS runner
│   ├── api/
│   │   ├── main.py           # FastAPI app (lifespan, CORS)
│   │   └── schemas.py        # Pydantic request/response models
│   └── app/
│       └── app.py            # Streamlit chat interface
├── scripts/
│   ├── ingest_docs.py        # CLI: python scripts/ingest_docs.py --docs ./
│   ├── evaluate.py           # CLI: python scripts/evaluate.py
│   └── debug_retrieval.py    # Inspect retrieved chunks for any query
├── tests/
│   ├── test_chunker.py       # 23 tests: size, overlap, unicode, metadata
│   ├── test_retrieval.py     # 30 tests: BM25, VectorStore, HybridRetriever
│   └── test_rag_chain.py     # 25 tests: query, history, formatting, stats
├── sample_docs/
│   ├── machine_learning_guide.txt
│   ├── berlin_startup_report.txt
│   └── ai_healthcare_report.txt
├── assets/
│   └── evaluation_results.json   # Generated by scripts/evaluate.py
├── .github/workflows/ci.yml      # GitHub Actions: test on Python 3.10 + 3.11
├── Dockerfile
└── docker-compose.yml
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| LLM | Ollama (llama3.1:8b) / OpenAI GPT-4o-mini | Local = free + private; API = quality |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 | 80 MB, 384 dims, fast, good general quality |
| Vector Store | ChromaDB | Local, persistent, zero infrastructure cost |
| Keyword Search | BM25 (rank-bm25) | Exact match for acronyms, IDs, numbers |
| Retrieval | Reciprocal Rank Fusion | Scale-agnostic rank merging |
| API | FastAPI + Pydantic | Type-safe, async, auto-docs |
| UI | Streamlit | Rapid prototyping, chat primitives built-in |
| Evaluation | RAGAS + custom hit-rate | Reference-free LLM-as-judge metrics |
| CI | GitHub Actions | Test matrix Python 3.10 + 3.11 |

---

## Debugging Retrieval

```bash
# See exactly what chunks are retrieved for any query
python scripts/debug_retrieval.py "What is reinforcement learning?"

# Compare semantic vs hybrid side-by-side
python scripts/debug_retrieval.py "HIPAA regulations" --semantic-only

# Wider retrieval window
python scripts/debug_retrieval.py "Berlin AI funding" --top-k 10
```

The debug script prints chunk text, scores, source rank, and highlights what hybrid retrieval adds or drops compared to semantic-only.

---

## 🧠 What I Learned

### 1. Chunk size matters more than the LLM

Switching from `chunk_size=500, overlap=50` to `chunk_size=800, overlap=100` improved answer quality more than any prompt change. Small chunks lose the surrounding context that makes a sentence meaningful — "RLHF is used for fine-tuning" is ambiguous without the preceding paragraph explaining what fine-tuning is and why. 800 characters is roughly one focused paragraph: large enough to carry a complete thought, small enough to stay semantically precise for embedding.

The overlap is equally important. With only 50-char overlap, a sentence that straddles a chunk boundary appears truncated in both adjacent chunks. 100-char overlap ensures the last 1–2 sentences of each chunk repeat at the start of the next, so boundary concepts are never lost.

### 2. Hybrid search catches what semantic search misses

Pure semantic search with `all-MiniLM-L6-v2` struggles with:
- **Acronyms and technical terms**: "HIPAA", "RLHF", "BM25" — the embedding space doesn't reliably cluster these with their expansions unless both appeared frequently in training data
- **Exact numbers**: "$2.6 billion", "10-15 years" — semantically similar to many number expressions
- **Proper nouns**: "Helsing", "DeepL", "Aleph Alpha" — may not appear in the embedding model's training data

In the 10-question evaluation, 2 questions depended on exact keyword matching that semantic search ranked outside the top 5. BM25's term-frequency scoring recovered both.

The tricky part is combining the two scores. Raw score combination (0.7 × cosine + 0.3 × BM25) is biased toward BM25 because cosine similarity is bounded [0, 1] while BM25 is unbounded. **Reciprocal Rank Fusion** (`score = w₁/(k + rank₁) + w₂/(k + rank₂)`) uses rank position instead of raw score, making the two sources directly comparable regardless of scale.

### 3. Prompt strictness is the most underrated lever

The initial prompt in `configs/local.yaml` was vague:

> *"Always cite your sources by referencing the document name and chunk number. If the context doesn't contain enough information to answer, say so clearly. Do not make up information."*

The model routinely mixed document facts with things it "knew" from pre-training. Replacing this with an explicit, structured constraint:

> *"Answer using ONLY the information in the context below. For every claim, cite the source document and chunk number, e.g. [machine_learning_guide.txt, Chunk 3]. If the context does not contain enough information, say: 'I don't have enough information in the uploaded documents to answer this.'"*

reduced hallucination significantly. The key insight: **"don't make things up" is too abstract**. "Cite the specific chunk for every claim" forces the model to ground each sentence, which self-polices the answer far more effectively.

### 4. ChromaDB embedding function is an architectural constraint

ChromaDB persists metadata about the embedding function alongside each collection. If you create a collection with `sentence-transformer-A`, close the client, then reopen the collection with `sentence-transformer-B`, ChromaDB raises a conflict error. This meant:

1. The embedding function must be passed identically every time a collection is opened or recreated (including after `clear()`)
2. Both `IngestionPipeline` and `RAGChain` must use the same `EmbeddingEngine` instance — if each loads its own, they're technically the same model but different Python objects, which also caused conflicts

The fix was a shared `EmbeddingEngine` pattern: the entry point (`main.py`, `app.py`) builds one instance and passes its `get_embedding_function()` to both classes. This also halved startup memory usage by avoiding loading the 80 MB model twice.

### 5. Evaluation cost with local models

RAGAS uses an LLM internally to judge faithfulness and relevancy (the "LLM-as-judge" approach). With OpenAI GPT-4, this costs ~$0.05–0.10 per question. I wired RAGAS to use the same local Ollama model via `LangchainLLMWrapper`, which makes evaluation free but slow (llama3.1:8b takes ~3-5 seconds per metric per question). For 10 questions × 4 metrics, evaluation takes ~5 minutes locally versus ~15 seconds with GPT-4o-mini. Worth knowing before running evaluation in CI.

---

## ⚠️ Limitations

### Chunk boundary context loss

Even with 100-character overlap, information that spans two chunks can be split in a way that makes both chunks incomplete. A conclusion stated in paragraph 3 of a document that depends on evidence from paragraph 2 may land in different chunks. Neither chunk alone contains enough context to answer questions that require reasoning across both. Approaches that could help: semantic sentence-boundary splitting, parent-document retrieval (store small chunks for retrieval but pass their parent chunk to the LLM), or late chunking with token-level embeddings.

### General-purpose embeddings on domain-specific text

`all-MiniLM-L6-v2` is trained on general web text. It places "myocardial infarction" and "heart attack" far apart in embedding space unless co-occurrence in training data connected them. For medical, legal, or scientific documents, domain-fine-tuned models (`PubMedBERT`, `LegalBERT`, `CodeBERT`) would improve semantic recall substantially. The tradeoff is model size, latency, and the overhead of maintaining separate embeddings per domain.

### Local LLM quality ceiling

`llama3.1:8b` is the practical limit for hardware without a dedicated GPU. It follows citation instructions well but occasionally:
- Ignores the "ONLY use context" rule for facts it's confident about from pre-training
- Produces repetitive answers when retrieved context is thin
- Loses citation format consistency on multi-paragraph answers

GPT-4o produces noticeably better citation adherence and stays in-context more reliably. For production use with quality requirements, an API model with the local embedding stack is a reasonable hybrid.

### BM25 index is ephemeral

The BM25 index is rebuilt in memory on every server restart. For small document sets (< 5,000 chunks) this takes under a second, but for larger collections it becomes significant. The index is also lost if the process crashes mid-ingestion. A persistent inverted index (Elasticsearch, or serializing the `BM25Okapi` object to disk with `pickle`) would eliminate both issues.

### No cross-encoder reranking

The pipeline retrieves `top_k × 2` chunks then truncates after RRF scoring, but there is no second-stage reranking pass. A cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each (query, chunk) pair jointly — rather than encoding them independently — which improves precision by 10–15% at the cost of ~100ms extra latency per query. For a production system this tradeoff is usually worth it.

### RAGAS evaluation requires a capable LLM

RAGAS faithfulness and relevancy metrics internally call an LLM to judge quality. With a weak local model as the judge, the scores are less reliable than with GPT-4 as judge (the "weak judge" problem). The evaluation numbers above used llama3.1:8b as judge; the same answers scored ~5–8% higher with GPT-4o-mini as judge.

---

## Running Tests

```bash
pytest tests/ -v

# With coverage
pip install pytest-cov
pytest tests/ -v --cov=src --cov-report=term-missing
```

78 tests across chunker, retrieval, and RAG chain. All tests use mocks — no Ollama, ChromaDB, or model downloads needed.

---

## License

MIT — see [LICENSE](LICENSE)

## Author

**Krishna Singh** — MSc Artificial Intelligence @ IU Berlin
- GitHub: [@krishna8399](https://github.com/krishna8399)
- LinkedIn: [krishna839](https://linkedin.com/in/krishna839)
