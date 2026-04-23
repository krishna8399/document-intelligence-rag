# 📄 RAG-Powered Document Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange.svg)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Retrieval-Augmented Generation (RAG) system that lets users upload documents (PDF, DOCX, TXT), ask natural language questions, and get accurate answers grounded in the uploaded content — with source citations. Built with LangChain, ChromaDB, sentence-transformers, and served through a FastAPI backend with a Streamlit chat interface.

<!-- TODO: Add demo GIF -->
<!-- ![Demo](assets/demo.gif) -->

## 🏗️ Architecture

```
User uploads documents (PDF / DOCX / TXT)
       │
       ▼
┌─────────────────────┐
│  Document Ingestion  │  Parse → Clean → Chunk
│  (ingestion/)        │  Recursive text splitting
└──────────┬──────────┘
           │ text chunks (500 tokens, 50 overlap)
           ▼
┌─────────────────────┐
│  Embedding Engine    │  sentence-transformers
│                      │  all-MiniLM-L6-v2 (384-dim)
└──────────┬──────────┘
           │ vectors
           ▼
┌─────────────────────┐
│  Vector Store        │  ChromaDB (persistent)
│  (retrieval/)        │  Metadata filtering
└──────────┬──────────┘
           │
    User asks a question
           │
           ▼
┌─────────────────────┐
│  Retrieval Pipeline  │  Semantic search + BM25
│                      │  Hybrid scoring + re-ranking
└──────────┬──────────┘
           │ top-k relevant chunks + sources
           ▼
┌─────────────────────┐
│  Generation (LLM)    │  Context + question → answer
│                      │  Source citation tracking
└──────────┬──────────┘
           │ answer + citations
           ▼
┌─────────────────────┐
│  FastAPI + Streamlit │  Chat UI with source display
│  (api/ + app/)       │  Conversation memory
└─────────────────────┘
```

## 📊 Evaluation

| Metric | Score |
|--------|-------|
| Faithfulness (RAGAS) | - |
| Answer Relevancy | - |
| Context Recall | - |
| Context Precision | - |

<!-- TODO: Fill after evaluation -->

## 🚀 Quick Start

```bash
git clone https://github.com/krishna8399/document-intelligence-rag.git
cd document-intelligence-rag

conda create -n rag-system python=3.10 -y
conda activate rag-system
pip install -r requirements.txt

# Set up API key (choose one)
export OPENAI_API_KEY="your-key"        # for OpenAI
# OR use local models (no API key needed — see configs/local.yaml)

# Ingest sample documents
python scripts/ingest_docs.py --docs sample_docs/

# Start API
uvicorn src.api.main:app --reload --port 8000

# Start chat UI (separate terminal)
streamlit run src/app/app.py
```

## 📁 Project Structure

```
document-intelligence-rag/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── configs/
│   ├── default.yaml             # Default config (OpenAI)
│   └── local.yaml               # Local model config (no API key)
├── src/
│   ├── ingestion/
│   │   ├── parser.py            # PDF, DOCX, TXT parsing
│   │   ├── chunker.py           # Text chunking strategies
│   │   └── pipeline.py          # Full ingestion pipeline
│   ├── retrieval/
│   │   ├── embedder.py          # Embedding model wrapper
│   │   ├── vector_store.py      # ChromaDB operations
│   │   ├── bm25.py              # BM25 keyword search
│   │   └── hybrid.py            # Hybrid search (semantic + BM25)
│   ├── generation/
│   │   ├── llm.py               # LLM wrapper (OpenAI / local)
│   │   ├── prompt.py            # Prompt templates
│   │   └── rag_chain.py         # Full RAG pipeline
│   ├── evaluation/
│   │   └── evaluate.py          # RAGAS evaluation
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── schemas.py           # Pydantic models
│   └── app/
│       └── app.py               # Streamlit chat interface
├── scripts/
│   ├── ingest_docs.py           # CLI document ingestion
│   └── evaluate.py              # Run RAGAS evaluation
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── test_rag_chain.py
├── sample_docs/                 # Example documents for testing
│   ├── sample_report.pdf
│   └── sample_article.txt
└── assets/
```

## 🔧 Tech Stack

- **LLM**: OpenAI GPT-4o-mini / local models via Ollama
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB (persistent, local)
- **Keyword Search**: BM25 (rank-bm25)
- **Framework**: LangChain
- **API**: FastAPI, Pydantic
- **UI**: Streamlit
- **Evaluation**: RAGAS
- **Deployment**: Docker, docker-compose

## 🧠 What I Learned

-
-
-

## 📄 License

MIT License

## 👤 Author

**Krishna Singh** — MSc Artificial Intelligence @ IU Berlin
- GitHub: [@krishna8399](https://github.com/krishna8399)
- LinkedIn: [krishna839](https://linkedin.com/in/krishna839)
