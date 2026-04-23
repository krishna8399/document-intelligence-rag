"""
Streamlit chat interface for the RAG system.

Features:
- Document upload sidebar
- Chat interface with conversation memory
- Source citations displayed below each answer
- System stats in sidebar
"""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.embedder import EmbeddingEngine
from src.ingestion.pipeline import IngestionPipeline
from src.generation.rag_chain import RAGChain


st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Intelligence — Ask Your Documents")
st.markdown("Upload documents and ask questions. Answers are grounded in your content with source citations.")


@st.cache_resource
def load_system():
    import yaml
    config = os.environ.get("RAG_CONFIG", "configs/local.yaml")
    with open(config) as f:
        cfg = yaml.safe_load(f)
    embedder = EmbeddingEngine(model_name=cfg["embedding"]["model"])
    ef = embedder.get_embedding_function()
    return IngestionPipeline(config, embedding_function=ef), RAGChain(config, embedding_function=ef)


ingestion, rag = load_system()

# --- Sidebar: Upload + Stats ---
st.sidebar.header("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    if st.sidebar.button("Ingest Documents"):
        with st.sidebar.status("Processing..."):
            for file in uploaded_files:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(file.name).suffix,
                )
                tmp.write(file.read())
                tmp.close()

                n = ingestion.ingest_file(tmp.name)
                st.sidebar.success(f"{file.name}: {n} chunks")
                os.unlink(tmp.name)

            rag.retriever.build_bm25_index()
            st.sidebar.success("All documents ingested!")

# Stats
stats = rag.get_stats()
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Documents:** {stats['total_chunks']} chunks indexed")
st.sidebar.markdown(f"**LLM:** {stats['llm_model']}")
st.sidebar.markdown(f"**Embeddings:** {stats['embedding_model']}")

if st.sidebar.button("Clear All Documents"):
    rag.vector_store.clear()
    rag.retriever.reset_bm25()
    rag.clear_history()
    st.sidebar.success("Cleared!")
    st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['source']}** (chunk {src['chunk_id']}, "
                        f"score: {src['score']:.3f})"
                    )
                    st.caption(src["text"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if stats["total_chunks"] == 0:
        st.warning("Please upload documents first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                response = rag.query(prompt)

            st.markdown(response.answer)

            if response.sources:
                with st.expander("📚 Sources"):
                    for src in response.sources:
                        st.markdown(
                            f"**{src['source']}** (chunk {src['chunk_id']}, "
                            f"score: {src['score']:.3f})"
                        )
                        st.caption(src["text"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        })

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Built by [Krishna Singh](https://github.com/krishna8399)**\n\n"
    "MSc AI @ IU Berlin"
)
