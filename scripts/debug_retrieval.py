"""
Debug script: inspect what the retriever actually returns for a query.

Usage:
    python scripts/debug_retrieval.py "your question here"
    python scripts/debug_retrieval.py "your question here" --semantic-only
    python scripts/debug_retrieval.py "your question here" --top-k 10
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.retrieval.embedder import EmbeddingEngine
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.hybrid import HybridRetriever


def print_separator(label: str = ""):
    width = 70
    if label:
        print(f"\n{'─' * 3} {label} {'─' * (width - len(label) - 5)}")
    else:
        print("─" * width)


def debug_query(query: str, top_k: int = 7, semantic_only: bool = False):
    with open("configs/local.yaml") as f:
        config = yaml.safe_load(f)

    embedder = EmbeddingEngine(model_name=config["embedding"]["model"])
    ef = embedder.get_embedding_function()

    vs = VectorStoreManager(
        persist_dir=config["vector_store"]["persist_directory"],
        collection_name=config["vector_store"]["collection_name"],
        embedding_function=ef,
    )

    total = vs.collection.count()
    print(f"\nVector store: {total} chunks indexed")

    if total == 0:
        print("No documents ingested yet. Run: python scripts/ingest_docs.py")
        return

    hybrid_cfg = config["retrieval"].get("hybrid", {})
    retriever = HybridRetriever(
        vector_store=vs,
        semantic_weight=hybrid_cfg.get("semantic_weight", 0.7),
        bm25_weight=hybrid_cfg.get("bm25_weight", 0.3),
    )
    retriever.build_bm25_index()

    print_separator(f"QUERY: {query}")

    # --- Semantic-only ---
    print_separator("SEMANTIC-ONLY RESULTS")
    semantic_results = vs.search(query, top_k=top_k)
    for i, r in enumerate(semantic_results):
        print(f"\n[{i+1}] {r['source']}  chunk={r.get('metadata',{}).get('chunk_id','?')}  "
              f"score={r.get('score', 0):.4f}")
        print(f"     chars={len(r['text'])}")
        preview = r["text"].replace("\n", " ")[:200]
        print(f"     {preview}{'...' if len(r['text']) > 200 else ''}")

    if not semantic_only:
        # --- Hybrid ---
        print_separator("HYBRID RESULTS (semantic + BM25)")
        hybrid_results = retriever.search(query, top_k=top_k)
        for i, r in enumerate(hybrid_results):
            print(f"\n[{i+1}] {r['source']}  chunk={r.get('metadata',{}).get('chunk_id','?')}  "
                  f"hybrid={r.get('hybrid_score', 0):.6f}  "
                  f"sem={r.get('semantic_score', 0):.4f}  "
                  f"bm25={r.get('bm25_score', 0):.4f}")
            preview = r["text"].replace("\n", " ")[:200]
            print(f"     {preview}{'...' if len(r['text']) > 200 else ''}")

        # --- Diff: what hybrid adds/removes vs semantic ---
        sem_keys = {r.get("id", r["text"][:100]) for r in semantic_results}
        hyb_keys = {r.get("id", r["text"][:100]) for r in hybrid_results}
        added = hyb_keys - sem_keys
        dropped = sem_keys - hyb_keys
        if added:
            print(f"\n  + Hybrid added  {len(added)} chunk(s) not in semantic results")
        if dropped:
            print(f"  - Hybrid dropped {len(dropped)} chunk(s) from semantic results")

    # --- Chunk boundary inspection ---
    print_separator("CHUNK BOUNDARY CHECK")
    if semantic_results:
        top = semantic_results[0]
        print(f"\nTop chunk: {top['source']} chunk {top.get('metadata',{}).get('chunk_id','?')}")
        print(f"Full text ({len(top['text'])} chars):\n")
        print(top["text"])

    print_separator()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Question to debug")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--semantic-only", action="store_true")
    args = parser.parse_args()

    debug_query(args.query, top_k=args.top_k, semantic_only=args.semantic_only)
