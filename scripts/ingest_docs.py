"""
CLI script to ingest documents into the RAG system.

Usage:
    python scripts/ingest_docs.py --docs sample_docs/
    python scripts/ingest_docs.py --docs path/to/file.pdf
    python scripts/ingest_docs.py --docs sample_docs/ --config configs/local.yaml
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("--docs", type=str, required=True, help="File or directory path")
    default_config = os.environ.get("RAG_CONFIG", "configs/local.yaml")
    parser.add_argument("--config", type=str, default=default_config, help="Config file")
    args = parser.parse_args()

    pipeline = IngestionPipeline(args.config)

    path = Path(args.docs)
    if path.is_file():
        n = pipeline.ingest_file(str(path))
        print(f"\nDone: {n} chunks from {path.name}")
    elif path.is_dir():
        result = pipeline.ingest_directory(str(path))
        print(f"\nDone: {result['files']} files, {result['chunks']} total chunks")
    else:
        print(f"Path not found: {args.docs}")
        sys.exit(1)

    stats = pipeline.get_stats()
    print(f"Vector store total: {stats['total_chunks']} chunks")


if __name__ == "__main__":
    main()
