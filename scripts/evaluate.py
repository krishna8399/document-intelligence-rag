"""
CLI runner for RAGAS evaluation.

Runs the 10 built-in ground-truth questions through the RAG pipeline,
computes RAGAS metrics, and saves results to JSON.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config configs/local.yaml
    python scripts/evaluate.py --output results/eval.json
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import RAGEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--config", default=os.environ.get("RAG_CONFIG", "configs/local.yaml"))
    parser.add_argument("--output", default="assets/evaluation_results.json")
    args = parser.parse_args()

    print(f"Config: {args.config}")
    print(f"Output: {args.output}")

    RAGEvaluator(config_path=args.config).run(output_path=args.output)


if __name__ == "__main__":
    main()
