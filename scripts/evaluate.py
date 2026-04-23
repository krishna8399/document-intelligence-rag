"""
CLI script to run RAGAS evaluation on the RAG system.

Usage:
    python scripts/evaluate.py --config configs/local.yaml
    python scripts/evaluate.py --config configs/local.yaml --questions questions.txt
    python scripts/evaluate.py --config configs/local.yaml --output results.json

questions.txt format — one question per line:
    What is the main topic of the document?
    What are the key findings?
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluate import RAGEvaluator

DEFAULT_QUESTIONS = [
    "What is the main topic of the uploaded documents?",
    "What are the key findings or conclusions?",
    "What recommendations are made?",
    "Who are the intended audiences for this document?",
    "What data or evidence supports the main claims?",
]


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG system")
    parser.add_argument("--config", type=str,
                        default=os.environ.get("RAG_CONFIG", "configs/local.yaml"))
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to a text file with one question per line")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results as JSON")
    args = parser.parse_args()

    # Load questions
    if args.questions:
        questions = Path(args.questions).read_text().strip().splitlines()
        questions = [q.strip() for q in questions if q.strip()]
    else:
        print("No questions file provided — using default evaluation questions.")
        questions = DEFAULT_QUESTIONS

    print(f"\nEvaluating {len(questions)} questions with config: {args.config}")
    print("=" * 60)

    evaluator = RAGEvaluator(args.config)

    print("\nRunning questions through RAG chain...")
    samples = evaluator.build_samples(questions)

    print("\nScoring with RAGAS metrics...")
    result = evaluator.evaluate(samples)

    print(f"\n{result}")

    if args.output:
        output = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_recall": result.context_recall,
            "context_precision": result.context_precision,
            "num_samples": result.num_samples,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
