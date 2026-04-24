"""
RAGAS evaluation for the RAG pipeline.

Defines 10 ground-truth Q&A pairs across the three sample documents, runs
them through the RAG pipeline, computes RAGAS metrics (faithfulness,
answer_relevancy, context_recall, context_precision), and measures source
retrieval hit-rate (correct document in top-5 chunks).

Usage:
    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --config configs/local.yaml --output assets/evaluation_results.json
"""

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import yaml
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


# ---------------------------------------------------------------------------
# Ground-truth evaluation set — 10 questions across 3 sample documents
# ---------------------------------------------------------------------------

EVAL_SAMPLES = [
    # ── machine_learning_guide.txt (4) ──────────────────────────────────────
    {
        "question": "Which paper introduced the Transformer architecture and in what year?",
        "ground_truth": (
            "The Transformer architecture was introduced in the 2017 paper "
            "'Attention Is All You Need' by Vaswani et al. At its core is the "
            "self-attention mechanism, which allows each position in a sequence "
            "to attend to all other positions, capturing long-range dependencies."
        ),
        "expected_source": "machine_learning_guide.txt",
    },
    {
        "question": "What landmark results has reinforcement learning achieved?",
        "ground_truth": (
            "Reinforcement learning achieved landmark results including AlphaGo "
            "(defeating the world Go champion), AlphaFold (predicting protein "
            "structures), and training large language models using Reinforcement "
            "Learning from Human Feedback (RLHF)."
        ),
        "expected_source": "machine_learning_guide.txt",
    },
    {
        "question": "What are the three gate types in an LSTM and what problem do they solve?",
        "ground_truth": (
            "LSTMs have input, forget, and output gates that control the flow of "
            "information through memory cells. They address the vanishing gradient "
            "problem that prevents vanilla RNNs from learning long-range dependencies "
            "in sequential data."
        ),
        "expected_source": "machine_learning_guide.txt",
    },
    {
        "question": "Which evaluation metrics should be used for imbalanced classification?",
        "ground_truth": (
            "For imbalanced classification problems, F1 or AUC-ROC metrics should "
            "be used rather than accuracy, which can be misleading when class "
            "distributions are skewed."
        ),
        "expected_source": "machine_learning_guide.txt",
    },
    # ── berlin_startup_report.txt (3) ────────────────────────────────────────
    {
        "question": "How much did Helsing raise in 2023 and what does the company do?",
        "ground_truth": (
            "Helsing, an AI defense technology company, raised over 200 million euros "
            "in 2023 and is developing AI systems for European defense applications."
        ),
        "expected_source": "berlin_startup_report.txt",
    },
    {
        "question": "What was the total venture investment in Berlin in 2023 compared to the 2021 peak?",
        "ground_truth": (
            "Total venture investment in Berlin-based companies reached approximately "
            "4.2 billion euros in 2023, a significant decline from the peak of "
            "9.8 billion euros in 2021."
        ),
        "expected_source": "berlin_startup_report.txt",
    },
    {
        "question": "Which three universities are part of the Einstein Center Digital Future?",
        "ground_truth": (
            "The Einstein Center Digital Future funds research across Berlin's three "
            "major universities: Freie Universität Berlin, Humboldt-Universität zu "
            "Berlin, and Technische Universität Berlin."
        ),
        "expected_source": "berlin_startup_report.txt",
    },
    # ── ai_healthcare_report.txt (3) ─────────────────────────────────────────
    {
        "question": "How long does traditional drug discovery take and what does it cost per approved drug?",
        "ground_truth": (
            "Traditional drug discovery takes an average of 10 to 15 years and costs "
            "approximately 2.6 billion dollars per approved drug."
        ),
        "expected_source": "ai_healthcare_report.txt",
    },
    {
        "question": "What is federated learning and why is it used in healthcare AI?",
        "ground_truth": (
            "Federated learning is an approach to train AI models across multiple "
            "institutions without sharing patient data. It is used in healthcare AI "
            "to comply with strict privacy regulations such as HIPAA in the United "
            "States and GDPR in Europe."
        ),
        "expected_source": "ai_healthcare_report.txt",
    },
    {
        "question": "Which medical conditions can AI imaging detect as well as expert radiologists?",
        "ground_truth": (
            "AI-powered diagnostic imaging using convolutional neural networks (CNNs) "
            "has demonstrated performance comparable to or exceeding expert radiologists "
            "in detecting diabetic retinopathy, skin cancer, and lung nodules."
        ),
        "expected_source": "ai_healthcare_report.txt",
    },
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    question: str
    ground_truth: str
    expected_source: str
    answer: str = ""
    contexts: List[str] = field(default_factory=list)
    source_hit: bool = False
    source_rank: Optional[int] = None   # 1-based rank; None = not found in top-k


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    source_hit_rate: float
    mean_source_rank: float
    num_samples: int
    per_sample: List[dict] = field(default_factory=list)
    raw_scores: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary_table(self) -> str:
        rows = [
            ("Faithfulness",      self.faithfulness,      True),
            ("Answer Relevancy",  self.answer_relevancy,  True),
            ("Context Recall",    self.context_recall,    True),
            ("Context Precision", self.context_precision, True),
            ("Source Hit Rate",   self.source_hit_rate,   True),
            ("Mean Source Rank",  self.mean_source_rank,  False),
        ]
        sep = "=" * 54
        lines = [
            f"\n{sep}",
            f"  RAGAS Evaluation Results  ({self.num_samples} questions)",
            sep,
        ]
        for name, val, is_ratio in rows:
            if math.isnan(val):
                lines.append(f"  {name:<24}   N/A")
                continue
            if is_ratio:
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                lines.append(f"  {name:<24} {val:.3f}  [{bar}]")
            else:
                lines.append(f"  {name:<24} {val:.2f}  (lower = retrieved sooner)")
        lines.append(sep)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:

    def __init__(self, config_path: str = "configs/local.yaml"):
        self.config_path = config_path
        self._rag_chain = None

    @property
    def rag_chain(self):
        if self._rag_chain is None:
            from src.generation.rag_chain import RAGChain
            self._rag_chain = RAGChain(self.config_path)
        return self._rag_chain

    def build_samples(self) -> List[EvalSample]:
        """Run every evaluation question through the RAG pipeline."""
        total = len(EVAL_SAMPLES)
        samples = []

        for i, item in enumerate(EVAL_SAMPLES, 1):
            print(f"  [{i:>2}/{total}] {item['question'][:68]}...")

            response = self.rag_chain.query(
                item["question"],
                use_history=False,   # isolate each question
            )

            # Check whether the expected source document appears in retrieved chunks
            retrieved_sources = [s["source"] for s in response.sources]
            expected = item["expected_source"]

            rank = next(
                (r + 1 for r, src in enumerate(retrieved_sources) if expected in src),
                None,
            )

            samples.append(EvalSample(
                question=item["question"],
                ground_truth=item["ground_truth"],
                expected_source=expected,
                answer=response.answer,
                contexts=[s["text"] for s in response.sources],
                source_hit=rank is not None,
                source_rank=rank,
            ))

        return samples

    def evaluate(self, samples: List[EvalSample]) -> EvalResult:
        """Score samples with RAGAS and compute source retrieval hit-rate."""
        print("\nRunning RAGAS metrics (uses the configured LLM for scoring)...")

        dataset = Dataset.from_dict({
            "question":     [s.question for s in samples],
            "answer":       [s.answer for s in samples],
            "contexts":     [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        })

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        # Optionally wire RAGAS to the same Ollama LLM so no OpenAI key is needed
        ragas_kwargs = self._ragas_llm_kwargs()
        result = ragas_evaluate(dataset, metrics=metrics, **ragas_kwargs)
        df = result.to_pandas()

        per_sample = [
            {
                "question":          s.question,
                "expected_source":   s.expected_source,
                "source_hit":        s.source_hit,
                "source_rank":       s.source_rank,
                "faithfulness":      _safe_float(df["faithfulness"].iloc[i]),
                "answer_relevancy":  _safe_float(df["answer_relevancy"].iloc[i]),
                "context_recall":    _safe_float(df["context_recall"].iloc[i]),
                "context_precision": _safe_float(df["context_precision"].iloc[i]),
                "answer_snippet":    s.answer[:300],
            }
            for i, s in enumerate(samples)
        ]

        hits = [s for s in samples if s.source_hit]
        ranked = [s.source_rank for s in samples if s.source_rank is not None]

        return EvalResult(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_recall=float(df["context_recall"].mean()),
            context_precision=float(df["context_precision"].mean()),
            source_hit_rate=len(hits) / len(samples),
            mean_source_rank=sum(ranked) / len(ranked) if ranked else float("nan"),
            num_samples=len(samples),
            per_sample=per_sample,
            raw_scores={
                "faithfulness":      df["faithfulness"].tolist(),
                "answer_relevancy":  df["answer_relevancy"].tolist(),
                "context_recall":    df["context_recall"].tolist(),
                "context_precision": df["context_precision"].tolist(),
            },
        )

    def run(self, output_path: str = "assets/evaluation_results.json") -> EvalResult:
        """Full pipeline: build → evaluate → print table → save JSON."""
        print(f"\nPreparing {len(EVAL_SAMPLES)} evaluation questions...")
        samples = self.build_samples()
        result = self.evaluate(samples)

        print(result.summary_table())
        self._print_retrieval_table(samples)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "faithfulness":      round(result.faithfulness, 4),
                "answer_relevancy":  round(result.answer_relevancy, 4),
                "context_recall":    round(result.context_recall, 4),
                "context_precision": round(result.context_precision, 4),
                "source_hit_rate":   round(result.source_hit_rate, 4),
                "mean_source_rank":  None if math.isnan(result.mean_source_rank)
                                     else round(result.mean_source_rank, 4),
                "num_samples":       result.num_samples,
                "timestamp":         result.timestamp,
            },
            "per_sample": result.per_sample,
            "raw_scores": result.raw_scores,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        print(f"\nFull results saved → {output_path}")
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _ragas_llm_kwargs(self) -> dict:
        """
        If the project is configured to use Ollama, wire RAGAS to the same
        local model so evaluation works without an OpenAI key.
        Falls back to RAGAS defaults (OpenAI) if wiring fails.
        """
        try:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg["llm"]["provider"] != "ollama":
                return {}

            from langchain_community.chat_models import ChatOllama
            from ragas.llms import LangchainLLMWrapper

            base_url = cfg["llm"].get("base_url", "http://localhost:11434")
            model = cfg["llm"]["model"]
            ollama_llm = ChatOllama(base_url=base_url, model=model)
            print(f"  RAGAS LLM: Ollama {model} at {base_url}")
            return {"llm": LangchainLLMWrapper(ollama_llm)}
        except Exception:
            print("  RAGAS LLM: falling back to default (set OPENAI_API_KEY if needed)")
            return {}

    def _print_retrieval_table(self, samples: List[EvalSample]):
        hits = sum(1 for s in samples if s.source_hit)
        sep = "─" * 54
        print(f"\n{sep}")
        print(f"  Retrieval Quality  ({hits}/{len(samples)} correct source in top-5)")
        print(sep)
        for s in samples:
            if s.source_hit:
                status = f"✓ rank {s.source_rank}"
            else:
                status = "✗ MISS"
            q = s.question if len(s.question) <= 52 else s.question[:49] + "..."
            print(f"  [{status:>8}]  {q}")
        print(sep)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if math.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--config", default="configs/local.yaml", help="Config YAML path")
    parser.add_argument("--output", default="assets/evaluation_results.json", help="Output JSON path")
    args = parser.parse_args()

    evaluator = RAGEvaluator(config_path=args.config)
    evaluator.run(output_path=args.output)
