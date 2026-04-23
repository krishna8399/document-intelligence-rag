"""
RAGAS evaluation for the RAG pipeline.

RAGAS measures four things without needing human-labelled ground truth:
  - Faithfulness:        Does the answer contradict the retrieved context?
  - Answer Relevancy:    Does the answer address the question?
  - Context Recall:      Did we retrieve the chunks needed to answer?
  - Context Precision:   Are the retrieved chunks actually relevant?

Usage (from project root):
    python scripts/evaluate.py --config configs/local.yaml
"""

from dataclasses import dataclass, field
from typing import List

import yaml
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset


@dataclass
class EvalSample:
    """One question/answer/context triple for evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str = ""


@dataclass
class EvalResult:
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    num_samples: int
    raw: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"Evaluation Results ({self.num_samples} samples)\n"
            f"  Faithfulness:       {self.faithfulness:.3f}\n"
            f"  Answer Relevancy:   {self.answer_relevancy:.3f}\n"
            f"  Context Recall:     {self.context_recall:.3f}\n"
            f"  Context Precision:  {self.context_precision:.3f}\n"
        )


class RAGEvaluator:
    """Runs RAGAS evaluation against a RAGChain instance."""

    def __init__(self, config_path: str = "configs/local.yaml"):
        self.config_path = config_path
        self._rag_chain = None

    @property
    def rag_chain(self):
        if self._rag_chain is None:
            from src.generation.rag_chain import RAGChain
            self._rag_chain = RAGChain(self.config_path)
        return self._rag_chain

    def build_samples(self, questions: List[str], ground_truths: List[str] | None = None) -> List[EvalSample]:
        """Run questions through the RAG chain to collect answers and contexts."""
        ground_truths = ground_truths or [""] * len(questions)
        samples = []
        for question, gt in zip(questions, ground_truths):
            response = self.rag_chain.query(question)
            samples.append(EvalSample(
                question=question,
                answer=response.answer,
                contexts=[s["text"] for s in response.sources],
                ground_truth=gt,
            ))
        return samples

    def evaluate(self, samples: List[EvalSample]) -> EvalResult:
        """Score samples using RAGAS metrics."""
        dataset = Dataset.from_dict({
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples],
        })

        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        result = ragas_evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        return EvalResult(
            faithfulness=df["faithfulness"].mean(),
            answer_relevancy=df["answer_relevancy"].mean(),
            context_recall=df["context_recall"].mean(),
            context_precision=df["context_precision"].mean(),
            num_samples=len(samples),
            raw=result,
        )
