"""Running the evaluation.

Two passes, split by cost, because they answer different questions and one of
them is free:

  retrieval pass   no LLM calls at all. Runs every question through every mode
                   and asks "did the right document come back?". Free, fast
                   (~2 minutes for 60 questions x 4 modes), and it produces the
                   headline dense-vs-hybrid number.

  answer pass      the full pipeline including generation and citation
                   verification. Costs roughly $0.05 per mode for 60 questions.
                   This is what tells you whether refusals work and whether
                   citations hold up.

Keeping them separate means you can iterate on chunking and fusion weights all
day for free, and only spend money when you want to check end-to-end behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from copilot.config import settings
from copilot.evaluation.metrics import (
    AnswerMetrics,
    QuestionAnswer,
    QuestionRetrieval,
    RetrievalMetrics,
    count_mentions,
    score_retrieval,
)
from copilot.evaluation.models import GoldenQuestion, GoldenSet


@dataclass
class ModeResult:
    """Everything measured for one retrieval mode."""

    mode: str
    retrieval: RetrievalMetrics
    per_question: list[QuestionRetrieval] = field(default_factory=list)
    seconds: float = 0.0
    mean_latency_ms: float = 0.0


@dataclass
class EvalRun:
    """A complete evaluation, with the configuration that produced it."""

    modes: dict[str, ModeResult] = field(default_factory=dict)
    answers: AnswerMetrics | None = None
    per_answer: list[QuestionAnswer] = field(default_factory=list)
    answer_mode: str = ""

    questions_total: int = 0
    questions_scored: int = 0
    questions_unverified: int = 0

    config: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    started_at: str = ""
    seconds: float = 0.0


def snapshot_config() -> dict:
    """Record every setting that could change the numbers.

    Without this, a report is a number with no provenance. Six weeks later,
    "hybrid scored 88%" is unreproducible unless you know it was 800-character
    heading chunks with rrf_k=60 and top_k=5.
    """
    return {
        "chunk_strategy": settings.chunk_strategy,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "min_chunk_chars": settings.min_chunk_chars,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "llm_model": settings.llm_model,
        "judge_model": settings.judge_model or f"{settings.llm_model} (same as generator)",
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rrf_k": settings.rrf_k,
        "dense_weight": settings.dense_weight,
        "sparse_weight": settings.sparse_weight,
        "rerank_top_n": settings.rerank_top_n,
        "final_top_k": settings.final_top_k,
        "min_confidence": settings.min_confidence,
    }


def run_retrieval_eval(
    golden: GoldenSet,
    modes: list[str],
    *,
    top_k: int | None = None,
    questions: list[GoldenQuestion] | None = None,
    quiet: bool = False,
) -> dict[str, ModeResult]:
    """Score retrieval for each mode. No LLM calls."""
    from copilot.retrieval.hybrid import HybridRetriever

    top_k = top_k or settings.final_top_k
    pool = questions if questions is not None else golden.scorable()
    # Refusal questions have no expected documents, so retrieval accuracy is
    # undefined for them - they are scored in the answer pass instead.
    answerable = [q for q in pool if q.expected_doc_ids]

    retriever = HybridRetriever()
    results: dict[str, ModeResult] = {}

    for mode in modes:
        started = time.time()
        latencies: list[float] = []
        per_question: list[QuestionRetrieval] = []

        if not quiet:
            print(f"  {mode:<8} ", end="", flush=True)

        for question in answerable:
            result = retriever.retrieve(question.question, mode=mode, top_k=top_k)
            latencies.append(sum(result.timings_ms.values()))
            per_question.append(score_retrieval(question, [c.doc_id for c in result.chunks]))

        metrics = RetrievalMetrics.compute(per_question)
        elapsed = time.time() - started

        results[mode] = ModeResult(
            mode=mode,
            retrieval=metrics,
            per_question=per_question,
            seconds=elapsed,
            mean_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        )

        if not quiet:
            print(f"hit@{top_k} {metrics.hit_rate:.1%}   MRR {metrics.mrr:.3f}   "
                  f"{elapsed:.0f}s")

    return results


def run_answer_eval(
    golden: GoldenSet,
    mode: str,
    *,
    top_k: int | None = None,
    questions: list[GoldenQuestion] | None = None,
    verify: bool = True,
    quiet: bool = False,
) -> tuple[AnswerMetrics, list[QuestionAnswer], dict]:
    """Run the full pipeline on every question. Costs money."""
    from copilot.generation.pipeline import Copilot

    pool = questions if questions is not None else golden.scorable()
    copilot = Copilot()
    per_answer: list[QuestionAnswer] = []

    for i, question in enumerate(pool, start=1):
        if not quiet:
            print(f"  [{i:>3}/{len(pool)}] {question.id} {question.question[:56]:<58}", end="", flush=True)

        try:
            response = copilot.ask(question.question, mode=mode, top_k=top_k, verify=verify)
        except Exception as exc:  # noqa: BLE001 - one bad question must not kill the run
            per_answer.append(
                QuestionAnswer(
                    question_id=question.id,
                    category=str(question.category),
                    should_refuse=question.should_refuse,
                    refused=False,
                    confidence=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            if not quiet:
                print(" ERROR")
            continue

        supported = sum(1 for c in response.citations if c.supported is True)
        unsupported = sum(1 for c in response.citations if c.supported is False)

        per_answer.append(
            QuestionAnswer(
                question_id=question.id,
                category=str(question.category),
                should_refuse=question.should_refuse,
                refused=not response.answered,
                confidence=response.confidence,
                citations_total=len(response.citations),
                citations_supported=supported,
                citations_unsupported=unsupported,
                mentions_expected=count_mentions(response.answer, question.answer_must_mention),
                mentions_required=len(question.answer_must_mention),
                mentions_forbidden_found=count_mentions(
                    response.answer, question.answer_must_not_mention
                ),
                answer_text=response.answer,
            )
        )

        if not quiet:
            state = "refused " if not response.answered else "answered"
            ok = "ok " if per_answer[-1].refusal_correct else "BAD"
            print(f" {state} conf {response.confidence:.2f} {ok}")

    return AnswerMetrics.compute(per_answer), per_answer, copilot.answerer.llm.usage_summary()
