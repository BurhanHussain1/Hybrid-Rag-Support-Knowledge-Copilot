#!/usr/bin/env python
"""Sweep the RRF k constant against the golden set.

    python scripts/sweep_rrf_k.py

In Step 3 a single query suggested rrf_k=60 was too flat: a chunk both retrievers
rated mediocre was beating one dense rated best. I deliberately did not change the
default then, because tuning on one example is overfitting to an anecdote.

Now there is a question set, so the same change can be made with evidence. This
sweep makes no LLM calls, so it is free to run as often as you like - which is the
main reason retrieval evaluation was kept separate from answer evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.config import settings  # noqa: E402
from copilot.evaluation.golden import load_golden  # noqa: E402
from copilot.evaluation.metrics import RetrievalMetrics, score_retrieval  # noqa: E402

K_VALUES = [1, 5, 10, 20, 40, 60, 100, 200]


def main() -> int:
    from copilot.retrieval.fusion import reciprocal_rank_fusion
    from copilot.retrieval.hybrid import HybridRetriever
    from copilot.retrieval.reranker import get_reranker

    golden = load_golden()
    questions = [q for q in golden.questions if q.expected_doc_ids and str(q.status) != "rejected"]
    print(f"{len(questions)} answerable questions\n")

    retriever = HybridRetriever()
    reranker = get_reranker()

    # Retrieve once per question and reuse the candidate lists for every k.
    # Fusion is pure arithmetic over ranks, so re-running the retrievers for each
    # k would repeat identical work - about eight times the runtime for the same
    # numbers.
    print("retrieving candidates once...")
    cached = []
    for question in questions:
        dense = retriever.dense.search(question.question, settings.dense_top_k)
        sparse = retriever.sparse.search(question.question, settings.sparse_top_k)
        cached.append((question, dense, sparse))
    print("done\n")

    print(f"{'k':>5}  {'hybrid hit@5':>13}  {'hybrid MRR':>11}  {'+rerank hit@5':>14}  {'+rerank MRR':>12}")
    print("-" * 66)

    best = None
    for k in K_VALUES:
        fused_results = []
        rerank_results = []

        for question, dense, sparse in cached:
            fused = reciprocal_rank_fusion(
                [dense, sparse],
                weights=[settings.dense_weight, settings.sparse_weight],
                k=k,
                limit=settings.rerank_top_n,
            )
            fused_results.append(
                score_retrieval(question, [c.doc_id for c in fused[: settings.final_top_k]])
            )

            reranked = reranker.rerank(question.question, fused, limit=settings.final_top_k)
            rerank_results.append(score_retrieval(question, [c.doc_id for c in reranked]))

        fused_metrics = RetrievalMetrics.compute(fused_results)
        rerank_metrics = RetrievalMetrics.compute(rerank_results)

        marker = ""
        if best is None or rerank_metrics.hit_rate > best[1]:
            best = (k, rerank_metrics.hit_rate)
            marker = "  <-- best so far"

        print(f"{k:>5}  {fused_metrics.hit_rate:>12.1%}  {fused_metrics.mrr:>11.3f}  "
              f"{rerank_metrics.hit_rate:>13.1%}  {rerank_metrics.mrr:>12.3f}{marker}")

    print(f"\ncurrent setting: rrf_k={settings.rrf_k}")
    print(f"best measured  : rrf_k={best[0]} at {best[1]:.1%} hit@5 after reranking")
    print("\nChange RRF_K in .env to adopt it, then re-run eval.py to confirm on the full metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
