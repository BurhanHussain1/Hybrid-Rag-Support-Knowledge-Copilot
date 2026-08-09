"""The retrieval pipeline, assembled.

This is the single entry point the rest of the project uses. Generation (Step 4),
the API (Step 5), evaluation (Step 6) and the dashboard (Step 7) all call
`HybridRetriever.retrieve()` and never touch Qdrant or BM25 directly.

Four modes, deliberately selectable at call time:

    dense    vector search only
    sparse   BM25 only
    hybrid   both, fused with RRF
    rerank   both, fused, then cross-encoder reranked   <- the real pipeline

The modes exist because Step 6 has to compare them on the same questions. A
system that can only run its best configuration cannot prove that configuration
is best. The comparison toggle in the dashboard is the same switch.

Every call returns a RetrievalResult that keeps the intermediate lists, so you can
always see what each stage contributed rather than just the final five.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from copilot.config import settings
from copilot.retrieval.fusion import reciprocal_rank_fusion
from copilot.retrieval.models import RetrievedChunk
from copilot.retrieval.reranker import Reranker, get_reranker
from copilot.retrieval.retrievers import DenseRetriever, SparseRetriever

Mode = Literal["dense", "sparse", "hybrid", "rerank"]


class RetrievalResult(BaseModel):
    """Final chunks plus the full trace of how they were selected."""

    query: str
    mode: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)

    dense_candidates: list[RetrievedChunk] = Field(default_factory=list)
    sparse_candidates: list[RetrievedChunk] = Field(default_factory=list)
    fused_candidates: list[RetrievedChunk] = Field(default_factory=list)

    filters: dict[str, Any] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def top_score(self) -> float:
        """Best final score. Feeds the confidence calculation in Step 4.3."""
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    def coverage(self) -> dict[str, int]:
        """How many final chunks each retriever found. A one-line hybrid summary."""
        counts = {"dense_only": 0, "sparse_only": 0, "both": 0}
        for chunk in self.chunks:
            found = chunk.found_by()
            if len(found) == 2:
                counts["both"] += 1
            elif found == ["dense"]:
                counts["dense_only"] += 1
            elif found == ["sparse"]:
                counts["sparse_only"] += 1
        return counts


class HybridRetriever:
    """Dense + sparse + RRF + cross-encoder rerank."""

    def __init__(
        self,
        *,
        dense: DenseRetriever | None = None,
        sparse: SparseRetriever | None = None,
        reranker: Reranker | None = None,
        strategy: str | None = None,
    ):
        self.strategy = strategy or settings.chunk_strategy
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever(strategy=self.strategy)
        # Not created eagerly: loading the cross-encoder costs seconds, and the
        # dense-only mode never needs it.
        self._reranker = reranker

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        *,
        mode: Mode = "rerank",
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.final_top_k
        result = RetrievalResult(query=query, mode=mode, filters=filters)

        # -- single-retriever modes ---------------------------------------
        if mode == "dense":
            start = time.perf_counter()
            result.dense_candidates = self.dense.search(query, settings.dense_top_k, filters)
            result.timings_ms["dense"] = (time.perf_counter() - start) * 1000
            result.chunks = result.dense_candidates[:top_k]
            return result

        if mode == "sparse":
            start = time.perf_counter()
            result.sparse_candidates = self.sparse.search(query, settings.sparse_top_k, filters)
            result.timings_ms["sparse"] = (time.perf_counter() - start) * 1000
            result.chunks = result.sparse_candidates[:top_k]
            return result

        # -- both retrievers, then fuse -----------------------------------
        start = time.perf_counter()
        result.dense_candidates = self.dense.search(query, settings.dense_top_k, filters)
        result.timings_ms["dense"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        result.sparse_candidates = self.sparse.search(query, settings.sparse_top_k, filters)
        result.timings_ms["sparse"] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        result.fused_candidates = reciprocal_rank_fusion(
            [result.dense_candidates, result.sparse_candidates],
            weights=[settings.dense_weight, settings.sparse_weight],
            k=settings.rrf_k,
            limit=settings.rerank_top_n,
        )
        result.timings_ms["fusion"] = (time.perf_counter() - start) * 1000

        if mode == "hybrid":
            result.chunks = result.fused_candidates[:top_k]
            return result

        # -- rerank the fused candidates ----------------------------------
        start = time.perf_counter()
        result.chunks = self.reranker.rerank(query, result.fused_candidates, limit=top_k)
        result.timings_ms["rerank"] = (time.perf_counter() - start) * 1000
        return result


_default: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """Shared instance, so models and the BM25 index load once per process."""
    global _default
    if _default is None:
        _default = HybridRetriever()
    return _default
