"""Reciprocal Rank Fusion: merging two ranked lists into one.

The problem it solves is a scale problem. Our two retrievers produce numbers that
have nothing to do with each other:

    dense  (cosine)  0.884, 0.803, 0.787, ...    bounded 0..1
    sparse (BM25)   39.722, 36.219, 33.535, ...  unbounded, corpus-dependent

You cannot add those. The obvious fix - normalise each list to 0..1 and take a
weighted sum - looks reasonable and behaves badly. Min-max normalisation depends
on the extremes of *this particular result list*, so the same chunk gets a
different normalised score depending on what else happened to be retrieved. A
query where BM25 finds one strong match and nineteen weak ones produces a wildly
different scale from a query where all twenty are mediocre. Your fusion weights
then mean something different for every query, which makes them impossible to
tune honestly.

RRF sidesteps all of it by throwing the scores away and using only **rank**:

    score(chunk) = SUM over retrievers of  weight / (k + rank)

Rank is scale-free. Being 1st means the same thing in both lists. A chunk ranked
1st by one retriever and 40th by the other still scores well; a chunk ranked
mid-table by both does not. And because 1/(k+rank) falls off smoothly, no single
retriever can dominate purely by being confident.

`k` (default 60, from the original RRF paper) controls how sharply rank matters:

    k=1    1st place is worth 30x more than 30th  - top-heavy, trusts rank hard
    k=60   1st place is worth ~1.5x more than 30th - gentle, values agreement
    k=1000 nearly flat - almost pure vote-counting

Small k trusts each retriever's ordering. Large k rewards chunks that *both*
retrievers liked, even if neither ranked them first. 60 is a sane default and is
exposed in .env so Step 6 can measure whether it is the right one for this corpus.
"""

from __future__ import annotations

from copilot.config import settings
from copilot.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    weights: list[float] | None = None,
    k: int | None = None,
    limit: int | None = None,
) -> list[RetrievedChunk]:
    """Merge ranked lists into one list, ordered by fused score.

    Args:
        result_lists: one ranked list per retriever, best first.
        weights: how much to trust each list. Defaults to 1.0 each.
        k: RRF smoothing constant. Higher = rank matters less.
        limit: how many results to return.

    Returns:
        A new list of RetrievedChunk with `fused_score` set and per-retriever
        scores and ranks preserved, so you can always see where a chunk came from.
    """
    k = k if k is not None else settings.rrf_k
    limit = limit or settings.rerank_top_n
    weights = weights or [1.0] * len(result_lists)

    if len(weights) != len(result_lists):
        raise ValueError(f"got {len(result_lists)} result lists but {len(weights)} weights")

    fused_scores: dict[str, float] = {}
    merged: dict[str, RetrievedChunk] = {}

    for results, weight in zip(result_lists, weights, strict=True):
        for hit in results:
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + weight / (k + hit.rank)

            if hit.chunk_id not in merged:
                # model_copy so we never mutate the caller's objects - the raw
                # dense and sparse lists stay intact for the dashboard to show
                # side by side.
                merged[hit.chunk_id] = hit.model_copy(deep=True)
            else:
                # Same chunk found by both retrievers: fold in the scores and
                # ranks from this list so the merged object carries both.
                existing = merged[hit.chunk_id]
                if hit.dense_rank is not None:
                    existing.dense_rank = hit.dense_rank
                    existing.dense_score = hit.dense_score
                if hit.sparse_rank is not None:
                    existing.sparse_rank = hit.sparse_rank
                    existing.sparse_score = hit.sparse_score

    ordered = sorted(merged.values(), key=lambda c: -fused_scores[c.chunk_id])[:limit]

    for rank, hit in enumerate(ordered, start=1):
        hit.fused_score = fused_scores[hit.chunk_id]
        hit.score = fused_scores[hit.chunk_id]
        hit.rank = rank
        hit.retriever = "fused"

    return ordered


def explain_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    fused: list[RetrievedChunk],
    *,
    top: int = 10,
) -> str:
    """A readable table of what fusion did. Used by the CLI and the dashboard.

    Being able to *show* this is most of the value of implementing RRF yourself
    rather than importing it: in an interview, "here is a chunk BM25 ranked 2nd
    that dense missed entirely, and fusion kept it at 3" is a concrete story.
    """
    lines = [
        f"{'rank':<5} {'fused':>7}  {'dense':>11}  {'sparse':>11}  found by      chunk",
        "-" * 100,
    ]
    for hit in fused[:top]:
        d = f"#{hit.dense_rank} ({hit.dense_score:.3f})" if hit.dense_rank else "-"
        s = f"#{hit.sparse_rank} ({hit.sparse_score:.1f})" if hit.sparse_rank else "-"
        found = "+".join(hit.found_by()) or "?"
        lines.append(
            f"{hit.rank:<5} {hit.fused_score:>7.4f}  {d:>11}  {s:>11}  {found:<12}  "
            f"{hit.breadcrumb[:44]}"
        )

    only_dense = {h.chunk_id for h in dense} - {h.chunk_id for h in sparse}
    only_sparse = {h.chunk_id for h in sparse} - {h.chunk_id for h in dense}
    both = {h.chunk_id for h in dense} & {h.chunk_id for h in sparse}

    lines.append("")
    lines.append(
        f"candidates: {len(only_dense)} dense-only, {len(only_sparse)} sparse-only, "
        f"{len(both)} found by both"
    )
    return "\n".join(lines)
