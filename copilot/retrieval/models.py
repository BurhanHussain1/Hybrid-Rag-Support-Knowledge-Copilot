"""The shape of a retrieval result.

One model carries a chunk through dense search, sparse search, fusion and
reranking. Each stage fills in its own fields and leaves the earlier ones alone,
so by the time a chunk reaches the LLM you can read its entire history: what
each retriever scored it, where each ranked it, what the reranker thought.

That history is not decoration. In Step 6 you will need to explain *why* hybrid
beat dense-only, and "chunk X was rank 14 in dense, rank 1 in sparse, rank 2
after fusion" is an explanation. A bare list of five chunks is not.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with the scores that got it there."""

    chunk_id: str
    text: str
    score: float = Field(description="Score from the stage that produced this result")
    rank: int = Field(description="1-based position in the result list")
    retriever: str = Field(description="dense, sparse, fused, or reranked")

    # Per-stage detail, all optional. A chunk found only by BM25 has no
    # dense_rank, and that absence is itself informative - it is exactly the case
    # hybrid retrieval exists to cover.
    dense_score: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    payload: dict[str, Any] = Field(default_factory=dict)

    # -- convenience accessors -------------------------------------------
    # The payload is a flat dict from Qdrant. These properties mean the rest of
    # the codebase never types payload["section_heading"] and never has to guess
    # whether a key might be missing.

    @property
    def doc_id(self) -> str:
        return self.payload.get("doc_id", "")

    @property
    def title(self) -> str:
        return self.payload.get("title") or ""

    @property
    def section_heading(self) -> str | None:
        return self.payload.get("section_heading")

    @property
    def url(self) -> str | None:
        return self.payload.get("url")

    @property
    def source_name(self) -> str:
        return self.payload.get("source_name", "")

    @property
    def doc_type(self) -> str:
        return self.payload.get("doc_type", "")

    @property
    def access_level(self) -> str:
        return self.payload.get("access_level", "")

    @property
    def age_days(self) -> int | None:
        return self.payload.get("age_days")

    @property
    def breadcrumb(self) -> str:
        """Human-readable location, for citations and the dashboard."""
        trail = self.payload.get("heading_path") or []
        parts = [self.title, *trail]
        seen: list[str] = []
        for part in parts:
            if part and (not seen or seen[-1].lower() != part.lower()):
                seen.append(part)
        return " > ".join(seen)

    def found_by(self) -> list[str]:
        """Which retrievers surfaced this chunk. Used in the dashboard."""
        found = []
        if self.dense_rank is not None:
            found.append("dense")
        if self.sparse_rank is not None:
            found.append("sparse")
        return found
