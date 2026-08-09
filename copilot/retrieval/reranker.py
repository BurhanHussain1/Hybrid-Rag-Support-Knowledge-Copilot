"""Cross-encoder reranking: the accuracy pass over a small candidate set.

There are two ways to score how well a document answers a question, and the
difference between them is the reason this file exists.

**Bi-encoder** (what the embedder does). Question and document are encoded
*separately* into vectors, then compared. Because documents are encoded ahead of
time, searching 26,000 chunks costs one encode of the question plus a fast
vector comparison. That is what makes search feasible. The cost is that the model
never sees the question and the document at the same time - it compresses each
into 384 numbers and hopes the comparison survives the compression.

**Cross-encoder** (this file). Question and document go through the model
*together*, as one input, and it outputs a single relevance score. The model can
actually attend from words in the question to words in the document. It is far
more accurate - and it cannot be precomputed, because the score depends on the
pair. Scoring 26,000 chunks per query would take minutes.

So you use both, in sequence:

    26,000 chunks  --dense + sparse-->  20 candidates  --cross-encoder-->  top 5
                     fast, approximate                    slow, accurate

This is the standard retrieve-then-rerank pattern. Fusion gets the right chunks
*into* the top 20; the reranker gets the right ones to the *top* of it. That
matters because only 5 chunks reach the LLM, and a correct chunk sitting at
position 12 might as well not have been retrieved at all.
"""

from __future__ import annotations

from copilot.config import settings
from copilot.retrieval.models import RetrievedChunk


class Reranker:
    """Scores (question, chunk) pairs with a cross-encoder."""

    def __init__(self, model_name: str | None = None, *, batch_size: int = 32):
        self.model_name = model_name or settings.reranker_model
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            print(f"  loading reranker {self.model_name} (first run downloads ~90 MB)...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        limit: int | None = None,
    ) -> list[RetrievedChunk]:
        """Reorder candidates by cross-encoder relevance, keeping the best `limit`."""
        limit = limit or settings.final_top_k

        if not candidates:
            return []

        # The pair text uses breadcrumb + chunk text, matching what we embedded.
        # Feeding the reranker a bare chunk while the retriever saw the breadcrumb
        # would mean the two stages are judging different things.
        pairs = [(query, f"{hit.breadcrumb}\n\n{hit.text}") for hit in candidates]

        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        scored = []
        for hit, score in zip(candidates, scores, strict=True):
            copy = hit.model_copy(deep=True)
            copy.rerank_score = float(score)
            scored.append(copy)

        scored.sort(key=lambda c: -(c.rerank_score or 0.0))

        for rank, hit in enumerate(scored[:limit], start=1):
            hit.rank = rank
            hit.score = hit.rerank_score or 0.0
            hit.retriever = "reranked"

        return scored[:limit]


_default: Reranker | None = None


def get_reranker() -> Reranker:
    """Shared instance, so the model loads once per process."""
    global _default
    if _default is None:
        _default = Reranker()
    return _default
