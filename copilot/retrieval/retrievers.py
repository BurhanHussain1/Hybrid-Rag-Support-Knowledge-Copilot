"""Dense and sparse retrievers behind one shared interface.

Both take a question and return a ranked list of RetrievedChunk. That shared
shape is what lets the fusion layer in Step 3.3 stay simple: it merges rankings
without caring how they were produced, and it will merge a third retriever
just as happily if you add one later.

The two differ in one operationally important way, and it is worth understanding
before reading the code: **Qdrant can filter, BM25 cannot.**

Qdrant applies a metadata filter *during* the search, so asking for the top 20
public troubleshooting chunks returns 20 of exactly those. rank_bm25 scores the
whole corpus with no notion of metadata, so filtering has to happen after the
fact - which means over-fetching and hoping enough survive. See SparseRetriever.
"""

from __future__ import annotations

from typing import Any, Protocol

from copilot.config import settings
from copilot.retrieval.bm25_index import BM25Index
from copilot.retrieval.embedder import Embedder, get_embedder
from copilot.retrieval.models import RetrievedChunk
from copilot.retrieval.vector_store import VectorStore


class Retriever(Protocol):
    """What every retriever must provide.

    A Protocol rather than a base class: no inheritance required, just the right
    method. Anything with this shape works with the fusion layer.
    """

    name: str

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        ...


class DenseRetriever:
    """Vector search: finds chunks that *mean* something similar to the question.

    Strong on paraphrase - "my pods keep restarting" matches a page titled
    "Container restarts" with no shared vocabulary. Weak on literal strings: it
    blurs `CrashLoopBackOff` toward whatever it resembles, because to an
    embedding model a rare token is mostly a shape.
    """

    name = "dense"

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None):
        self.store = store or VectorStore()
        self.embedder = embedder or get_embedder()

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        vector = self.embedder.embed_query(query)
        hits = self.store.search(vector, limit=limit, filters=filters)

        return [
            RetrievedChunk(
                chunk_id=hit["chunk_id"],
                text=hit["payload"].get("text", ""),
                score=hit["score"],
                rank=rank,
                retriever=self.name,
                # Kept separately as well as in `score`, because after fusion
                # `score` will hold the fused value and we still want to be able
                # to say "dense scored this 0.83".
                dense_score=hit["score"],
                dense_rank=rank,
                payload=hit["payload"],
            )
            for rank, hit in enumerate(hits, start=1)
        ]


class SparseRetriever:
    """BM25 keyword search: finds chunks containing the question's actual words.

    Strong on anything quoted verbatim - error codes, CLI flags, API names, SKUs.
    Useless on paraphrase: it has no idea that "restarting" relates to
    "CrashLoopBackOff", so a question sharing no words with the right document
    scores that document zero.
    """

    name = "sparse"

    # How much to over-fetch when a filter is active. BM25 scores the whole
    # corpus and knows nothing about metadata, so the only way to end up with 20
    # results that match a filter is to take a larger slice and discard the rest.
    #
    # This is a genuine weakness of sparse-side filtering, not a detail to hide:
    # if a filter is very selective, even 5x may not yield enough survivors, and
    # the sparse arm quietly contributes fewer candidates than the dense one.
    # `search` reports that in the returned list length rather than pretending.
    FILTER_OVERFETCH = 5

    def __init__(self, index: BM25Index | None = None, strategy: str | None = None):
        self.strategy = strategy or settings.chunk_strategy
        self._index = index

    @property
    def index(self) -> BM25Index:
        # Loaded on first use: unpickling a 43 MB index takes a moment, and
        # importing this module should stay free.
        if self._index is None:
            self._index = BM25Index.load(self.strategy)
        return self._index

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        fetch = limit * self.FILTER_OVERFETCH if filters else limit
        hits = self.index.search(query, limit=fetch)

        if filters:
            hits = [hit for hit in hits if _matches(hit["payload"], filters)][:limit]

        return [
            RetrievedChunk(
                chunk_id=hit["chunk_id"],
                text=hit["payload"].get("text", ""),
                score=hit["score"],
                rank=rank,
                retriever=self.name,
                sparse_score=hit["score"],
                sparse_rank=rank,
                payload=hit["payload"],
            )
            for rank, hit in enumerate(hits, start=1)
        ]


def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Post-filter for BM25 results, mirroring Qdrant's filter semantics.

    Both sides must agree on what a filter means, or dense and sparse would be
    searching subtly different corpora and fusion would compare apples to pears.
    A list value means "any of these"; a scalar means "exactly this".
    """
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = payload.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
