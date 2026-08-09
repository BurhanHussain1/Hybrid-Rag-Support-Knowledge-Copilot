"""Turning text into vectors with bge-small, locally on the CPU.

An embedding is a list of numbers describing what a piece of text *means*. Two
texts about the same idea land close together even when they share no words,
which is exactly what keyword search cannot do. bge-small produces 384 numbers
per text.

Running locally rather than calling an API is the deliberate choice here. You
will re-embed all 26,000 chunks every time you change chunk size, and an API bill
attached to that is a bill attached to experimenting. Free re-runs keep the
Step 6 comparison honest.

The one non-obvious detail is the query prefix - see `embed_query` below.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from copilot.config import settings

# bge models are trained asymmetrically: questions and passages are embedded
# differently, and questions get an instruction prefix. Skipping it costs a few
# points of retrieval accuracy for no visible error - the kind of silent quality
# loss that is very hard to attribute later.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Wraps SentenceTransformer with batching and the right query prefix.

    The model is loaded lazily, on first use. Importing this module therefore
    stays instant, which matters because `--help` and unit tests import it too,
    and loading the model takes several seconds.
    """

    def __init__(self, model_name: str | None = None, *, batch_size: int = 64):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"  loading {self.model_name} (first run downloads ~130 MB)...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: Sequence[str], *, show_progress: bool = True) -> list[list[float]]:
        """Embed passages. No prefix - passages are indexed as-is."""
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            # Normalising makes every vector length 1, which turns the dot
            # product into cosine similarity. Qdrant then does less work per
            # query, and scores land in a predictable 0..1 range - much easier
            # to reason about when you are tuning thresholds later.
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question, with the bge instruction prefix applied."""
        vector = self.model.encode(
            BGE_QUERY_PREFIX + text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch version of embed_query, for evaluation runs."""
        vectors = self.model.encode(
            [BGE_QUERY_PREFIX + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def batched(items: Iterable, size: int):
    """Yield lists of at most `size` items.

    Used to keep memory flat while embedding: 26,000 chunks encoded in one call
    would build one enormous array. Batching also lets us upsert to Qdrant as we
    go, so an interrupted run leaves a partially built index rather than nothing.
    """
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


_default: Embedder | None = None


def get_embedder() -> Embedder:
    """Shared instance, so the model is loaded into memory only once."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default
