"""Qdrant: storing chunk vectors and searching them.

Qdrant holds, for each chunk, a vector plus a "payload" - the metadata we
attached in Step 1.2. The payload is what makes filtered search possible:
"only troubleshooting guides", "only documents a public user may see".

Two details are worth understanding before reading the code.

**Point IDs.** Qdrant only accepts an unsigned integer or a UUID as a point ID.
Our chunk IDs are readable strings like
`k8s-website/tasks/debug/debug-pods#h3`. So we derive a UUID from the string
with uuid5, which is a hash: the same chunk ID always produces the same UUID.
That makes re-indexing idempotent - running it twice updates the same points
instead of creating duplicates - and we keep the readable ID in the payload.

**Cosine distance.** Our vectors are normalised in the embedder, so cosine
similarity is the right metric and scores land in a predictable range.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from copilot.config import settings
from copilot.ingest.models import Chunk

# Any fixed UUID works as a namespace; it just has to never change, or every
# chunk would get a new ID and re-indexing would duplicate the whole collection.
CHUNK_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk ID. Same input, same output, forever."""
    return str(uuid.uuid5(CHUNK_NAMESPACE, chunk_id))


def chunk_payload(chunk: Chunk) -> dict[str, Any]:
    """What we store next to the vector.

    The chunk text is stored here as well as the metadata. That costs disk, and
    it buys not having to re-read a 31 MB JSONL file to display a search result -
    Qdrant returns everything needed to render a citation in one round trip.
    """
    meta = chunk.meta
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "strategy": chunk.strategy,
        "index": chunk.index,
        "heading_path": chunk.heading_path,
        "section_heading": chunk.section_heading,
        "page": chunk.page,
        # Flattened rather than nested, because Qdrant filters address fields by
        # name and flat keys keep filter expressions simple.
        "source_name": meta.source_name,
        "doc_type": str(meta.doc_type),
        "access_level": str(meta.access_level),
        "title": meta.title,
        "url": meta.url,
        "last_updated": meta.last_updated.isoformat() if meta.last_updated else None,
        "age_days": meta.age_days,
    }


class VectorStore:
    """Thin wrapper over the Qdrant client, scoped to one collection."""

    def __init__(self, collection: str | None = None, url: str | None = None):
        self.collection = collection or settings.qdrant_collection
        self.url = url or settings.qdrant_url
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, timeout=60)
        return self._client

    # -- lifecycle ---------------------------------------------------------

    def ping(self) -> bool:
        """Is Qdrant reachable? Checked before long work starts.

        Embedding 26,000 chunks takes minutes. Discovering at the end that the
        database was never running is an avoidable waste - so we ask first.
        """
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def count(self) -> int:
        if not self.exists():
            return 0
        return self.client.count(self.collection, exact=True).count

    def create(self, *, dimension: int | None = None, recreate: bool = False) -> None:
        from qdrant_client.models import Distance, VectorParams

        dimension = dimension or settings.embedding_dim

        if self.exists():
            if not recreate:
                return
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Payload indexes for the fields we filter on.

        Without these, a filtered search scans every point to test the condition.
        With them, Qdrant narrows the candidate set first. At 26,000 chunks the
        difference is small; the habit is what matters, and it costs one call.
        """
        from qdrant_client.models import PayloadSchemaType

        for field in ("source_name", "doc_type", "access_level", "strategy", "doc_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

    # -- writing -----------------------------------------------------------

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or update a batch. 'Upsert' = insert if new, overwrite if not.

        Because point IDs are derived from chunk IDs, re-running the indexer over
        unchanged content is harmless: it rewrites the same points rather than
        piling up duplicates.
        """
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(id=point_id(chunk.chunk_id), vector=vector, payload=chunk_payload(chunk))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=False)

    def flush(self, *, timeout: int = 120) -> None:
        """Wait for Qdrant to finish applying writes sent with wait=False.

        Writes are sent asynchronously for speed, so a count taken immediately
        after the last batch can read low.

        My first attempt at this sent an upsert with an empty point list, assuming
        that would act as a synchronisation barrier. Qdrant rejects it outright:
        `400 Bad request: Empty update request`. There is no flush endpoint.

        What Qdrant does expose is collection status: 'yellow' while the optimizer
        is still working, 'green' when it has caught up. Polling that is the
        supported way to know writes have landed.
        """
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            status = str(self.client.get_collection(self.collection).status).lower()
            if status.endswith("green"):
                return
            time.sleep(1)

    # -- reading -----------------------------------------------------------

    def search(
        self,
        vector: list[float],
        *,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search, optionally filtered by payload fields."""
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if value is None:
                    continue
                match = MatchAny(any=list(value)) if isinstance(value, (list, tuple, set)) else MatchValue(value=value)
                conditions.append(FieldCondition(key=key, match=match))
            if conditions:
                query_filter = Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {"chunk_id": p.payload["chunk_id"], "score": p.score, "payload": p.payload}
            for p in response.points
        ]

    def iter_payloads(self, batch: int = 1000) -> Iterable[dict[str, Any]]:
        """Stream every payload. Used by the BM25 builder in Step 2.2.

        Reading chunk text back out of Qdrant rather than re-parsing the JSONL
        guarantees both indexes describe exactly the same set of chunks. If they
        drifted apart, fusion would be merging rankings of different things.
        """
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield point.payload
            if offset is None:
                break
