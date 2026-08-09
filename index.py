#!/usr/bin/env python
"""Build the search indexes from the chunk files produced by ingest.py.

    python index.py --strategy heading --rebuild     # embed and load into Qdrant
    python index.py --stats                          # what is indexed right now
    python index.py --search "why is my pod crashing"  # quick sanity check

Deliberately separate from ingest.py. Chunking takes 15 seconds; embedding takes
minutes. Keeping them apart means you can re-embed without re-chunking, and
re-chunk without waiting on the GPU-less encoder. One command per expensive
thing is a good default.
"""

from __future__ import annotations

import argparse
import sys
import time

from copilot.config import settings
from copilot.ingest.pipeline import chunks_path, load_chunks
from copilot.retrieval.bm25_index import BM25Index
from copilot.retrieval.embedder import batched, get_embedder
from copilot.retrieval.vector_store import VectorStore

STRATEGIES = ("heading", "fixed")


def build_bm25(strategy: str) -> int:
    """Build the sparse index by reading chunks back out of Qdrant.

    Reading from Qdrant rather than the JSONL file is the important bit. It makes
    it structurally impossible for the two indexes to cover different chunk sets:
    whatever is in the vector store is exactly what BM25 sees. Re-parsing the
    JSONL would work today and silently diverge the first time someone re-ingests
    without re-indexing.
    """
    store = VectorStore()
    if not store.exists():
        print("error: no Qdrant collection to read from", file=sys.stderr)
        return 1

    print(f"\nbuilding BM25 index over the same chunk IDs")
    records = [
        (payload["chunk_id"], payload["text"], payload)
        for payload in store.iter_payloads(batch=1000)
    ]

    started = time.time()
    bm25 = BM25Index.build(records)
    target = bm25.save(strategy)

    size_mb = target.stat().st_size / 1_000_000
    print(f"  {len(bm25)} chunks indexed in {time.time() - started:.0f}s")
    print(f"  saved to {target.name} ({size_mb:.1f} MB)\n")
    return 0


def build_index(strategy: str, *, rebuild: bool, batch_size: int, limit: int | None) -> int:
    store = VectorStore()

    # Check the database is reachable before doing minutes of work. Finding out
    # at the end that Qdrant was never running is a completely avoidable waste.
    if not store.ping():
        print(
            f"error: cannot reach Qdrant at {settings.qdrant_url}\n"
            "       start it with: docker compose up -d",
            file=sys.stderr,
        )
        return 1

    if not chunks_path(strategy).exists():
        print(
            f"error: {chunks_path(strategy).name} not found\n"
            f"       run: python ingest.py --strategy {strategy} --rebuild",
            file=sys.stderr,
        )
        return 1

    if store.exists() and not rebuild:
        print(
            f"error: collection '{store.collection}' already has {store.count()} points\n"
            "       pass --rebuild to replace it",
            file=sys.stderr,
        )
        return 1

    embedder = get_embedder()
    dimension = embedder.dimension

    # Guard against a silent, painful mismatch: if .env says 384 but the model
    # emits 768, Qdrant would reject every write with an opaque error, or worse,
    # the collection would be built wrong.
    if dimension != settings.embedding_dim:
        print(
            f"error: {embedder.model_name} outputs {dimension} dimensions, "
            f"but EMBEDDING_DIM is {settings.embedding_dim}. Update .env.",
            file=sys.stderr,
        )
        return 1

    print(f"collection  {store.collection}  ({dimension}-dim, cosine)")
    store.create(dimension=dimension, recreate=rebuild)

    # Load everything, then sort by text length before batching.
    #
    # This is the single biggest performance win in the whole pipeline, and it
    # costs nothing in quality. A transformer processes a batch as one rectangular
    # matrix, so every text in a batch is padded to the length of the longest one.
    # Our chunks run from 200 to 1,719 characters with a median of 603 - so a
    # random batch of 256 pads almost everything to ~1,700 and spends roughly
    # two-thirds of its compute on padding tokens.
    #
    # Sorting first means each batch holds chunks of near-identical length, so
    # there is almost nothing to pad. Measured: 9 chunks/s -> see the run output.
    #
    # Insertion order is irrelevant because point IDs are derived from chunk IDs,
    # not from position.
    print("  loading chunks...")
    all_chunks = list(load_chunks(strategy))
    if limit:
        all_chunks = all_chunks[:limit]
    all_chunks.sort(key=lambda c: len(c.embedding_text))
    print(f"  {len(all_chunks)} chunks, sorted by length to minimise padding waste")

    started = time.time()
    total = 0

    for batch in batched(all_chunks, batch_size):

        # embedding_text, not text: the breadcrumb prefix is part of what makes a
        # chunk findable. See Chunk.embedding_text in copilot/ingest/models.py.
        vectors = embedder.embed_documents(
            [chunk.embedding_text for chunk in batch], show_progress=False
        )
        store.upsert(batch, vectors)
        total += len(batch)

        elapsed = time.time() - started
        rate = total / elapsed if elapsed else 0
        print(f"  {total:>6} chunks  {rate:>6.0f}/s", end="\r", flush=True)

    store.flush()
    elapsed = time.time() - started

    print(f"\n\n  indexed     {total} chunks in {elapsed:.0f}s ({total/max(elapsed,1):.0f}/s)")
    print(f"  in qdrant   {store.count()} points")
    print(f"  dashboard   {settings.qdrant_url}/dashboard")
    return 0


def show_stats() -> int:
    store = VectorStore()
    if not store.ping():
        print(f"Qdrant unreachable at {settings.qdrant_url}. Run: docker compose up -d")
        return 1
    if not store.exists():
        print(f"Collection '{store.collection}' does not exist. Run: python index.py --rebuild")
        return 1

    print(f"\ncollection  {store.collection}")
    print(f"points      {store.count()}")

    # Sample a page of payloads to show the mix, without scanning everything.
    from collections import Counter

    sources: Counter = Counter()
    strategies: Counter = Counter()
    for i, payload in enumerate(store.iter_payloads(batch=1000)):
        sources[payload.get("source_name")] += 1
        strategies[payload.get("strategy")] += 1
        if i >= 4999:
            break

    print(f"\nsample of {sum(sources.values())} points:")
    print(f"  strategies  {dict(strategies)}")
    print(f"  sources     {dict(sources.most_common())}")

    for strategy in STRATEGIES:
        path = BM25Index.path(strategy)
        if path.exists():
            print(f"\nbm25        {path.name} ({path.stat().st_size / 1_000_000:.1f} MB)")
    print()
    return 0


def quick_search(query: str, limit: int, mode: str, strategy: str) -> int:
    """Single-index search. A sanity check, not the real retriever - that is Step 3."""
    if mode == "sparse":
        results = BM25Index.load(strategy).search(query, limit=limit)
    else:
        store = VectorStore()
        if not store.exists():
            print("Nothing indexed yet. Run: python index.py --rebuild", file=sys.stderr)
            return 1
        results = store.search(get_embedder().embed_query(query), limit=limit)

    print(f'\nquery: "{query}"  ({mode} only)\n')
    if not results:
        print("  no matches\n")
        return 0

    for rank, hit in enumerate(results, start=1):
        payload = hit["payload"]
        trail = " > ".join(payload.get("heading_path") or []) or "(no heading)"
        print(f"{rank:>2}. {hit['score']:.3f}  {payload['source_name']:<13} {payload['doc_type']}")
        print(f"     {trail}")
        print(f"     {payload['text'][:150].replace(chr(10), ' ')}...")
        print(f"     {payload['chunk_id']}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="index.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", choices=STRATEGIES, default=settings.chunk_strategy)
    parser.add_argument("--rebuild", action="store_true", help="wipe and rebuild the collection")
    parser.add_argument("--batch-size", type=int, default=256, help="chunks per embedding batch")
    parser.add_argument("--limit", type=int, help="stop after N chunks (for a fast smoke test)")
    parser.add_argument("--stats", action="store_true", help="show what is indexed and exit")
    parser.add_argument("--search", metavar="QUERY", help="run a search and exit")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse"),
        default="dense",
        help="which index --search uses (fusion arrives in Step 3)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="results for --search")
    parser.add_argument("--skip-bm25", action="store_true", help="build only the vector index")
    parser.add_argument("--bm25-only", action="store_true", help="rebuild only the sparse index")
    args = parser.parse_args(argv)

    if args.stats:
        return show_stats()
    if args.search:
        return quick_search(args.search, args.top_k, args.mode, args.strategy)
    if args.bm25_only:
        return build_bm25(args.strategy)

    code = build_index(
        args.strategy,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if code != 0 or args.skip_bm25:
        return code

    return build_bm25(args.strategy)


if __name__ == "__main__":
    raise SystemExit(main())
