"""The ingestion pipeline: load -> enrich -> chunk -> write.

This module holds the logic. `ingest.py` at the repo root holds the argument
parsing. Keeping them apart means the pipeline can be imported and called from a
test, a notebook, or Step 2's indexer without going through a command line - and
the CLI stays small enough to read in one sitting.

Output is JSONL: one JSON object per line, in `data/processed/`.

Why JSONL rather than one big JSON array:
  - it streams. We write chunk-by-chunk and never hold 26,000 objects in memory,
    and Step 2 can read it back the same way.
  - it survives interruption. A half-written JSON array is unparseable; a
    half-written JSONL file is just a file with fewer lines.
  - `wc -l` tells you the chunk count instantly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from copilot.config import PROCESSED_DIR, RAW_DIR, settings
from copilot.ingest.chunking import chunk_document
from copilot.ingest.loaders import load_corpus
from copilot.ingest.metadata import GitDateIndex, enrich


@dataclass
class IngestionStats:
    """What happened during a run. Printed at the end and saved to the manifest."""

    documents_loaded: int = 0
    documents_chunked: int = 0
    documents_dropped: int = 0
    chunks_written: int = 0
    chars_in: int = 0
    chars_out: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_doc_type: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def chunks_path(strategy: str, out_dir: Path | None = None) -> Path:
    """One file per strategy, so both can exist side by side.

    Step 6 compares 'heading' against 'fixed' on the same questions. That is only
    possible if both chunk sets are on disk at once, which means the strategy
    name has to be in the filename.
    """
    return (out_dir or PROCESSED_DIR) / f"chunks_{strategy}.jsonl"


def manifest_path(strategy: str, out_dir: Path | None = None) -> Path:
    return (out_dir or PROCESSED_DIR) / f"manifest_{strategy}.json"


def run_ingestion(
    strategy: str,
    *,
    source: str | None = None,
    limit: int | None = None,
    rebuild: bool = False,
    out_dir: Path | None = None,
    raw_dir: Path | None = None,
    quiet: bool = False,
) -> IngestionStats:
    """Load, enrich, chunk, and write every document to JSONL."""
    import time

    started = time.time()
    out_dir = out_dir or PROCESSED_DIR
    raw_dir = raw_dir or RAW_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    target = chunks_path(strategy, out_dir)

    # Without --rebuild we refuse to overwrite. Silently replacing an index that
    # a saved evaluation result refers to is how you end up comparing numbers
    # that were never produced by the same data.
    if target.exists() and not rebuild:
        raise FileExistsError(
            f"{target.name} already exists. Pass --rebuild to replace it."
        )

    git_index = GitDateIndex(raw_dir=raw_dir)
    stats = IngestionStats()

    def log(message: str) -> None:
        if not quiet:
            print(message)

    log(f"ingesting  strategy={strategy}  size={settings.chunk_size}  "
        f"overlap={settings.chunk_overlap}  min={settings.min_chunk_chars}")
    if source:
        log(f"filter     source={source}")

    # Open the file once and stream into it. Buffering everything and writing at
    # the end would mean a crash at document 3,000 leaves you with nothing.
    with target.open("w", encoding="utf-8") as handle:
        for doc in load_corpus(raw_dir):
            # `source` accepts a corpus name ("posthog") or any path prefix
            # ("posthog/contents/handbook"), because both are things you actually
            # want to re-ingest on their own while iterating.
            if source and not doc.rel_path.startswith(source.replace("\\", "/").strip("/")):
                continue

            stats.documents_loaded += 1
            stats.chars_in += doc.char_count

            enrich(doc, git_index)
            chunks = chunk_document(doc, strategy)

            if not chunks:
                stats.documents_dropped += 1
                continue

            stats.documents_chunked += 1
            for chunk in chunks:
                # model_dump_json handles the datetime in last_updated; plain
                # json.dumps would raise "Object of type datetime is not JSON
                # serializable" partway through the run.
                handle.write(chunk.model_dump_json() + "\n")
                stats.chunks_written += 1
                stats.chars_out += chunk.char_count
                stats.by_source[chunk.meta.source_name] = stats.by_source.get(chunk.meta.source_name, 0) + 1
                key = str(chunk.meta.doc_type)
                stats.by_doc_type[key] = stats.by_doc_type.get(key, 0) + 1

            if not quiet and stats.documents_loaded % 500 == 0:
                log(f"  ... {stats.documents_loaded} docs, {stats.chunks_written} chunks")

            if limit and stats.documents_loaded >= limit:
                break

    stats.seconds = time.time() - started
    _write_manifest(strategy, stats, source=source, out_dir=out_dir)
    return stats


def _write_manifest(strategy: str, stats: IngestionStats, *, source: str | None, out_dir: Path) -> None:
    """Record the settings that produced this chunk file.

    This is the difference between "hybrid scored 88%" and "hybrid scored 88%
    with 800-character heading chunks ingested on 9 August". Six weeks later,
    only the second sentence is worth anything - and you cannot reconstruct it
    from memory.
    """
    manifest = {
        "strategy": strategy,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_filter": source,
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "min_chunk_chars": settings.min_chunk_chars,
            "max_chunk_chars": settings.max_chunk_chars,
            "embedding_model": settings.embedding_model,
        },
        "stats": {
            "documents_loaded": stats.documents_loaded,
            "documents_chunked": stats.documents_chunked,
            "documents_dropped": stats.documents_dropped,
            "chunks_written": stats.chunks_written,
            "chars_in": stats.chars_in,
            "chars_out": stats.chars_out,
            "seconds": round(stats.seconds, 2),
            "by_source": stats.by_source,
            "by_doc_type": stats.by_doc_type,
        },
    }
    manifest_path(strategy, out_dir).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_chunks(strategy: str, out_dir: Path | None = None):
    """Read chunks back from JSONL, one at a time.

    Step 2 uses this to build the indexes. A generator keeps memory flat whether
    the file has 26,000 chunks or 26 million.
    """
    from copilot.ingest.models import Chunk

    path = chunks_path(strategy, out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python ingest.py --strategy {strategy}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Chunk.model_validate_json(line)


def print_stats(stats: IngestionStats, strategy: str) -> None:
    ratio = stats.chars_out / stats.chars_in if stats.chars_in else 0
    print()
    print(f"  strategy        {strategy}")
    print(f"  documents       {stats.documents_chunked} chunked, {stats.documents_dropped} dropped (too short)")
    print(f"  chunks          {stats.chunks_written}")
    print(f"  avg chunk       {stats.chars_out // max(stats.chunks_written, 1)} chars")
    print(f"  text ratio      {ratio:.2f}x source")
    print(f"  time            {stats.seconds:.1f}s")
    print(f"  output          {chunks_path(strategy).relative_to(chunks_path(strategy).parents[2])}")
    print()
    print("  by source:")
    for name, count in sorted(stats.by_source.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<16} {count:>6}")
    print("  by doc type:")
    for name, count in sorted(stats.by_doc_type.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<16} {count:>6}")
