#!/usr/bin/env python
"""Ingest the documentation corpus into chunk files.

    python ingest.py --rebuild                     # heading strategy (default)
    python ingest.py --strategy fixed --rebuild    # fixed-size strategy
    python ingest.py --strategy both --rebuild     # both, for Step 6 comparison
    python ingest.py --source posthog --limit 50   # quick iteration
    python ingest.py --stats                       # what is already on disk

Output goes to data/processed/chunks_<strategy>.jsonl, with a matching
manifest_<strategy>.json recording the settings that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys

from copilot.config import settings
from copilot.ingest.pipeline import (
    chunks_path,
    manifest_path,
    print_stats,
    run_ingestion,
)

STRATEGIES = ("heading", "fixed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ingest.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strategy",
        choices=[*STRATEGIES, "both"],
        default=settings.chunk_strategy,
        help=f"chunking strategy (default: {settings.chunk_strategy}, from .env)",
    )
    parser.add_argument(
        "--source",
        help="only ingest paths starting with this, e.g. 'posthog' or 'posthog/contents/handbook'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="stop after N documents - useful when iterating on chunking",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="replace existing chunk files (required if they already exist)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="show what is already ingested and exit, without running anything",
    )
    return parser


def show_existing() -> int:
    """Report what is on disk. Read-only: safe to run any time."""
    found = False
    for strategy in STRATEGIES:
        path = chunks_path(strategy)
        if not path.exists():
            continue
        found = True

        manifest = {}
        if manifest_path(strategy).exists():
            manifest = json.loads(manifest_path(strategy).read_text(encoding="utf-8"))

        stats = manifest.get("stats", {})
        conf = manifest.get("settings", {})
        size_mb = path.stat().st_size / 1_000_000

        print(f"\n{path.name}  ({size_mb:.1f} MB)")
        print(f"  created    {manifest.get('created_at', 'unknown')}")
        print(f"  chunks     {stats.get('chunks_written', '?')}")
        print(f"  documents  {stats.get('documents_chunked', '?')}")
        print(f"  chunk_size {conf.get('chunk_size', '?')}  overlap {conf.get('chunk_overlap', '?')}")

    if not found:
        print("Nothing ingested yet. Run: python ingest.py --rebuild")
        return 1
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.stats:
        return show_existing()

    targets = list(STRATEGIES) if args.strategy == "both" else [args.strategy]

    for strategy in targets:
        try:
            stats = run_ingestion(
                strategy,
                source=args.source,
                limit=args.limit,
                rebuild=args.rebuild,
                quiet=args.quiet,
            )
        except FileExistsError as exc:
            # A clear message and a non-zero exit code, not a traceback. Anything
            # scripting this - a Makefile, CI, Step 2 - checks the exit code.
            print(f"error: {exc}", file=sys.stderr)
            return 1

        if not args.quiet:
            print_stats(stats, strategy)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
