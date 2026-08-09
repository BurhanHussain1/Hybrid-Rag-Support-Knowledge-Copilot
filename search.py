#!/usr/bin/env python
"""Query the retrieval pipeline from the command line.

    python search.py "why is my pod crashing"
    python search.py "CrashLoopBackOff" --mode sparse
    python search.py "how do I change my email" --explain
    python search.py "parental leave" --compare
    python search.py "pod stuck pending" --filter doc_type=troubleshooting

This is the tool you will actually use while tuning. --compare runs all four
modes on one question and prints them side by side, which is how you build an
intuition for what fusion and reranking are really doing before you try to
measure it in Step 6.
"""

from __future__ import annotations

import argparse
import sys

from copilot.config import settings
from copilot.retrieval.fusion import explain_fusion
from copilot.retrieval.hybrid import HybridRetriever, RetrievalResult

MODES = ("dense", "sparse", "hybrid", "rerank")


def parse_filters(pairs: list[str] | None) -> dict[str, str] | None:
    """Turn ['doc_type=faq', 'source_name=zulip'] into a filter dict."""
    if not pairs:
        return None
    filters: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"bad --filter {pair!r}, expected key=value")
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def print_result(result: RetrievalResult, *, show_text: bool = True) -> None:
    timings = "  ".join(f"{k} {v:.0f}ms" for k, v in result.timings_ms.items())
    total = sum(result.timings_ms.values())
    print(f"\nmode={result.mode}  {timings}  (total {total:.0f}ms)")

    if result.mode in ("hybrid", "rerank"):
        cov = result.coverage()
        print(
            f"final {len(result.chunks)} chunks: {cov['both']} found by both, "
            f"{cov['dense_only']} dense-only, {cov['sparse_only']} sparse-only"
        )
    print()

    for hit in result.chunks:
        marks = []
        if hit.dense_rank:
            marks.append(f"dense #{hit.dense_rank}")
        if hit.sparse_rank:
            marks.append(f"sparse #{hit.sparse_rank}")
        origin = ", ".join(marks) or "-"

        print(f"{hit.rank}. [{hit.score:.4f}]  {hit.source_name} / {hit.doc_type}")
        print(f"   {hit.breadcrumb[:88]}")
        print(f"   found by: {origin}")
        if hit.rerank_score is not None:
            print(f"   rerank score: {hit.rerank_score:.3f}   fused: {hit.fused_score:.4f}")
        if hit.age_days is not None:
            print(f"   updated {hit.age_days} days ago")
        if hit.url:
            print(f"   {hit.url}")
        if show_text:
            snippet = " ".join(hit.text.split())[:230]
            print(f"   \"{snippet}...\"")
        print(f"   id: {hit.chunk_id}")
        print()


def compare_modes(retriever: HybridRetriever, query: str, filters, top_k: int) -> None:
    """Run every mode on one query and show the top results of each."""
    print(f'\nQUERY: "{query}"')
    if filters:
        print(f"FILTER: {filters}")

    for mode in MODES:
        result = retriever.retrieve(query, mode=mode, filters=filters, top_k=top_k)
        total = sum(result.timings_ms.values())
        print(f"\n{'=' * 92}")
        print(f"{mode.upper():<8}  {total:>6.0f}ms")
        print("=" * 92)
        for hit in result.chunks:
            origin = "+".join(hit.found_by()) or "-"
            print(f"  {hit.rank}. [{hit.score:>8.4f}] {origin:<13} {hit.source_name:<12} "
                  f"{hit.breadcrumb[:52]}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="search.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", help="the question to search for")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument(
        "--filter",
        action="append",
        metavar="KEY=VALUE",
        help="metadata filter, repeatable (doc_type, source_name, access_level)",
    )
    parser.add_argument("--explain", action="store_true", help="show the RRF fusion table")
    parser.add_argument("--compare", action="store_true", help="run all four modes side by side")
    parser.add_argument("--no-text", action="store_true", help="hide chunk snippets")
    parser.add_argument("--strategy", default=settings.chunk_strategy, help="which chunk set to use")
    args = parser.parse_args(argv)

    filters = parse_filters(args.filter)
    retriever = HybridRetriever(strategy=args.strategy)

    if args.compare:
        compare_modes(retriever, args.query, filters, args.top_k)
        return 0

    result = retriever.retrieve(args.query, mode=args.mode, filters=filters, top_k=args.top_k)

    if not result.chunks:
        print("\nNo results. Check that the index is built: python index.py --stats", file=sys.stderr)
        return 1

    print(f'\nQUERY: "{args.query}"')
    if filters:
        print(f"FILTER: {filters}")
    print_result(result, show_text=not args.no_text)

    if args.explain and result.fused_candidates:
        print(explain_fusion(
            result.dense_candidates, result.sparse_candidates, result.fused_candidates
        ))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
