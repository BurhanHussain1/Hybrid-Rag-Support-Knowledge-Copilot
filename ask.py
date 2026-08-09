#!/usr/bin/env python
"""Ask the support copilot a question.

    python ask.py "why is my pod stuck in pending"
    python ask.py "how do I archive a channel" --mode hybrid
    python ask.py "what is the refund policy" --dry-run      # show the prompt, call nothing
    python ask.py "pod crashlooping" --show-sources

--dry-run assembles the full prompt and prints it without contacting OpenAI.
Worth running once before you spend anything: reading the exact text the model
receives is the fastest way to understand why it answers the way it does.
"""

from __future__ import annotations

import argparse
import sys

from copilot.config import settings
from copilot.generation.answerer import Answerer
from copilot.generation.llm import MissingAPIKey
from copilot.retrieval.hybrid import HybridRetriever

MODES = ("dense", "sparse", "hybrid", "rerank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ask.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--dry-run", action="store_true", help="print the prompt, do not call the LLM")
    parser.add_argument("--show-sources", action="store_true", help="print retrieved chunks first")
    parser.add_argument(
        "--filter", action="append", metavar="KEY=VALUE", help="metadata filter, repeatable"
    )
    args = parser.parse_args(argv)

    filters = None
    if args.filter:
        filters = dict(pair.split("=", 1) for pair in args.filter)

    retriever = HybridRetriever()
    result = retriever.retrieve(args.question, mode=args.mode, filters=filters, top_k=args.top_k)

    if args.show_sources or args.dry_run:
        print(f"\nRETRIEVED {len(result.chunks)} chunks "
              f"({sum(result.timings_ms.values()):.0f}ms, mode={args.mode})")
        for hit in result.chunks:
            print(f"  [{hit.rank}] {hit.score:.4f}  {hit.breadcrumb[:70]}")

    answerer = Answerer()

    if args.dry_run:
        system, user, _ = answerer.build_prompt(args.question, result.chunks)
        print("\n" + "=" * 78)
        print("SYSTEM PROMPT")
        print("=" * 78)
        print(system)
        print("\n" + "=" * 78)
        print("USER PROMPT")
        print("=" * 78)
        print(user)
        print("\n" + "=" * 78)
        print(f"approx prompt size: {len(system) + len(user)} chars "
              f"(~{(len(system) + len(user)) // 4} tokens)")
        return 0

    try:
        generated = answerer.answer(args.question, result.chunks)
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("ANSWER" if generated.answerable else "NO ANSWER FOUND")
    print("=" * 78)
    print(generated.answer)

    if generated.citations:
        print("\nCITATIONS")
        for cite in generated.citations:
            stale = "  [STALE]" if cite.is_stale else ""
            print(f"  [{cite.label}] {cite.breadcrumb[:66]}{stale}")
            if cite.url:
                print(f"      {cite.url}")

    if generated.unverified:
        print("\nWHAT I COULD NOT VERIFY")
        for item in generated.unverified:
            print(f"  - {item}")

    usage = answerer.llm.usage_summary()
    print(f"\n{usage['prompt_tokens']} in + {usage['completion_tokens']} out tokens "
          f"(~${usage['estimated_cost_usd']:.4f})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
