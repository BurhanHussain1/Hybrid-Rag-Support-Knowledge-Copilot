#!/usr/bin/env python
"""Ask the support copilot a question.

    python ask.py "why is my pod stuck in pending"
    python ask.py "what is the refund policy"                 # expect a refusal
    python ask.py "how do I archive a channel" --mode hybrid
    python ask.py "pod crashlooping" --dry-run                # show the prompt, call nothing
    python ask.py "parental leave" --no-verify                # skip citation checking
    python ask.py "how do I change my email" --json           # machine-readable

--dry-run assembles the full prompt and prints it without contacting OpenAI.
Reading the exact text the model receives is the fastest way to understand why it
answers the way it does, and it costs nothing.
"""

from __future__ import annotations

import argparse
import sys

from copilot.config import settings
from copilot.generation.answerer import Answerer
from copilot.generation.llm import MissingAPIKey
from copilot.generation.pipeline import Copilot, CopilotAnswer
from copilot.retrieval.hybrid import HybridRetriever

MODES = ("dense", "sparse", "hybrid", "rerank")

VERDICT_MARK = {
    "supported": "OK      ",
    "partial": "PARTIAL ",
    "unsupported": "FAILED  ",
    "unverifiable": "SKIPPED ",
    None: "unchecked",
}


def print_answer(response: CopilotAnswer) -> None:
    header = "ANSWER" if response.answered else f"REFUSED ({response.refusal_reason})"
    print("\n" + "=" * 78)
    print(f"{header}     confidence {response.confidence:.2f}")
    print("=" * 78)
    print(response.answer)

    if response.citations:
        print("\nCITATIONS")
        for cite in response.citations:
            mark = VERDICT_MARK.get(cite.verdict, cite.verdict or "?")
            stale = "  [STALE]" if cite.is_stale else ""
            print(f"  [{cite.label}] {mark}  {cite.breadcrumb[:60]}{stale}")
            if cite.verdict_reason:
                print(f"        why: {cite.verdict_reason[:110]}")
            if cite.url:
                print(f"        {cite.url}")

    if response.unverified:
        print("\nWHAT I COULD NOT VERIFY")
        for item in response.unverified:
            print(f"  - {item}")

    if response.nearest_sections:
        print("\nCLOSEST MATCHING SECTIONS")
        for near in response.nearest_sections:
            print(f"  - {near.breadcrumb[:64]}")
            if near.url:
                print(f"    {near.url}")

    if response.confidence_breakdown:
        print("\nCONFIDENCE BREAKDOWN")
        for line in response.confidence_breakdown.explain().splitlines():
            print(f"  {line}")

    if response.citations and response.judge_shares_model_with_generator:
        print("\n  caveat: the citation judge is the same model that wrote the answer,")
        print("          so it is predisposed to agree with itself. Set JUDGE_MODEL in")
        print("          .env to a different model for an independent check.")

    timings = "  ".join(f"{k} {v:.0f}ms" for k, v in response.timings_ms.items())
    print(f"\n{timings}")
    if response.usage:
        print(f"{response.usage['prompt_tokens']} in + {response.usage['completion_tokens']} out "
              f"tokens across {response.usage['calls']} calls "
              f"(~${response.usage['estimated_cost_usd']:.4f})\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ask.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="rerank")
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--dry-run", action="store_true", help="print the prompt, call nothing")
    parser.add_argument("--no-verify", action="store_true", help="skip citation verification")
    parser.add_argument("--show-sources", action="store_true", help="print retrieved chunks first")
    parser.add_argument("--json", action="store_true", help="emit the full response as JSON")
    parser.add_argument(
        "--filter", action="append", metavar="KEY=VALUE", help="metadata filter, repeatable"
    )
    args = parser.parse_args(argv)

    filters = dict(pair.split("=", 1) for pair in args.filter) if args.filter else None

    # --dry-run stops before the model call, so it must not go through Copilot.ask.
    if args.dry_run:
        result = HybridRetriever().retrieve(
            args.question, mode=args.mode, filters=filters, top_k=args.top_k
        )
        system, user, _ = Answerer().build_prompt(args.question, result.chunks)
        print(f"\nRETRIEVED {len(result.chunks)} chunks "
              f"({sum(result.timings_ms.values()):.0f}ms, mode={args.mode})")
        for hit in result.chunks:
            print(f"  [{hit.rank}] {hit.score:.4f}  {hit.breadcrumb[:70]}")
        print("\n" + "=" * 78 + "\nSYSTEM PROMPT\n" + "=" * 78)
        print(system)
        print("\n" + "=" * 78 + "\nUSER PROMPT\n" + "=" * 78)
        print(user)
        size = len(system) + len(user)
        print(f"\napprox prompt size: {size} chars (~{size // 4} tokens)")
        return 0

    try:
        response = Copilot().ask(
            args.question,
            mode=args.mode,
            filters=filters,
            top_k=args.top_k,
            verify=not args.no_verify,
        )
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 1

    if args.json:
        print(response.model_dump_json(indent=2))
        return 0

    if args.show_sources:
        print(f"\nmode={args.mode}")

    print_answer(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
