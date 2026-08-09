#!/usr/bin/env python
"""Run the evaluation suite and write a Markdown report.

    python eval.py                              # retrieval only, all modes - free
    python eval.py --strategy hybrid            # retrieval for one mode
    python eval.py --full                       # adds generation + citation checks (costs ~$0.05)
    python eval.py --full --answer-mode rerank
    python eval.py --limit 10 --full            # quick smoke test before the real run
    python eval.py --allow-draft                # score unverified questions (marks report provisional)

The retrieval pass makes no LLM calls, so iterating on chunking, fusion weights
or rrf_k is free. Only --full spends money.

Output: reports/eval-<mode>.md, which is the artefact to open first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from copilot.config import REPORTS_DIR, settings
from copilot.evaluation.golden import corpus_doc_ids, load_golden, summarise, validate_against_corpus
from copilot.evaluation.report import build_markdown, timestamp
from copilot.evaluation.runner import EvalRun, run_answer_eval, run_retrieval_eval, snapshot_config

ALL_MODES = ("dense", "sparse", "hybrid", "rerank")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--strategy", choices=ALL_MODES, help="evaluate one retrieval mode only")
    parser.add_argument("--full", action="store_true", help="also run generation and citation checks")
    parser.add_argument("--answer-mode", choices=ALL_MODES, default="rerank")
    parser.add_argument("--top-k", type=int, default=settings.final_top_k)
    parser.add_argument("--limit", type=int, help="use only the first N questions")
    parser.add_argument("--allow-draft", action="store_true",
                        help="score unverified questions; the report is marked provisional")
    parser.add_argument("--no-verify-citations", action="store_true",
                        help="skip the citation judge during --full (cheaper, less informative)")
    parser.add_argument("--out", help="report path (default reports/eval-<mode>.md)")
    parser.add_argument("--json", action="store_true", help="also write the raw numbers as JSON")
    args = parser.parse_args(argv)

    golden = load_golden()

    # Ground truth pointing at documents that are not in the index can never be
    # matched, so it looks like a retrieval failure forever. Fail loudly.
    problems = validate_against_corpus(golden, corpus_doc_ids())
    hard = [p for p in problems if "not in corpus" in p or "duplicate" in p]
    if hard:
        print("golden set does not match the corpus:", file=sys.stderr)
        for problem in hard[:15]:
            print(f"  {problem}", file=sys.stderr)
        return 1

    scorable = golden.scorable()
    unverified = [q for q in golden.questions
                  if str(q.status) == "draft" and (q.should_refuse or q.expected_doc_ids)]

    if not scorable:
        if not args.allow_draft:
            print(
                f"No verified questions to score ({len(unverified)} drafts waiting).\n"
                "  Review them:  python review_questions.py\n"
                "  Then approve: python review_questions.py --verify-unflagged\n"
                "  Or score the drafts anyway with --allow-draft (report is marked provisional).",
                file=sys.stderr,
            )
            return 1
        scorable = unverified
        print(f"WARNING: scoring {len(scorable)} UNVERIFIED questions. Results are provisional.\n")

    if args.limit:
        scorable = scorable[: args.limit]

    modes = [args.strategy] if args.strategy else list(ALL_MODES)

    run = EvalRun(
        questions_total=len(golden.questions),
        questions_scored=len(scorable),
        questions_unverified=sum(1 for q in scorable if str(q.status) == "draft"),
        config=snapshot_config(),
        started_at=timestamp(),
        answer_mode=args.answer_mode if args.full else "",
    )

    started = time.time()

    print(f"retrieval pass  ({len(scorable)} questions, top-{args.top_k}, no LLM calls)")
    run.modes = run_retrieval_eval(golden, modes, top_k=args.top_k, questions=scorable)

    if args.full:
        answerable_and_refusals = scorable
        print(f"\nanswer pass     ({len(answerable_and_refusals)} questions, mode={args.answer_mode})")
        metrics, per_answer, usage = run_answer_eval(
            golden,
            args.answer_mode,
            top_k=args.top_k,
            questions=answerable_and_refusals,
            verify=not args.no_verify_citations,
        )
        run.answers = metrics
        run.per_answer = per_answer
        run.usage = usage

    run.seconds = time.time() - started

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.strategy or ("full" if args.full else "all-modes")
    out_path = args.out or (REPORTS_DIR / f"eval-{suffix}.md")

    markdown = build_markdown(run, golden_summary=summarise(golden))
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    if args.json:
        json_path = str(out_path).replace(".md", ".json")
        payload = {
            "started_at": run.started_at,
            "config": run.config,
            "questions_scored": run.questions_scored,
            "modes": {
                name: {
                    "hit_rate": r.retrieval.hit_rate,
                    "full_recall": r.retrieval.full_recall,
                    "mrr": r.retrieval.mrr,
                    "precision": r.retrieval.precision,
                    "by_category": r.retrieval.by_category,
                    "mean_latency_ms": r.mean_latency_ms,
                }
                for name, r in run.modes.items()
            },
        }
        if run.answers:
            payload["answers"] = run.answers.__dict__
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nraw numbers  {json_path}")

    print(f"\nreport       {out_path}")

    # Headline to stdout, so a terminal run is useful without opening the file.
    if run.modes:
        print("\n" + "=" * 62)
        base = run.modes.get("dense")
        for name, result in run.modes.items():
            m = result.retrieval
            delta = ""
            if base and name != "dense":
                diff = (m.hit_rate - base.retrieval.hit_rate) * 100
                delta = f"  ({diff:+.1f}pp vs dense)"
            print(f"  {name:<8} hit@{args.top_k} {m.hit_rate:6.1%}   MRR {m.mrr:.3f}{delta}")
        print("=" * 62)

    if run.answers:
        a = run.answers
        print(f"  refusal accuracy   {a.refusal_accuracy:.1%}  "
              f"({a.missed_refusals} missed, {a.false_refusals} false)")
        print(f"  citation support   {a.citation_support_rate:.1%}  "
              f"({a.unsupported_citations} unsupported of {a.total_citations})")
        print(f"  cost               ${run.usage.get('estimated_cost_usd', 0):.4f}")
        print("=" * 62)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
