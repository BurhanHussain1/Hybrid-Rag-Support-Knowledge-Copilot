"""Turning an evaluation run into a Markdown report.

This file is the artefact reviewers open first, so it is written for a reader who
has not seen the code. That means:

  - the headline comparison at the top, not buried under methodology
  - the configuration that produced the numbers, because a number without its
    settings is not reproducible
  - the caveats stated in the report itself rather than left for someone to
    discover, including how many questions a human actually verified

A report that only shows good numbers is marketing. One that shows where the
system fails, and says how it was measured, is engineering - and it is far more
convincing to anyone who has built one of these before.
"""

from __future__ import annotations

from datetime import datetime, timezone

from copilot.evaluation.runner import EvalRun


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _delta(new: float, base: float) -> str:
    diff = (new - base) * 100
    if abs(diff) < 0.05:
        return "="
    return f"{diff:+.1f}pp"


def build_markdown(run: EvalRun, *, golden_summary: str = "") -> str:
    lines: list[str] = []
    add = lines.append

    add("# Retrieval and Answer Evaluation")
    add("")
    add(f"_Generated {run.started_at} · {run.seconds:.0f}s total_")
    add("")

    # ---- headline ---------------------------------------------------------
    if run.modes:
        baseline_name = "dense" if "dense" in run.modes else next(iter(run.modes))
        baseline = run.modes[baseline_name].retrieval
        best_name = max(run.modes, key=lambda m: run.modes[m].retrieval.hit_rate)
        best = run.modes[best_name].retrieval

        add("## Headline")
        add("")
        if best_name != baseline_name and best.hit_rate != baseline.hit_rate:
            add(
                f"**{best_name.capitalize()} retrieval found the correct source document for "
                f"{_pct(best.hit_rate)} of questions, against {_pct(baseline.hit_rate)} for "
                f"{baseline_name}-only** — a {_delta(best.hit_rate, baseline.hit_rate)} change "
                f"across {best.n} questions."
            )
        else:
            add(
                f"**{baseline_name.capitalize()} retrieval found the correct source document for "
                f"{_pct(baseline.hit_rate)} of {baseline.n} questions.**"
            )
        add("")

    # ---- caveats, stated up front ----------------------------------------
    add("## How to read this")
    add("")
    add(f"- **{run.questions_scored} questions scored** out of {run.questions_total} in the set.")
    if run.questions_unverified:
        add(
            f"- **{run.questions_unverified} questions were not human-verified.** "
            "These numbers are provisional until they are reviewed."
        )
    add("- Ground truth is at **document** level, not chunk level, so it survives re-chunking.")
    add(
        "- `hit@k` is the metric that matters most in practice: the model needs only one "
        "correct source to answer well."
    )
    add(
        "- `precision` is reported but not optimised for. A question is often answerable from "
        "documents the answer key does not list, so a \"wrong\" retrieval is frequently a valid "
        "unlisted source."
    )
    add("- Refusal questions are excluded from retrieval metrics (they have no expected document) "
        "and scored in the answer section instead.")
    add("")

    # ---- retrieval comparison --------------------------------------------
    if run.modes:
        top_k = run.config.get("final_top_k", 5)
        add(f"## Retrieval (top-{top_k})")
        add("")
        add("| mode | hit@k | full recall | MRR | precision | mean latency |")
        add("|---|---|---|---|---|---|")
        for name, result in run.modes.items():
            m = result.retrieval
            add(
                f"| {name} | **{_pct(m.hit_rate)}** | {_pct(m.full_recall)} | {m.mrr:.3f} | "
                f"{_pct(m.precision)} | {result.mean_latency_ms:.0f}ms |"
            )
        add("")
        add("**hit@k** — at least one correct document retrieved. "
            "**full recall** — every correct document retrieved (only differs for multi-document "
            "questions). **MRR** — 1/rank of the first correct document, so it rewards ranking the "
            "right thing *first*, not merely somewhere.")
        add("")

        # ---- per-category -------------------------------------------------
        categories = sorted({c for r in run.modes.values() for c in r.retrieval.by_category})
        if categories:
            add("### hit@k by question type")
            add("")
            counts = next(iter(run.modes.values())).retrieval.category_counts
            header = "| mode | " + " | ".join(f"{c} (n={counts.get(c, 0)})" for c in categories) + " |"
            add(header)
            add("|---" * (len(categories) + 1) + "|")
            for name, result in run.modes.items():
                cells = [_pct(result.retrieval.by_category.get(c, 0.0)) for c in categories]
                add(f"| {name} | " + " | ".join(cells) + " |")
            add("")
            add("This table is the useful one. An aggregate score hides which *kind* of question "
                "fails, which is the only thing that tells you what to fix next.")
            add("")

    # ---- answers ----------------------------------------------------------
    if run.answers:
        a = run.answers
        add(f"## Answer quality (mode: {run.answer_mode})")
        add("")
        add("| metric | value |")
        add("|---|---|")
        add(f"| questions | {a.n} |")
        add(f"| refusal accuracy | **{_pct(a.refusal_accuracy)}** |")
        add(f"| correct refusals | {a.correct_refusals} |")
        add(f"| missed refusals (answered when it should not) | {a.missed_refusals} |")
        add(f"| false refusals (refused when it could answer) | {a.false_refusals} |")
        add(f"| citation support rate | **{_pct(a.citation_support_rate)}** |")
        add(f"| unsupported citations caught | {a.unsupported_citations} of {a.total_citations} |")
        add(f"| key-fact coverage | {_pct(a.mention_coverage)} |")
        add(f"| forbidden mentions | {a.forbidden_mentions} |")
        add(f"| mean confidence when answering | {a.mean_confidence_answered:.2f} |")
        add(f"| mean confidence when refusing | {a.mean_confidence_refused:.2f} |")
        if a.errors:
            add(f"| errors | {a.errors} |")
        add("")
        add("**Refusal accuracy is scored in both directions on purpose.** A system that refuses "
            "everything scores perfectly on unanswerable questions and is useless, so the same "
            "metric has to punish refusing answerable ones.")
        add("")
        add("**Key-fact coverage is substring matching** and therefore a weak proxy — a correct "
            "answer phrased differently scores low. It is a signal, not a verdict.")
        add("")

        if run.per_answer:
            failures = [
                r for r in run.per_answer
                if r.error is None and (not r.refusal_correct or r.citations_unsupported)
            ]
            if failures:
                add("### Where it went wrong")
                add("")
                add("| question | type | expected | got | unsupported citations |")
                add("|---|---|---|---|---|")
                for r in failures[:20]:
                    expected = "refuse" if r.should_refuse else "answer"
                    got = "refused" if r.refused else "answered"
                    add(f"| {r.question_id} | {r.category} | {expected} | {got} | "
                        f"{r.citations_unsupported} |")
                add("")

    # ---- configuration ----------------------------------------------------
    add("## Configuration")
    add("")
    add("Every setting that could change these numbers:")
    add("")
    add("| setting | value |")
    add("|---|---|")
    for key, value in run.config.items():
        add(f"| `{key}` | {value} |")
    add("")

    if run.usage:
        add("## Cost")
        add("")
        add(f"{run.usage.get('calls', 0)} LLM calls · "
            f"{run.usage.get('prompt_tokens', 0):,} in + "
            f"{run.usage.get('completion_tokens', 0):,} out tokens · "
            f"**${run.usage.get('estimated_cost_usd', 0):.4f}**")
        add("")

    if golden_summary:
        add("## Question set")
        add("")
        add("```")
        add(golden_summary)
        add("```")
        add("")

    add("## Known limitations")
    add("")
    add("- The citation judge defaults to the same model that writes the answers, so it is "
        "predisposed to agree with itself. Set `JUDGE_MODEL` to a different model for an "
        "independent check, and spot-check a sample by hand.")
    add("- Questions were drafted with LLM assistance from real corpus chunks, then screened "
        "automatically and promoted by a human. Drafting from the corpus makes ground truth "
        "reliable; it does not make the questions representative of real user traffic.")
    add("- The ambiguous category is small, because the four corpora overlap little "
        "semantically. Its numbers carry correspondingly less weight.")
    add("")

    return "\n".join(lines)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
