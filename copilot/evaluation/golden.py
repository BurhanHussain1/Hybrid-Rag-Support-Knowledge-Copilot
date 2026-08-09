"""Loading, saving and sanity-checking the golden question set.

The set lives in `data/golden/questions.yaml` and IS committed to git, unlike
everything else under `data/`. It is small, it is hand-verified, and it is the
most expensive artefact in the project to reproduce - losing it would mean
rewriting 60 questions by hand. It is also what makes anyone else's run of this
repo comparable to yours.

YAML rather than JSON purely because a human edits this file. Comments and
readable multi-line strings matter more here than parser speed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from copilot.config import PROJECT_ROOT
from copilot.evaluation.models import GoldenQuestion, GoldenSet, VerificationStatus

GOLDEN_DIR = PROJECT_ROOT / "data" / "golden"
GOLDEN_PATH = GOLDEN_DIR / "questions.yaml"


def load_golden(path: Path | None = None) -> GoldenSet:
    import yaml

    path = path or GOLDEN_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "  Draft candidates first: python scripts/draft_questions.py"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return GoldenSet.model_validate(raw)


def save_golden(golden: GoldenSet, path: Path | None = None) -> Path:
    import yaml

    path = path or GOLDEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if not golden.created_at:
        golden.created_at = datetime.now(timezone.utc).isoformat()

    payload = golden.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    return path


def validate_against_corpus(golden: GoldenSet, known_doc_ids: set[str]) -> list[str]:
    """Check every expected_doc_id actually exists in the indexed corpus.

    This catches the quiet killer: a question whose ground truth points at a
    document that was renamed, dropped as too short, or never ingested. Retrieval
    can never match it, so the question scores zero forever and looks like a
    retrieval failure. It is a data-entry bug, and it is invisible without this
    check.
    """
    problems: list[str] = []

    for question in golden.questions:
        if question.should_refuse and question.expected_doc_ids:
            problems.append(
                f"{question.id}: marked should_refuse but also lists expected_doc_ids"
            )
        if not question.should_refuse and not question.expected_doc_ids:
            problems.append(f"{question.id}: has no expected_doc_ids and is not a refusal case")

        for doc_id in question.expected_doc_ids:
            if doc_id not in known_doc_ids:
                problems.append(f"{question.id}: expected_doc_id not in corpus -> {doc_id}")

    ids = [q.id for q in golden.questions]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        problems.append(f"duplicate question ids: {sorted(duplicates)}")

    return problems


def corpus_doc_ids(strategy: str | None = None) -> set[str]:
    """Every doc_id present in the current chunk file."""
    from copilot.config import settings
    from copilot.ingest.pipeline import load_chunks

    return {chunk.doc_id for chunk in load_chunks(strategy or settings.chunk_strategy)}


def summarise(golden: GoldenSet) -> str:
    lines = [
        f"questions   {len(golden.questions)}",
        f"verified    {len(golden.verified())}",
        f"scorable    {len(golden.scorable())}",
        "",
        "by category:",
    ]
    for category, count in sorted(golden.by_category().items()):
        lines.append(f"  {category:<16} {count:>3}")
    lines.append("")
    lines.append("by status:")
    for status, count in sorted(golden.status_counts().items()):
        lines.append(f"  {status:<16} {count:>3}")
    return "\n".join(lines)


def mark_all_verified(golden: GoldenSet) -> int:
    """Bulk-approve every draft. Convenience for 'I read them all and they are fine'."""
    changed = 0
    for question in golden.questions:
        if question.status is VerificationStatus.DRAFT:
            question.status = VerificationStatus.VERIFIED
            changed += 1
    return changed


def next_question_id(golden: GoldenSet) -> str:
    numbers = [int(q.id[1:]) for q in golden.questions if q.id.startswith("q") and q.id[1:].isdigit()]
    return f"q{max(numbers, default=0) + 1:03d}"


__all__ = [
    "GOLDEN_PATH",
    "GoldenQuestion",
    "GoldenSet",
    "VerificationStatus",
    "corpus_doc_ids",
    "load_golden",
    "mark_all_verified",
    "next_question_id",
    "save_golden",
    "summarise",
    "validate_against_corpus",
]
