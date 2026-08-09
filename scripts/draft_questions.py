#!/usr/bin/env python
"""Draft candidate golden questions from the real corpus.

    python scripts/draft_questions.py                 # default mix, ~70 questions
    python scripts/draft_questions.py --simple 10 --no-answer 5
    python scripts/draft_questions.py --dry-run       # show what it would sample

Output goes to data/golden/questions.yaml with every question marked `draft`.
Nothing here is usable for scoring until a human reads it and marks it verified -
`eval.py` enforces that.

WHY DRAFTING IS NOT CHEATING, AND WHERE THE LINE IS
---------------------------------------------------
The questions are generated from chunks that are actually in the corpus, so their
ground truth is known by construction: we asked for a question about document X,
so the answer is in document X. That is a genuine labour saving and it is honest.

What it cannot do is decide whether a question is *fair* - whether a real support
user would ask it, whether it is answerable from that document alone, whether the
phrasing accidentally quotes the source so exactly that retrieval becomes trivial.
Those are judgement calls, and a set the model wrote and approved is a test the
system helped write. Hence: every question lands as `draft`, and a person promotes
it.

The categories are built differently on purpose:

  simple_lookup   one sampled chunk -> one question
  multi_doc       a chunk plus its nearest neighbour in a DIFFERENT document,
                  found with the vector index, so the pair is genuinely related
  ambiguous       a chunk plus a near neighbour from a DIFFERENT product, so the
                  question plausibly belongs to either
  outdated_trap   sampled only from documents over two years old
  no_answer       invented from topics adjacent to the corpus but absent from it
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copilot.config import settings  # noqa: E402
from copilot.evaluation.golden import (  # noqa: E402
    GOLDEN_PATH,
    GoldenQuestion,
    GoldenSet,
    save_golden,
    summarise,
)
from copilot.evaluation.models import QuestionCategory  # noqa: E402
from copilot.generation.llm import get_llm  # noqa: E402
from copilot.ingest.pipeline import load_chunks  # noqa: E402

SEED = 42  # fixed, so re-running samples the same chunks and diffs stay readable

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "A realistic support question."},
        "answer_must_mention": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-4 specific facts or exact tokens a correct answer contains.",
        },
        "notes": {"type": "string", "description": "One line: what makes this question hard or interesting."},
    },
    "required": ["question", "answer_must_mention", "notes"],
    "additionalProperties": False,
}

NO_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Realistic support questions that the described documentation does NOT cover.",
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}

SINGLE_SYSTEM = """You write evaluation questions for a documentation search system.

Given a passage, write ONE question that a real user would ask, whose answer is in that passage.

Rules:
- Sound like a person with a problem, not like the documentation. "why won't my pod start" beats "what causes Pending pod status".
- Do NOT copy distinctive phrases from the passage. If the question quotes the passage, retrieval becomes trivial and the test measures nothing.
- The question must be answerable from the passage alone.
- Keep it under 20 words."""

PAIR_SYSTEM = """You write evaluation questions for a documentation search system.

Given TWO passages from DIFFERENT documents, write ONE question that needs BOTH to answer fully.

Rules:
- A complete answer must draw on both passages. If either alone is sufficient, the question is wrong.
- Sound like a real user, not like documentation.
- Do NOT copy distinctive phrases from either passage.
- Keep it under 25 words."""

AMBIGUOUS_SYSTEM = """You write evaluation questions that test whether a search system handles ambiguity.

Given two passages from DIFFERENT products, write ONE question that could plausibly refer to either.

Rules:
- Do not name the product. The ambiguity is the point.
- It must be a question a confused user would genuinely type.
- A good system should either ask which product, or answer for both and say so.
- Keep it under 20 words."""

NO_ANSWER_SYSTEM = """You write evaluation questions that a documentation set CANNOT answer.

You will be told which products the documentation covers. Write questions that:
- sound completely plausible for a support assistant covering those products
- are close enough in topic that a naive search WILL retrieve something confident-looking
- are genuinely NOT answerable from that documentation - pricing not published, unrelated products, personal account specifics, future roadmap, legal advice

The point is to test whether the system refuses instead of assembling a plausible answer from loosely related pages. Keep each under 20 words."""


def passage(chunk) -> str:
    return f"{chunk.meta.title} > {' > '.join(chunk.heading_path)}\n\n{chunk.text[:1200]}"


def sample_chunks(chunks, count, *, rng, predicate=None, spread_sources=True):
    """Pick `count` chunks, spread across sources so one corpus cannot dominate."""
    pool = [c for c in chunks if predicate(c)] if predicate else list(chunks)
    if not pool:
        return []

    if not spread_sources:
        return rng.sample(pool, min(count, len(pool)))

    by_source = defaultdict(list)
    for chunk in pool:
        by_source[chunk.meta.source_name].append(chunk)

    picked = []
    sources = sorted(by_source)
    index = 0
    while len(picked) < count and any(by_source.values()):
        source = sources[index % len(sources)]
        index += 1
        bucket = by_source[source]
        if bucket:
            picked.append(bucket.pop(rng.randrange(len(bucket))))
    return picked


def neighbour_of(chunk, *, different_source: bool):
    """Nearest chunk from another document, via the vector index.

    Using real vector neighbours rather than random pairs is what makes multi-doc
    and ambiguous questions meaningful: the two passages are genuinely about
    related things, so a question spanning both is natural rather than contrived.
    """
    from copilot.retrieval.embedder import get_embedder
    from copilot.retrieval.vector_store import VectorStore

    store = VectorStore()
    vector = get_embedder().embed_query(chunk.text[:400])

    # 100, not 25. Cross-source neighbours are rare - most of a chunk's nearest
    # neighbours are its own document's siblings - so a shallow search returned
    # nothing for 7 of 10 ambiguous questions and the category came out short.
    for hit in store.search(vector, limit=100):
        payload = hit["payload"]
        if payload["doc_id"] == chunk.doc_id:
            continue
        if different_source and payload["source_name"] == chunk.meta.source_name:
            continue
        if not different_source and payload["source_name"] != chunk.meta.source_name:
            continue
        return payload
    return None


def draft_single(llm, chunk, category, next_id) -> GoldenQuestion | None:
    payload, _ = llm.complete_json(
        SINGLE_SYSTEM,
        f"PASSAGE\n{passage(chunk)}\n\nWrite one question.",
        QUESTION_SCHEMA,
        schema_name="draft_question",
        max_tokens=300,
    )
    return GoldenQuestion(
        id=next_id,
        question=payload["question"],
        category=category,
        expected_doc_ids=[chunk.doc_id],
        expected_chunk_ids=[chunk.chunk_id],
        answer_must_mention=payload.get("answer_must_mention", [])[:4],
        source_name=chunk.meta.source_name,
        notes=payload.get("notes", ""),
    )


def draft_pair(llm, chunk, other, category, system, next_id) -> GoldenQuestion | None:
    other_text = (
        f"{other.get('title')} > {other.get('section_heading') or ''}\n\n{other.get('text', '')[:1000]}"
    )
    payload, _ = llm.complete_json(
        system,
        f"PASSAGE A ({chunk.meta.source_name})\n{passage(chunk)}\n\n"
        f"PASSAGE B ({other.get('source_name')})\n{other_text}\n\nWrite one question.",
        QUESTION_SCHEMA,
        schema_name="draft_question",
        max_tokens=300,
    )
    return GoldenQuestion(
        id=next_id,
        question=payload["question"],
        category=category,
        expected_doc_ids=[chunk.doc_id, other["doc_id"]],
        expected_chunk_ids=[chunk.chunk_id, other["chunk_id"]],
        answer_must_mention=payload.get("answer_must_mention", [])[:4],
        source_name=None,
        notes=payload.get("notes", ""),
    )


# A no-answer candidate is only kept if retrieval finds nothing this convincing.
# Cross-encoder logits: above ~2 the reranker considers a chunk clearly relevant.
NO_ANSWER_MAX_SCORE = 2.0


def draft_no_answer(llm, count, start_number, *, oversample: int = 4) -> list[GoldenQuestion]:
    """Generate no-answer candidates, then keep only the ones that really are.

    The first version of this just asked the model what the corpus could not
    answer and trusted the reply. Ten of thirteen were wrong: the corpus does
    cover privacy compliance, refunds, startup pricing, Kubernetes security
    practices and organisation customisation. A mislabelled no-answer question is
    worse than a missing one - it punishes the system for correctly answering, so
    your refusal metric measures the labels rather than the assistant.

    The model cannot know what is in 26,000 chunks. Retrieval can. So we
    over-generate and then check each candidate against the real index, keeping
    only those where nothing convincing comes back.
    """
    from copilot.retrieval.hybrid import HybridRetriever

    wanted = count * oversample
    payload, _ = llm.complete_json(
        NO_ANSWER_SYSTEM,
        "The documentation covers: Kubernetes (concepts, tasks, kubectl reference), "
        "FastAPI (tutorials, API reference, release notes), PostHog (product docs, "
        "tutorials, and an internal company handbook covering policies, compensation, "
        "onboarding and support processes), and Zulip (help centre and API docs).\n\n"
        "Note the handbook is broad: pricing discussions, privacy compliance, refunds, "
        "security practices and support workflows ARE covered. Avoid those.\n\n"
        f"Write {wanted} questions this documentation cannot answer.",
        NO_ANSWER_SCHEMA,
        schema_name="no_answer_questions",
        max_tokens=1600,
    )

    candidates = payload.get("questions", [])
    print(f"  generated {len(candidates)} candidates, checking each against the index...")

    retriever = HybridRetriever()
    kept: list[GoldenQuestion] = []

    for text in candidates:
        if len(kept) >= count:
            break
        result = retriever.retrieve(text, mode="rerank", top_k=3)
        top_score = result.chunks[0].score if result.chunks else 0.0

        if top_score >= NO_ANSWER_MAX_SCORE:
            print(f"    reject ({top_score:5.1f})  {text[:62]}")
            continue

        print(f"    keep   ({top_score:5.1f})  {text[:62]}")
        kept.append(
            GoldenQuestion(
                id=f"q{start_number + len(kept):03d}",
                question=text,
                category=QuestionCategory.NO_ANSWER,
                expected_doc_ids=[],
                should_refuse=True,
                notes=(
                    f"Confirmed unanswerable: best retrieval score {top_score:.1f}, "
                    f"below the {NO_ANSWER_MAX_SCORE} relevance threshold."
                ),
            )
        )

    if len(kept) < count:
        print(f"  warning: only {len(kept)} of {count} candidates survived the check")

    return kept


def main() -> int:
    parser = argparse.ArgumentParser(prog="draft_questions.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--simple", type=int, default=25)
    parser.add_argument("--multi-doc", type=int, default=12)
    parser.add_argument("--ambiguous", type=int, default=10)
    parser.add_argument("--outdated", type=int, default=10)
    parser.add_argument("--no-answer", type=int, default=13)
    parser.add_argument("--strategy", default=settings.chunk_strategy)
    parser.add_argument("--dry-run", action="store_true", help="show the sample, call no LLM")
    parser.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    rng = random.Random(SEED)

    print("loading chunks...")
    chunks = list(load_chunks(args.strategy))
    # Very short chunks make poor questions - not enough substance to ask about.
    usable = [c for c in chunks if len(c.text) > 400]
    print(f"  {len(chunks)} chunks, {len(usable)} long enough to build questions from\n")

    simple = sample_chunks(usable, args.simple, rng=rng)
    multi = sample_chunks(usable, args.multi_doc, rng=rng)
    ambiguous = sample_chunks(usable, args.ambiguous, rng=rng)
    stale = sample_chunks(
        usable, args.outdated, rng=rng,
        predicate=lambda c: (c.meta.age_days or 0) > 730,
    )

    print(f"sampled: {len(simple)} simple, {len(multi)} multi-doc, "
          f"{len(ambiguous)} ambiguous, {len(stale)} stale, {args.no_answer} no-answer")

    if not stale:
        print("  warning: no documents over 730 days old were found in this chunk set")

    if args.dry_run:
        print("\n--- sample preview ---")
        for label, group in (("simple", simple), ("stale", stale)):
            for chunk in group[:3]:
                age = chunk.meta.age_days
                trail = " > ".join([chunk.meta.title, *chunk.heading_path])
                print(f"  [{label}] {chunk.meta.source_name:<12} age={age:<5} {trail[:56]}")
        return 0

    llm = get_llm()
    questions: list[GoldenQuestion] = []
    counter = 1

    print("\ndrafting simple lookups...")
    for chunk in simple:
        questions.append(draft_single(llm, chunk, QuestionCategory.SIMPLE_LOOKUP, f"q{counter:03d}"))
        counter += 1
        print(f"  {questions[-1].id}  {questions[-1].question[:70]}")

    print("\ndrafting multi-document questions...")
    for chunk in multi:
        other = neighbour_of(chunk, different_source=False)
        if not other:
            continue
        questions.append(draft_pair(llm, chunk, other, QuestionCategory.MULTI_DOC, PAIR_SYSTEM, f"q{counter:03d}"))
        counter += 1
        print(f"  {questions[-1].id}  {questions[-1].question[:70]}")

    print("\ndrafting ambiguous questions...")
    for chunk in ambiguous:
        other = neighbour_of(chunk, different_source=True)
        if not other:
            continue
        questions.append(draft_pair(llm, chunk, other, QuestionCategory.AMBIGUOUS, AMBIGUOUS_SYSTEM, f"q{counter:03d}"))
        counter += 1
        print(f"  {questions[-1].id}  {questions[-1].question[:70]}")

    print("\ndrafting outdated-document traps...")
    for chunk in stale:
        question = draft_single(llm, chunk, QuestionCategory.OUTDATED_TRAP, f"q{counter:03d}")
        question.notes = (
            f"Answer lives in a document last updated {chunk.meta.age_days} days ago. "
            "A good answer flags its age. " + question.notes
        )
        questions.append(question)
        counter += 1
        print(f"  {questions[-1].id}  {questions[-1].question[:70]}")

    print("\ndrafting no-answer questions...")
    for question in draft_no_answer(llm, args.no_answer, counter):
        questions.append(question)
        print(f"  {question.id}  {question.question[:70]}")

    golden = GoldenSet(
        created_at=datetime.now(timezone.utc).isoformat(),
        corpus_note=(
            f"Drafted from {len(chunks)} chunks (strategy={args.strategy}, "
            f"chunk_size={settings.chunk_size}). Ground truth is doc-level and "
            "survives chunking changes."
        ),
        questions=questions,
    )

    path = save_golden(golden, args.out)
    print(f"\nwrote {len(questions)} draft questions to {path}")
    print(f"cost: ${llm.estimated_cost_usd():.4f}\n")
    print(summarise(golden))
    print("\nNEXT: read them and mark the good ones verified.")
    print("      python review_questions.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
