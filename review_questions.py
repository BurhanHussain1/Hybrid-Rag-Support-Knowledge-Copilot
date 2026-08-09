#!/usr/bin/env python
"""Review and verify draft golden questions.

    python review_questions.py                      # screen everything, print a report
    python review_questions.py --show q058          # full detail on one question
    python review_questions.py --verify q001,q002   # mark as verified
    python review_questions.py --reject q058        # mark as rejected
    python review_questions.py --verify-unflagged   # accept everything the screen liked
    python review_questions.py --stats

WHY A SCREENING PASS EXISTS
---------------------------
Reading 61 questions carefully is slow, and most of them are fine. The screen
does the mechanical checks so a human only has to adjudicate the suspicious ones.

It checks three things, and only the first is certain:

  1. Does every expected_doc_id exist in the corpus? A question whose ground
     truth points at a missing document can never be answered and will look like
     a retrieval failure forever. This is a hard error.

  2. For no_answer questions: does retrieval actually find nothing? If the top
     result is a confident match, the question is probably answerable and the
     label is wrong. This is the most valuable check, because a mislabelled
     no_answer question punishes the system for being correct.

  3. Does the question copy distinctive wording from its source? If so retrieval
     becomes trivial and the question measures nothing.

The screen FLAGS, it does not decide. Marking a question verified is a human
action, deliberately.
"""

from __future__ import annotations

import argparse
import re
import sys

from copilot.evaluation.golden import (
    GOLDEN_PATH,
    VerificationStatus,
    corpus_doc_ids,
    load_golden,
    save_golden,
    summarise,
    validate_against_corpus,
)
from copilot.evaluation.models import QuestionCategory

# Above this retrieval score, a "no_answer" question probably does have an answer.
# Cross-encoder logits: ~2+ means the reranker thinks it is clearly relevant.
SUSPICIOUS_NO_ANSWER_SCORE = 2.0

_WORD = re.compile(r"[a-z0-9]{4,}")


def copied_phrase(question: str, source_text: str, length: int = 4) -> str | None:
    """The longest run of consecutive words the question lifts from the source.

    My first attempt measured what fraction of the question's words appear in the
    passage, and it was wrong in a way worth recording. "How do I import my
    Mattermost data into Zulip?" scored 100% - every word appears in a document
    titled "Import from Mattermost". But that is a perfectly natural question a
    real user would type. Naming the feature you are asking about is not cheating.

    Word overlap cannot distinguish "used the same vocabulary" from "copied the
    sentence". A shared run of consecutive words can: sharing four words in a row
    with the source is copying, while sharing four words scattered through a
    sentence is just English.
    """
    words = _WORD.findall(question.lower())
    if len(words) < length:
        return None

    source_words = _WORD.findall(source_text.lower())
    source_ngrams = {
        " ".join(source_words[i : i + length]) for i in range(len(source_words) - length + 1)
    }

    for i in range(len(words) - length + 1):
        phrase = " ".join(words[i : i + length])
        if phrase in source_ngrams:
            return phrase
    return None


def screen(golden, *, run_retrieval: bool = True) -> dict[str, list[str]]:
    """Return {question_id: [flags]} for anything that needs a human look."""
    flags: dict[str, list[str]] = {}

    def flag(qid: str, message: str) -> None:
        flags.setdefault(qid, []).append(message)

    print("checking ground truth against the corpus...")
    known = corpus_doc_ids()
    for problem in validate_against_corpus(golden, known):
        qid = problem.split(":")[0].strip()
        flag(qid, f"HARD  {problem}")

    if not run_retrieval:
        return flags

    from copilot.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    print("screening with live retrieval (this loads models once)...\n")

    for question in golden.questions:
        result = retriever.retrieve(question.question, mode="rerank", top_k=5)
        if not result.chunks:
            if not question.should_refuse:
                flag(question.id, "retrieval returned nothing for an answerable question")
            continue

        top = result.chunks[0]
        retrieved_docs = {c.doc_id for c in result.chunks}

        if question.should_refuse:
            if top.score >= SUSPICIOUS_NO_ANSWER_SCORE:
                flag(
                    question.id,
                    f"labelled no_answer but retrieval is confident ({top.score:.1f}): "
                    f"{top.breadcrumb[:60]}",
                )
        else:
            missing = [d for d in question.expected_doc_ids if d not in retrieved_docs]
            if len(missing) == len(question.expected_doc_ids):
                # Not necessarily a bad question - it may be a genuine retrieval
                # failure, which is exactly what we want to measure. Worth a look
                # either way, because it might also be an unanswerable question.
                flag(question.id, "none of the expected documents were retrieved (check the question is fair)")

            phrase = copied_phrase(question.question, top.text)
            if phrase:
                flag(question.id, f'question copies a phrase from the source: "{phrase}"')

        if question.category is QuestionCategory.MULTI_DOC and len(question.expected_doc_ids) < 2:
            flag(question.id, "multi_doc question has fewer than 2 expected documents")

    return flags


def show(golden, qid: str) -> int:
    question = next((q for q in golden.questions if q.id == qid), None)
    if question is None:
        print(f"no question with id {qid}", file=sys.stderr)
        return 1

    print(f"\n{question.id}  [{question.category}]  status={question.status}")
    print(f"question   {question.question}")
    print(f"refuse?    {question.should_refuse}")
    print(f"docs       {question.expected_doc_ids or '(none)'}")
    print(f"must say   {question.answer_must_mention or '(nothing specified)'}")
    print(f"notes      {question.notes}")

    from copilot.retrieval.hybrid import HybridRetriever

    result = HybridRetriever().retrieve(question.question, mode="rerank", top_k=5)
    print("\nwhat retrieval actually finds:")
    for hit in result.chunks:
        mark = "*" if hit.doc_id in question.expected_doc_ids else " "
        print(f" {mark} {hit.rank}. [{hit.score:6.2f}] {hit.breadcrumb[:66]}")
        print(f"       {hit.doc_id}")
    print("\n  * = one of this question's expected documents\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="review_questions.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", metavar="QID")
    parser.add_argument("--verify", metavar="QIDS", help="comma-separated ids to mark verified")
    parser.add_argument("--reject", metavar="QIDS", help="comma-separated ids to mark rejected")
    parser.add_argument("--verify-unflagged", action="store_true",
                        help="mark every draft the screen did not flag as verified")
    parser.add_argument("--no-retrieval", action="store_true", help="skip the live retrieval screen")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    golden = load_golden()

    if args.stats:
        print(summarise(golden))
        return 0

    if args.show:
        return show(golden, args.show)

    if args.verify or args.reject:
        changed = 0
        for qid in (args.verify or "").split(","):
            qid = qid.strip()
            for question in golden.questions:
                if question.id == qid:
                    question.status = VerificationStatus.VERIFIED
                    changed += 1
        for qid in (args.reject or "").split(","):
            qid = qid.strip()
            for question in golden.questions:
                if question.id == qid:
                    question.status = VerificationStatus.REJECTED
                    changed += 1
        save_golden(golden)
        print(f"updated {changed} question(s) in {GOLDEN_PATH.name}")
        print(summarise(golden))
        return 0

    flags = screen(golden, run_retrieval=not args.no_retrieval)

    clean = [q for q in golden.questions if q.id not in flags]
    print("=" * 92)
    print(f"SCREEN RESULT: {len(clean)} clean, {len(flags)} flagged, {len(golden.questions)} total")
    print("=" * 92)

    for question in golden.questions:
        if question.id not in flags:
            continue
        print(f"\n{question.id}  [{question.category}]  {question.question}")
        for message in flags[question.id]:
            print(f"    ! {message}")

    if args.verify_unflagged:
        changed = 0
        for question in clean:
            if question.status is VerificationStatus.DRAFT:
                question.status = VerificationStatus.VERIFIED
                changed += 1
        save_golden(golden)
        print(f"\nmarked {changed} unflagged question(s) verified")

    print(f"\n{summarise(golden)}")
    print("\nInspect any of them with:  python review_questions.py --show q058")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
