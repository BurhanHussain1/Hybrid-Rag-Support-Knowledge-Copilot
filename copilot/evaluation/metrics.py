"""Evaluation metrics, defined precisely.

Retrieval and answer quality are measured separately, and that separation is the
whole point. If you only score final answers, a wrong answer tells you nothing
about *why*: did retrieval fail to find the document, or did the model have the
right document and misread it? Those need completely different fixes.

Retrieval metrics answer "did we put the right document in front of the model?"
Answer metrics answer "given that, did it do the right thing?"

Every metric here is computed at DOCUMENT level, never chunk level - see the
docstring in models.py for why chunk IDs are unusable as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from copilot.evaluation.models import GoldenQuestion


@dataclass
class QuestionRetrieval:
    """What retrieval did for one question."""

    question_id: str
    category: str
    expected_docs: list[str]
    retrieved_docs: list[str]  # in rank order, deduplicated

    @property
    def hit(self) -> bool:
        """Did we retrieve at least one document that contains the answer?

        This is the headline number, and it is the one that matters most in
        practice: the model only needs one good source to answer correctly.
        """
        return any(doc in self.retrieved_docs for doc in self.expected_docs)

    @property
    def full_hit(self) -> bool:
        """Did we retrieve ALL the documents needed?

        Only meaningfully different from `hit` for multi-document questions,
        which is exactly why those are their own category. A system that always
        finds one of the two needed sources looks fine on `hit` and gives
        half-answers in production.
        """
        return bool(self.expected_docs) and all(
            doc in self.retrieved_docs for doc in self.expected_docs
        )

    @property
    def first_hit_rank(self) -> int | None:
        """1-based rank of the first correct document, or None if absent."""
        for rank, doc in enumerate(self.retrieved_docs, start=1):
            if doc in self.expected_docs:
                return rank
        return None

    @property
    def reciprocal_rank(self) -> float:
        """1/rank of the first correct document. 0 if never retrieved.

        Rewards putting the right document *high*, not merely somewhere in the
        list. Rank 1 scores 1.0, rank 5 scores 0.2. That matters because only the
        top few chunks reach the model - a correct document at rank 18 is a miss
        in every way that counts.
        """
        rank = self.first_hit_rank
        return 1.0 / rank if rank else 0.0

    @property
    def precision(self) -> float:
        """Fraction of retrieved documents that were expected.

        Reported but not optimised for. Low precision is often fine here: a
        question may be answerable from several documents we never listed, so a
        "wrong" retrieval is frequently just an unlisted-but-valid source. Treat
        it as a signal about noise, not a target.
        """
        if not self.retrieved_docs:
            return 0.0
        return sum(1 for d in self.retrieved_docs if d in self.expected_docs) / len(self.retrieved_docs)


@dataclass
class RetrievalMetrics:
    """Aggregated retrieval performance over a set of questions."""

    n: int = 0
    hit_rate: float = 0.0
    full_recall: float = 0.0
    mrr: float = 0.0
    precision: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    category_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def compute(cls, results: list[QuestionRetrieval]) -> "RetrievalMetrics":
        # Refusal questions have no expected documents, so retrieval accuracy is
        # undefined for them. Including them would silently drag every number
        # toward zero and make hybrid look worse than it is.
        scored = [r for r in results if r.expected_docs]
        if not scored:
            return cls()

        by_cat: dict[str, list[QuestionRetrieval]] = {}
        for result in scored:
            by_cat.setdefault(result.category, []).append(result)

        return cls(
            n=len(scored),
            hit_rate=sum(r.hit for r in scored) / len(scored),
            full_recall=sum(r.full_hit for r in scored) / len(scored),
            mrr=sum(r.reciprocal_rank for r in scored) / len(scored),
            precision=sum(r.precision for r in scored) / len(scored),
            by_category={
                cat: sum(r.hit for r in group) / len(group) for cat, group in by_cat.items()
            },
            category_counts={cat: len(group) for cat, group in by_cat.items()},
        )


@dataclass
class QuestionAnswer:
    """What the full pipeline did for one question."""

    question_id: str
    category: str
    should_refuse: bool
    refused: bool
    confidence: float

    citations_total: int = 0
    citations_supported: int = 0
    citations_unsupported: int = 0

    mentions_expected: int = 0
    mentions_required: int = 0
    mentions_forbidden_found: int = 0

    answer_text: str = ""
    error: str | None = None

    @property
    def refusal_correct(self) -> bool:
        """Did the assistant refuse exactly when it should have?

        Scored both directions on purpose. A system that refuses everything gets
        a perfect score on unanswerable questions and is worthless - so the same
        metric has to punish refusing answerable ones.
        """
        return self.refused == self.should_refuse

    @property
    def citation_support_rate(self) -> float:
        if not self.citations_total:
            return 0.0
        return self.citations_supported / self.citations_total

    @property
    def mention_coverage(self) -> float:
        """Fraction of the expected key facts that appear in the answer.

        A crude proxy for correctness - it is substring matching, so a correct
        answer phrased differently scores low. Reported as a signal, never as a
        verdict, and paired with the citation metrics which are more trustworthy.
        """
        if not self.mentions_required:
            return 0.0
        return self.mentions_expected / self.mentions_required


@dataclass
class AnswerMetrics:
    """Aggregated answer quality."""

    n: int = 0
    refusal_accuracy: float = 0.0
    correct_refusals: int = 0
    missed_refusals: int = 0        # should have refused, answered anyway
    false_refusals: int = 0         # could have answered, refused
    citation_support_rate: float = 0.0
    unsupported_citations: int = 0
    total_citations: int = 0
    mention_coverage: float = 0.0
    forbidden_mentions: int = 0
    mean_confidence_answered: float = 0.0
    mean_confidence_refused: float = 0.0
    errors: int = 0

    @classmethod
    def compute(cls, results: list[QuestionAnswer]) -> "AnswerMetrics":
        usable = [r for r in results if r.error is None]
        if not usable:
            return cls(errors=len(results))

        answered = [r for r in usable if not r.refused]
        refused = [r for r in usable if r.refused]
        with_citations = [r for r in usable if r.citations_total]
        with_mentions = [r for r in usable if r.mentions_required]

        return cls(
            n=len(usable),
            refusal_accuracy=sum(r.refusal_correct for r in usable) / len(usable),
            correct_refusals=sum(1 for r in usable if r.should_refuse and r.refused),
            missed_refusals=sum(1 for r in usable if r.should_refuse and not r.refused),
            false_refusals=sum(1 for r in usable if not r.should_refuse and r.refused),
            citation_support_rate=(
                sum(r.citation_support_rate for r in with_citations) / len(with_citations)
                if with_citations else 0.0
            ),
            unsupported_citations=sum(r.citations_unsupported for r in usable),
            total_citations=sum(r.citations_total for r in usable),
            mention_coverage=(
                sum(r.mention_coverage for r in with_mentions) / len(with_mentions)
                if with_mentions else 0.0
            ),
            forbidden_mentions=sum(r.mentions_forbidden_found for r in usable),
            mean_confidence_answered=(
                sum(r.confidence for r in answered) / len(answered) if answered else 0.0
            ),
            mean_confidence_refused=(
                sum(r.confidence for r in refused) / len(refused) if refused else 0.0
            ),
            errors=len(results) - len(usable),
        )


def score_retrieval(question: GoldenQuestion, retrieved_doc_ids: list[str]) -> QuestionRetrieval:
    """Build a per-question retrieval result, preserving rank order."""
    seen: list[str] = []
    for doc_id in retrieved_doc_ids:
        if doc_id not in seen:
            seen.append(doc_id)

    return QuestionRetrieval(
        question_id=question.id,
        category=str(question.category),
        expected_docs=list(question.expected_doc_ids),
        retrieved_docs=seen,
    )


def count_mentions(answer_text: str, phrases: list[str]) -> int:
    """How many expected phrases appear in the answer, case-insensitively."""
    lowered = answer_text.lower()
    return sum(1 for phrase in phrases if phrase.lower() in lowered)
