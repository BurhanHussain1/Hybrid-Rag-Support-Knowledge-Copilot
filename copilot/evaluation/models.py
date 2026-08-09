"""The golden question set, and the shape of an evaluation run.

One design decision here matters more than everything else in this file.

**Ground truth is recorded at DOCUMENT level, not chunk level.**

The tempting thing is to record "the answer is in chunk
`k8s-website/.../debug-pods#h3`". It is precise, and it breaks the moment you
change anything about chunking. Chunk IDs are positional: `#h3` means "the fourth
heading section", so changing chunk_size from 800 to 600, or switching from the
heading strategy to fixed-size, renumbers everything. Every recorded answer would
silently point at the wrong text.

That would be fatal for this project specifically, because comparing chunking
strategies is one of the things we are here to measure. A ground truth that
changes when the thing under test changes measures nothing.

`doc_id` is derived from the file path, so it is stable across every chunking
strategy, chunk size and re-ingestion. "Did retrieval surface the right
*document*?" is answerable forever; "did it surface chunk #h3?" is answerable
until Tuesday.

`expected_chunk_ids` is kept as an optional convenience for debugging, and is
never used for scoring.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuestionCategory(StrEnum):
    """The five kinds of question the brief asks for.

    Reporting per category is the point. "Retrieval was 84%" is a number;
    "simple lookups 96%, multi-document 61%, refusals 90%" tells you what to fix
    next. An aggregate score hides exactly the information you need.
    """

    SIMPLE_LOOKUP = "simple_lookup"      # one fact, one document
    MULTI_DOC = "multi_doc"              # needs two or more documents
    AMBIGUOUS = "ambiguous"              # plausibly matches several products
    OUTDATED_TRAP = "outdated_trap"      # answer sits in a stale document
    NO_ANSWER = "no_answer"              # the corpus genuinely cannot answer it


class VerificationStatus(StrEnum):
    """Whether a human has confirmed this question is fair and answerable.

    Draft questions are LLM-generated and must not be scored: a set written by
    the same model family being evaluated is a test the system helped write.
    `eval.py` refuses to run on unverified questions unless explicitly forced,
    and the report always states how many were human-verified.
    """

    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


class GoldenQuestion(BaseModel):
    """One test case."""

    id: str = Field(description="Stable identifier, e.g. q001")
    question: str
    category: QuestionCategory

    # The scoring ground truth. Empty for no_answer questions.
    expected_doc_ids: list[str] = Field(
        default_factory=list,
        description="Documents that contain the answer. Stable across chunking changes.",
    )

    # Debugging aid only. Never used for scoring - see the module docstring.
    expected_chunk_ids: list[str] = Field(default_factory=list)

    should_refuse: bool = Field(
        default=False,
        description="True when the honest response is 'the documentation does not cover this'",
    )
    answer_must_mention: list[str] = Field(
        default_factory=list,
        description="Key facts or exact tokens a correct answer should contain",
    )
    answer_must_not_mention: list[str] = Field(
        default_factory=list,
        description="Traps: stale values or wrong-product facts a correct answer avoids",
    )

    source_name: str | None = Field(default=None, description="Which corpus source, for slicing results")
    notes: str = Field(default="", description="Why this question exists, or what makes it hard")
    status: VerificationStatus = VerificationStatus.DRAFT

    @property
    def is_scorable(self) -> bool:
        """A question is usable if a human has verified it and it has ground truth."""
        if self.status is not VerificationStatus.VERIFIED:
            return False
        return self.should_refuse or bool(self.expected_doc_ids)


class GoldenSet(BaseModel):
    """The whole question set, plus provenance."""

    version: str = "1"
    created_at: str = ""
    corpus_note: str = Field(
        default="",
        description="Which corpus snapshot these were written against",
    )
    questions: list[GoldenQuestion] = Field(default_factory=list)

    def verified(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.status is VerificationStatus.VERIFIED]

    def scorable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.is_scorable]

    def by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[str(q.category)] = counts.get(str(q.category), 0) + 1
        return counts

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[str(q.status)] = counts.get(str(q.status), 0) + 1
        return counts
