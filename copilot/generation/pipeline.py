"""The full assistant: retrieve, answer, verify, score, then decide.

This assembles everything into the contract from Phase 1 of the brief - answer,
citations, confidence, and an explicit "what I could not verify" - and it owns one
more decision: whether to answer at all.

The refusal path is not an error path. A support assistant that answers every
question is not more useful than one that refuses sometimes; it is less useful,
because you can no longer tell its confident answers from its guesses. Three
things can trigger a refusal:

  1. Nothing was retrieved.
  2. The model reported that the sources do not answer the question.
  3. Confidence came out below MIN_CONFIDENCE.

In all three cases we still return the closest matching sections. "I don't know"
is honest but useless on its own; "I don't know, and here are the three nearest
pages" gives the user somewhere to go. That distinction is most of the difference
between a demo and something a support team would actually deploy.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from copilot.config import settings
from copilot.generation.answerer import Answerer
from copilot.generation.confidence import ConfidenceBreakdown, ConfidenceScorer
from copilot.generation.models import Citation
from copilot.generation.verifier import CitationVerifier
from copilot.retrieval.hybrid import HybridRetriever, Mode, RetrievalResult, get_retriever

REFUSAL_TEXT = (
    "I could not find enough information in the documentation to answer this confidently."
)


class NearestSection(BaseModel):
    """A pointer offered when the assistant refuses to answer."""

    breadcrumb: str
    chunk_id: str
    url: str | None = None
    source_name: str = ""
    score: float = 0.0


class CopilotAnswer(BaseModel):
    """The assistant's complete response. This is the API and dashboard contract."""

    question: str
    answered: bool = Field(description="False when the assistant refused")
    answer: str

    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    confidence_breakdown: ConfidenceBreakdown | None = None
    unverified: list[str] = Field(default_factory=list)

    nearest_sections: list[NearestSection] = Field(
        default_factory=list, description="Populated on refusal, so the user has somewhere to look"
    )

    mode: str = ""
    refusal_reason: str | None = None
    judge_shares_model_with_generator: bool = True

    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict = Field(default_factory=dict)

    def summary_line(self) -> str:
        state = "answered" if self.answered else "refused"
        supported = self.confidence_breakdown.citations_supported if self.confidence_breakdown else 0
        total = self.confidence_breakdown.citations_total if self.confidence_breakdown else 0
        return f"{state}  confidence {self.confidence:.2f}  citations {supported}/{total} verified"


class Copilot:
    """The end-to-end assistant."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever | None = None,
        answerer: Answerer | None = None,
        verifier: CitationVerifier | None = None,
        scorer: ConfidenceScorer | None = None,
    ):
        self.retriever = retriever or get_retriever()
        self.answerer = answerer or Answerer()
        self._verifier = verifier
        self.scorer = scorer or ConfidenceScorer()

    @property
    def verifier(self) -> CitationVerifier:
        if self._verifier is None:
            self._verifier = CitationVerifier()
        return self._verifier

    def ask(
        self,
        question: str,
        *,
        mode: Mode = "rerank",
        filters: dict | None = None,
        top_k: int | None = None,
        verify: bool = True,
    ) -> CopilotAnswer:
        timings: dict[str, float] = {}

        start = time.perf_counter()
        retrieval = self.retriever.retrieve(question, mode=mode, filters=filters, top_k=top_k)
        timings["retrieval"] = (time.perf_counter() - start) * 1000

        if not retrieval.chunks:
            return self._refuse(
                question, retrieval, mode, "nothing_retrieved", timings,
                extra="No documents in the corpus matched this question.",
            )

        start = time.perf_counter()
        generated = self.answerer.answer(question, retrieval.chunks)
        timings["generation"] = (time.perf_counter() - start) * 1000

        citations: list[Citation] = generated.citations
        if verify and citations:
            start = time.perf_counter()
            citations = self.verifier.verify(citations)
            timings["verification"] = (time.perf_counter() - start) * 1000

        breakdown = self.scorer.score(retrieval, generated, citations)

        # Unsupported citations are a gap the user must be told about. The model
        # does not know its citations failed - only the verifier does - so this
        # is added here rather than expected from generation.
        unverified = list(generated.unverified)
        for citation in citations:
            if citation.supported is False:
                unverified.append(
                    f"Source [{citation.label}] ({citation.breadcrumb}) does not support the claim "
                    f"it was cited for: {citation.verdict_reason}"
                )

        answered = generated.answerable and breakdown.confidence >= settings.min_confidence

        if not answered:
            reason = (
                "model_reported_no_answer" if not generated.answerable
                else "confidence_below_threshold"
            )
            return self._refuse(
                question, retrieval, mode, reason, timings,
                breakdown=breakdown,
                citations=citations,
                unverified=unverified,
            )

        return CopilotAnswer(
            question=question,
            answered=True,
            answer=generated.answer,
            citations=citations,
            confidence=breakdown.confidence,
            confidence_breakdown=breakdown,
            unverified=unverified,
            mode=mode,
            judge_shares_model_with_generator=self.verifier.uses_same_model_as_generator if verify else True,
            timings_ms=timings,
            usage=self.answerer.llm.usage_summary(),
        )

    def _refuse(
        self,
        question: str,
        retrieval: RetrievalResult,
        mode: str,
        reason: str,
        timings: dict[str, float],
        *,
        breakdown: ConfidenceBreakdown | None = None,
        citations: list[Citation] | None = None,
        unverified: list[str] | None = None,
        extra: str | None = None,
    ) -> CopilotAnswer:
        """Build a refusal that still gives the user something to act on."""
        message = REFUSAL_TEXT
        if extra:
            message = f"{message} {extra}"

        # Offered even though we are refusing: these are the best candidates we
        # found, and a human can judge them faster than they can re-search.
        nearest = [
            NearestSection(
                breadcrumb=chunk.breadcrumb or chunk.title,
                chunk_id=chunk.chunk_id,
                url=chunk.url,
                source_name=chunk.source_name,
                score=round(chunk.score, 4),
            )
            for chunk in retrieval.chunks[:3]
        ]

        return CopilotAnswer(
            question=question,
            answered=False,
            answer=message,
            citations=citations or [],
            confidence=breakdown.confidence if breakdown else 0.0,
            confidence_breakdown=breakdown,
            unverified=unverified or [],
            nearest_sections=nearest,
            mode=mode,
            refusal_reason=reason,
            timings_ms=timings,
            usage=self.answerer.llm.usage_summary(),
        )


_default: Copilot | None = None


def get_copilot() -> Copilot:
    global _default
    if _default is None:
        _default = Copilot()
    return _default
