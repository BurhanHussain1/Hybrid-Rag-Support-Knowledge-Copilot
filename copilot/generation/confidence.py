"""Confidence scoring: one number, plus the breakdown that produced it.

The brief asks for four inputs, and each one catches a different failure:

  retrieval strength     did we find anything that looked relevant at all?
  citation support rate  do the cited sources actually back the claims?
  grounding rate         is every sentence cited, or are some floating free?
  completeness           did the model admit to gaps it could not fill?

The breakdown is returned alongside the number, and that is not a nicety. "0.42"
tells a user nothing. "0.42, because only 2 of 4 citations held up" tells them
exactly how much to trust the answer and why. It is also what makes the number
debuggable: when confidence looks wrong, you can see which component was wrong.

One honest caveat, worth saying out loud rather than burying: these weights are
chosen by judgement, not fitted to data. They are a starting point that Step 6
can check - if refusals correlate badly with actual correctness on the golden set,
the weights are wrong and the evaluation will show it.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from copilot.config import settings
from copilot.generation.answerer import extract_labels, split_sentences
from copilot.generation.models import Citation, GeneratedAnswer
from copilot.retrieval.hybrid import RetrievalResult


class ConfidenceBreakdown(BaseModel):
    """The components behind a confidence score."""

    retrieval_strength: float = Field(description="0-1, how strong the top retrieval scores were")
    citation_support_rate: float = Field(description="0-1, fraction of citations that held up")
    grounding_rate: float = Field(description="0-1, fraction of answer sentences carrying a citation")
    completeness: float = Field(description="0-1, penalised by admitted gaps")
    staleness_penalty: float = Field(default=0.0, description="subtracted from the total")

    confidence: float = Field(description="0-1, the weighted result")

    citations_total: int = 0
    citations_supported: int = 0
    citations_unsupported: int = 0
    citations_unjudged: int = 0
    stale_citations: int = 0

    notes: list[str] = Field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"confidence {self.confidence:.2f}",
            f"  retrieval strength    {self.retrieval_strength:.2f}  x{settings.weight_retrieval}",
            f"  citation support      {self.citation_support_rate:.2f}  x{settings.weight_citation_support}"
            f"   ({self.citations_supported}/{self.citations_total} supported)",
            f"  grounding             {self.grounding_rate:.2f}  x{settings.weight_grounding}",
            f"  completeness          {self.completeness:.2f}  x{settings.weight_completeness}",
        ]
        if self.staleness_penalty:
            lines.append(f"  staleness penalty    -{self.staleness_penalty:.2f}"
                         f"   ({self.stale_citations} cited docs over 2 years old)")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def normalise_retrieval_score(result: RetrievalResult) -> float:
    """Map the top retrieval score onto 0-1, whichever mode produced it.

    This needs care because the modes produce incompatible numbers:

      dense   cosine similarity, already 0-1
      sparse  BM25, unbounded (we have seen 40+)
      fused   RRF sums, tiny (around 0.01-0.03)
      rerank  cross-encoder logits, roughly -11 to +11

    Using the raw number would make confidence mean something different in every
    mode, which would quietly invalidate the mode comparison in Step 6. So we
    normalise per mode, preferring the dense cosine when we have it because it is
    the only one that is natively a 0-1 similarity.
    """
    if not result.chunks:
        return 0.0

    top = result.chunks[0]

    if top.dense_score is not None:
        # Cosine on normalised vectors. In practice, relevant chunks land around
        # 0.7-0.9 and unrelated ones around 0.3-0.5, so we stretch that band to
        # use the full 0-1 range instead of compressing everything into the top third.
        return max(0.0, min(1.0, (top.dense_score - 0.35) / 0.5))

    if top.rerank_score is not None:
        # Logistic squash. Cross-encoder logits above ~2 mean "clearly relevant",
        # below ~-2 mean "clearly not".
        return 1.0 / (1.0 + math.exp(-top.rerank_score / 2.0))

    if top.sparse_score is not None:
        # BM25 has no ceiling, so this is a soft saturating curve rather than a
        # true normalisation. 20 is roughly "a good match" on this corpus.
        return min(1.0, top.sparse_score / 25.0)

    return min(1.0, max(0.0, top.score))


def grounding_rate(answer_text: str) -> float:
    """Fraction of substantive sentences that carry at least one citation.

    An uncited sentence in a grounded answer is either a fact with no source or
    connective prose. We cannot tell which automatically, so short sentences are
    excluded - "Here are the steps:" should not count against the score.
    """
    sentences = [s for s in split_sentences(answer_text) if len(s) > 40]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if extract_labels(s))
    return cited / len(sentences)


def completeness(generated: GeneratedAnswer) -> float:
    """1.0 for a complete answer, reduced by each admitted gap.

    Note the direction: admitting a gap *lowers* confidence but is the honest
    behaviour we asked for. The score reflects how much of the question was
    answered, not how well the model behaved.
    """
    if not generated.answerable:
        return 0.0
    gaps = len(generated.unverified)
    return max(0.0, 1.0 - 0.25 * gaps)


class ConfidenceScorer:
    def score(
        self,
        result: RetrievalResult,
        generated: GeneratedAnswer,
        citations: list[Citation] | None = None,
    ) -> ConfidenceBreakdown:
        citations = citations if citations is not None else generated.citations

        judged = [c for c in citations if c.supported is not None]
        supported = [c for c in judged if c.supported]
        unsupported = [c for c in judged if not c.supported]
        stale = [c for c in citations if c.is_stale]

        support = len(supported) / len(judged) if judged else 0.0
        retrieval = normalise_retrieval_score(result)
        grounding = grounding_rate(generated.answer)
        complete = completeness(generated)

        # Normalise the weights so they always sum to 1, which means someone can
        # change one weight in .env without silently rescaling the whole score.
        weights = [
            settings.weight_retrieval,
            settings.weight_citation_support,
            settings.weight_grounding,
            settings.weight_completeness,
        ]
        total_weight = sum(weights) or 1.0
        w_ret, w_sup, w_gnd, w_cmp = (w / total_weight for w in weights)

        raw = retrieval * w_ret + support * w_sup + grounding * w_gnd + complete * w_cmp

        penalty = 0.0
        if citations and len(stale) == len(citations):
            # Every source is old. Not proof of being wrong, but the user should
            # be told, and the number should reflect it.
            penalty = settings.staleness_penalty

        notes: list[str] = []
        if not generated.answerable:
            notes.append("model reported the sources do not answer this question")
        if unsupported:
            notes.append(f"{len(unsupported)} citation(s) did not support their claim")
        if judged and grounding < 0.6:
            notes.append("several sentences carry no citation")
        if citations and len(stale) == len(citations):
            notes.append("every cited document is over two years old")
        if not citations and generated.answerable:
            notes.append("the answer claims to be grounded but cites nothing")

        confidence = max(0.0, min(1.0, raw - penalty))

        return ConfidenceBreakdown(
            retrieval_strength=round(retrieval, 4),
            citation_support_rate=round(support, 4),
            grounding_rate=round(grounding, 4),
            completeness=round(complete, 4),
            staleness_penalty=round(penalty, 4),
            confidence=round(confidence, 4),
            citations_total=len(citations),
            citations_supported=len(supported),
            citations_unsupported=len(unsupported),
            citations_unjudged=len(citations) - len(judged),
            stale_citations=len(stale),
            notes=notes,
        )
