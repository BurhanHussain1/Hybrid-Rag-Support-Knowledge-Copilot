"""Citation verification: does each cited source actually support its claim?

This is the part of the project that most RAG demos skip, and the part worth
building carefully.

The failure it catches is specific and common. Retrieval finds five roughly
relevant chunks. The model writes a fluent answer and sprinkles citations across
it. Every citation points at a real, topically related document - and one of the
sentences is not actually stated anywhere in the source it cites. The answer
looks impeccably sourced and contains a claim nobody wrote.

So after generating, we take each (claim, cited chunk) pair and ask: is this
claim stated in, or directly inferable from, this chunk?

Three design choices matter:

**1. The judge sees one pair at a time, in isolation.** It gets the claim and the
one chunk it cites - not the answer, not the other sources, not the question's
other citations. If it saw everything, it could rationalise a claim from a
neighbouring source and mark the wrong citation as supported. The question we are
asking is narrow on purpose: does *this* source support *this* claim?

**2. The prompt is biased toward "unsupported" under uncertainty.** A verifier
that waves things through is worse than no verifier, because it produces a
confidence number that looks meaningful and is not.

**3. A cheap lexical check runs first.** If a claim quotes an exact token - an
error code, a flag, a command - and that token does not appear in the chunk at
all, that is strong evidence without spending a model call. It does not decide
the verdict alone (paraphrase is legitimate), but it is recorded as a signal, and
it is free.

Known limitation, stated plainly: by default the judge is the same model that
wrote the answer, which makes it predisposed to agree with itself. `JUDGE_MODEL`
in .env points it at a different model. Step 6 should report which was used and
spot-check a sample by hand - a judge nobody has audited is just a second opinion
from the same source.
"""

from __future__ import annotations

import re

from copilot.config import settings
from copilot.generation.llm import LLMClient, get_llm
from copilot.generation.models import Citation

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["supported", "partial", "unsupported"],
            "description": (
                "supported: the source states this claim, or it follows directly from the source. "
                "partial: the source supports some of the claim but not all of it. "
                "unsupported: the source does not state this, or contradicts it."
            ),
        },
        "reason": {
            "type": "string",
            "description": "One sentence. Quote the relevant part of the source, or say what is missing.",
        },
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You check whether a source supports a claim. You are strict.

You will be shown ONE claim and ONE source passage. Decide whether the passage supports the claim.

- "supported" only if the passage states the claim, or the claim follows directly and unambiguously from it.
- "partial" if the passage supports part of the claim but leaves some of it unstated.
- "unsupported" if the passage does not state the claim, is merely on a related topic, or contradicts it.

Being on the same topic is NOT support. Sounding plausible is NOT support. If you are unsure, answer "unsupported".

Judge only against the passage you are given. Do not use outside knowledge, and do not assume other sources exist."""


# Tokens worth checking literally: error codes, flags, commands, identifiers.
# Deliberately narrow - matching ordinary words would flag every paraphrase.
_EXACT_TOKEN = re.compile(r"`([^`]{2,40})`|\b(--[a-z][a-z0-9-]{2,})\b|\b([A-Z][a-z]+(?:[A-Z][a-z]+){1,})\b")


def exact_tokens(text: str) -> list[str]:
    """Literal strings a claim quotes: `code spans`, --flags, CamelCaseNames."""
    found: list[str] = []
    for match in _EXACT_TOKEN.finditer(text):
        token = next((g for g in match.groups() if g), None)
        if token and token not in found:
            found.append(token)
    return found


def missing_tokens(claim: str, source_text: str) -> list[str]:
    """Exact tokens the claim quotes that do not appear in the source at all."""
    lowered = source_text.lower()
    return [t for t in exact_tokens(claim) if t.lower() not in lowered]


class CitationVerifier:
    """Judges whether each citation supports the claim attached to it."""

    def __init__(self, llm: LLMClient | None = None):
        # A separate client so judge calls are counted separately from generation
        # calls, and so JUDGE_MODEL can differ from LLM_MODEL.
        self.llm = llm or (
            LLMClient(model=settings.judge_model) if settings.judge_model else get_llm()
        )

    @property
    def uses_same_model_as_generator(self) -> bool:
        """Reported alongside results, because it qualifies how much they mean."""
        return not settings.judge_model or settings.judge_model == settings.llm_model

    def verify(self, citations: list[Citation]) -> list[Citation]:
        """Fill in `supported`, `verdict` and `verdict_reason` on each citation."""
        checked: list[Citation] = []

        for citation in citations:
            claim = citation.claim
            if not claim:
                # A citation with no sentence attached - usually a trailing [3]
                # after a list. Nothing to judge, so we say so rather than
                # guessing a verdict.
                citation.verdict = "unverifiable"
                citation.verdict_reason = "No claim sentence could be attached to this citation."
                citation.supported = None
                checked.append(citation)
                continue

            gaps = missing_tokens(claim, citation.text)

            payload, _ = self.llm.complete_json(
                JUDGE_SYSTEM,
                self._build_prompt(claim, citation, gaps),
                VERDICT_SCHEMA,
                schema_name="citation_verdict",
                max_tokens=200,
            )

            verdict = payload.get("verdict", "unsupported")
            citation.verdict = verdict
            citation.verdict_reason = payload.get("reason", "")
            # "partial" counts as not supported for scoring. A half-backed claim
            # presented as sourced is the exact problem this checks for.
            citation.supported = verdict == "supported"
            checked.append(citation)

        return checked

    @staticmethod
    def _build_prompt(claim: str, citation: Citation, gaps: list[str]) -> str:
        hint = ""
        if gaps:
            # Given as an observation, not an instruction. The judge still decides;
            # legitimate paraphrase should survive this.
            hint = (
                "\n\nNote: these exact terms appear in the claim but not in the passage: "
                + ", ".join(repr(g) for g in gaps)
                + ". That may be paraphrasing, or it may mean the claim is not supported."
            )

        return f"""CLAIM
{claim}

SOURCE PASSAGE ({citation.breadcrumb})
{citation.text.strip()}{hint}

Does the passage support the claim?"""


def support_rate(citations: list[Citation]) -> float:
    """Fraction of judged citations that were supported.

    Citations we could not judge are excluded from the denominator rather than
    counted as failures - punishing the score for our own inability to attach a
    claim would confuse two different problems.
    """
    judged = [c for c in citations if c.supported is not None]
    if not judged:
        return 0.0
    return sum(1 for c in judged if c.supported) / len(judged)
