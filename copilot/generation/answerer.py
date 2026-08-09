"""Generate a grounded answer from retrieved chunks, and resolve its citations.

Two jobs:

  1. Ask the model for an answer that cites its sources by number.
  2. Parse those numbers back out and attach the real chunks.

Step 2 is where fabricated citations get caught. The model can only cite numbers
we gave it; anything else is invented, and we drop it and record that it happened.
Note the ordering - this check costs nothing and runs before the expensive
LLM-as-judge verification in Step 4.2. Cheap checks first.
"""

from __future__ import annotations

import re

from copilot.generation.llm import LLMClient, get_llm
from copilot.generation.models import Citation, GeneratedAnswer
from copilot.generation.prompts import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
)
from copilot.retrieval.models import RetrievedChunk

# Matches [1] and also [2][4] written together, which models do constantly.
_CITATION = re.compile(r"\[(\d{1,2})\]")


def extract_labels(text: str) -> list[int]:
    """Citation numbers in the order they appear, deduplicated."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        label = int(match.group(1))
        if label not in seen:
            seen.append(label)
    return seen


def split_sentences(text: str) -> list[str]:
    """Rough sentence split, used to attach each citation to what it supports.

    Deliberately not a real NLP sentence splitter. The verifier only needs the
    surrounding claim to judge, and a spaCy dependency for this would be a lot of
    weight for a heuristic that is already good enough on documentation prose.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def claim_for_label(answer_text: str, label: int) -> str | None:
    """The sentence a given citation was attached to."""
    for sentence in split_sentences(answer_text):
        if f"[{label}]" in sentence:
            # Strip the markers so the verifier judges the claim, not the notation.
            return _CITATION.sub("", sentence).strip()
    return None


class Answerer:
    """Turns retrieved chunks into a cited answer."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or get_llm()

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> tuple[str, str, dict]:
        """Assemble the prompt without calling the model. Used by --dry-run."""
        context, mapping = build_context(chunks)
        return SYSTEM_PROMPT, build_user_prompt(question, context), mapping

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
        if not chunks:
            # Nothing retrieved: there is nothing to ground an answer in, and
            # calling the model would only invite it to improvise.
            return GeneratedAnswer(
                answer="I could not find anything in the documentation related to this question.",
                answerable=False,
                unverified=["No relevant documentation was retrieved for this question."],
                model=self.llm.model,
            )

        system, user, mapping = self.build_prompt(question, chunks)
        payload, raw = self.llm.complete_json(
            system, user, ANSWER_SCHEMA, schema_name="grounded_answer"
        )

        answer_text = payload.get("answer", "")
        citations, fabricated = self._resolve_citations(answer_text, mapping)

        unverified = list(payload.get("unverified") or [])
        if fabricated:
            # Surfaced to the user rather than silently swallowed. A model that
            # invents source numbers is a fact worth reporting.
            unverified.append(
                f"The answer referenced source numbers that do not exist: "
                f"{', '.join(f'[{n}]' for n in fabricated)}."
            )

        return GeneratedAnswer(
            answer=answer_text,
            answerable=bool(payload.get("answerable", False)),
            unverified=unverified,
            citations=citations,
            model=self.llm.model,
            raw_response=raw,
        )

    @staticmethod
    def _resolve_citations(
        answer_text: str, mapping: dict[int, RetrievedChunk]
    ) -> tuple[list[Citation], list[int]]:
        """Map [n] markers back to chunks. Returns (citations, fabricated labels)."""
        citations: list[Citation] = []
        fabricated: list[int] = []

        for label in extract_labels(answer_text):
            chunk = mapping.get(label)
            if chunk is None:
                fabricated.append(label)
                continue

            citations.append(
                Citation(
                    label=label,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    title=chunk.title,
                    section_heading=chunk.section_heading,
                    url=chunk.url,
                    source_name=chunk.source_name,
                    doc_type=chunk.doc_type,
                    age_days=chunk.age_days,
                    retrieval_score=chunk.score,
                    claim=claim_for_label(answer_text, label),
                )
            )

        return citations, fabricated
