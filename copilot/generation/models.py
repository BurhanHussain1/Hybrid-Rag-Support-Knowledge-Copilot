"""The assistant's output contract.

The project brief specifies four things every response must carry: an answer,
source citations, a confidence score, and an explicit "what I could not verify"
section. These models are that contract, written down.

Making it a typed contract rather than a free-form string matters because the
API (Step 5), the dashboard (Step 7) and the evaluation harness (Step 6) all
consume it. If the shape is implicit, each of them invents its own parsing and
they drift apart.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field


class Citation(BaseModel):
    """One source the answer pointed at, plus whether it actually backs the claim."""

    label: int = Field(description="The [n] marker used in the answer text")
    chunk_id: str
    text: str = Field(default="", description="The cited chunk's text, for the verifier and the UI")

    title: str = ""
    section_heading: str | None = None
    url: str | None = None
    source_name: str = ""
    doc_type: str = ""
    age_days: int | None = None

    retrieval_score: float | None = None

    # Filled by the verifier in Step 4.2. None means "not yet checked", which is
    # deliberately different from False ("checked and it does not support this").
    supported: bool | None = None
    verdict: str | None = Field(default=None, description="supported, partial, or unsupported")
    verdict_reason: str | None = None
    claim: str | None = Field(default=None, description="The sentence this citation was attached to")

    # @computed_field, not a plain @property.
    #
    # A plain @property works fine inside Python and vanishes from the JSON:
    # pydantic serialises declared fields only. The API returned citations with no
    # breadcrumb and no staleness flag, which the dashboard in Step 7 needs, and
    # nothing errored - the keys were simply absent.
    #
    # @computed_field includes them in model_dump()/model_dump_json() and in the
    # OpenAPI schema, while still being derived rather than stored.

    @computed_field  # type: ignore[prop-decorator]
    @property
    def breadcrumb(self) -> str:
        parts = [p for p in (self.title, self.section_heading) if p]
        seen: list[str] = []
        for part in parts:
            if not seen or seen[-1].lower() != part.lower():
                seen.append(part)
        return " > ".join(seen)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_stale(self) -> bool:
        """Older than two years. A signal, not a verdict - old docs can be correct."""
        return self.age_days is not None and self.age_days > 730


class GeneratedAnswer(BaseModel):
    """Raw output of the generation step, before verification and scoring."""

    answer: str = Field(description="Answer text with inline [n] citation markers")
    answerable: bool = Field(description="Did the model believe the context contained an answer?")
    unverified: list[str] = Field(
        default_factory=list,
        description="Things the model could not confirm from the provided context",
    )
    citations: list[Citation] = Field(default_factory=list)

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_response: str = ""

    @property
    def cited_labels(self) -> set[int]:
        return {c.label for c in self.citations}
