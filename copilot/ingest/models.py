"""Data shapes that flow through the ingestion pipeline.

Defining these as real typed objects instead of passing dictionaries around is a
deliberate choice. A dict lets you write `doc["titel"]` and find out three hours
later, from a wrong answer, that the title was never set. A model fails at the
moment the mistake is made, with the field name in the error.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from copilot.ingest.metadata import AccessLevel, DocType


class DocumentMetadata(BaseModel):
    """The filterable, displayable facts about a document.

    Kept as its own model rather than loose fields on RawDocument because this
    exact object gets copied onto every chunk, stored in the Qdrant payload, and
    shown in the citation UI. One definition, one shape, everywhere.
    """

    source_name: str
    doc_type: DocType
    access_level: AccessLevel

    last_updated: datetime | None = None
    age_days: int | None = Field(
        default=None,
        description="Days since last change, computed at ingestion time",
    )

    url: str | None = Field(default=None, description="Public URL, so a human can check the citation")
    title: str = ""

    # Filled during chunking (Step 1.3): the nearest heading above the chunk.
    section_heading: str | None = None


class RawDocument(BaseModel):
    """One source file, loaded and normalized but not yet chunked.

    We keep `raw_text` alongside `clean_text` on purpose. When a retrieved chunk
    later looks mangled, you need to answer "was it broken on disk, or did my
    cleaner break it?" Without the original you are guessing.
    """

    doc_id: str = Field(description="Stable, human-readable ID, e.g. 'k8s-website/tasks/debug/debug-pods'")
    source_name: str = Field(description="Top-level corpus folder: fastapi, k8s-website, posthog, zulip")
    rel_path: str = Field(description="Path relative to data/raw, forward slashes")
    abs_path: Path

    file_type: str = Field(description="md, mdx, html, pdf, or txt")
    title: str | None = None
    frontmatter: dict = Field(default_factory=dict)

    raw_text: str = Field(description="Exactly what was on disk")
    clean_text: str = Field(description="Normalized text, headings preserved")

    # None until copilot.ingest.metadata.enrich() runs. Optional rather than
    # required so loading and enriching stay independently testable.
    meta: DocumentMetadata | None = None

    # PDFs only: character offset where each page starts, so a chunk can report
    # the page number it came from. Citations that say "page 14" are far more
    # checkable than citations that just name a file.
    page_offsets: list[tuple[int, int]] = Field(
        default_factory=list,
        description="(page_number, char_offset) pairs, PDFs only",
    )

    @property
    def char_count(self) -> int:
        return len(self.clean_text)

    def page_for_offset(self, offset: int) -> int | None:
        """Which PDF page does this character offset fall on?"""
        if not self.page_offsets:
            return None
        page = self.page_offsets[0][0]
        for page_no, start in self.page_offsets:
            if start > offset:
                break
            page = page_no
        return page


class Chunk(BaseModel):
    """One searchable piece of a document.

    This is the unit everything downstream works with: it gets embedded, indexed
    in BM25, retrieved, reranked, fed to the LLM, and cited by ID. Both indexes
    point at these same IDs, which is what keeps the fusion layer honest.
    """

    chunk_id: str = Field(description="Stable ID, e.g. 'k8s-website/tasks/debug/debug-pods#h3'")
    doc_id: str
    text: str

    strategy: str = Field(description="'heading' or 'fixed' - recorded so Step 6 can compare them")
    index: int = Field(description="Position within the document, 0-based")

    char_start: int = Field(description="Offset in the document's clean_text")
    char_end: int

    heading_path: list[str] = Field(
        default_factory=list,
        description="Breadcrumb of headings above this chunk, outermost first",
    )
    page: int | None = Field(default=None, description="PDF page number, if applicable")

    meta: DocumentMetadata

    @property
    def section_heading(self) -> str | None:
        """The nearest heading above this chunk - shown in citations."""
        return self.heading_path[-1] if self.heading_path else None

    @property
    def embedding_text(self) -> str:
        """What we actually embed - the chunk plus its breadcrumb.

        A chunk that reads "Click Manage channel, then Archive." is meaningless
        on its own. Prefixed with "Zulip help > Archive a channel", it becomes
        findable. This costs a few tokens and reliably improves retrieval, so the
        embedded text and the displayed text are deliberately different.
        """
        # Drop consecutive repeats. A document titled "Extending OpenAPI" whose
        # H1 is also "Extending OpenAPI" would otherwise embed the phrase twice,
        # which skews the vector toward whatever those words mean and away from
        # what the chunk actually says.
        parts: list[str] = []
        for piece in [self.meta.title, *self.heading_path]:
            if piece and (not parts or parts[-1].lower() != piece.lower()):
                parts.append(piece)

        trail = " > ".join(parts)
        return f"{trail}\n\n{self.text}" if trail else self.text

    @property
    def char_count(self) -> int:
        return len(self.text)
