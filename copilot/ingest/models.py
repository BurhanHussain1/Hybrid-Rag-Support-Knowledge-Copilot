"""Data shapes that flow through the ingestion pipeline.

Defining these as real typed objects instead of passing dictionaries around is a
deliberate choice. A dict lets you write `doc["titel"]` and find out three hours
later, from a wrong answer, that the title was never set. A model fails at the
moment the mistake is made, with the field name in the error.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


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
