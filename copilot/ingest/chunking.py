"""Chunking: cut documents into searchable pieces.

Why chunk at all? Two reasons.

  1. A whole document is too big to embed usefully. One vector for a 7,000
     character page averages away everything specific in it, so a question about
     one paragraph matches weakly against the blur of the whole page.
  2. The LLM only gets a handful of pieces. Send whole documents and you burn the
     context window on text nobody asked about.

Two strategies are implemented, and every chunk records which one made it, so
Step 6 can compare them on the same questions instead of guessing.

  heading  Split at markdown headings. Boundaries land where the author already
           decided one topic ends and the next begins. Sections vary wildly in
           size, so anything oversized gets packed down further.

  fixed    Slide a fixed-size window with overlap. Predictable sizes, works on
           documents with no headings, but cuts through the middle of ideas -
           which is exactly what the overlap is there to patch.

Preview any document's chunks:

    python -m copilot.ingest.chunking data/raw/k8s-website/.../debug-pods.md heading
"""

from __future__ import annotations

import re

from copilot.config import settings
from copilot.ingest.models import Chunk, RawDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------

def split_blocks(text: str) -> list[str]:
    """Split into paragraphs, treating fenced code as one unbreakable block.

    Splitting naively on blank lines would cut a code block in half at the first
    empty line inside it. Half a code block is worse than useless in an answer:
    it looks like a complete command, and it isn't.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current.append(line)
            continue

        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    return [b for b in blocks if b]


def force_split(block: str, target: int) -> list[str]:
    """Split a single oversized block that has no blank lines to split on.

    This exists because of a bug found by measuring rather than assuming. Big
    markdown tables - PostHog's environment-variable reference is 65 rows and
    29,686 characters - contain no blank lines at all, so they arrive here as one
    indivisible block.

    Leaving them whole is not an option: bge-small truncates at 512 tokens
    (~2,000 characters), so everything past that would be silently dropped from
    the embedding. The chunk would exist, look fine, and be unfindable.

    Three cases, each split so the pieces stay valid on their own:
      table  keep the header row on every piece, or later rows lose their column
             names and become meaningless
      code   re-open and re-close the fence, so each piece is still a code block
      other  split on line boundaries
    """
    lines = block.split("\n")

    is_table = lines[0].lstrip().startswith("|") and len(lines) > 2
    header: list[str] = []
    if is_table:
        header = lines[:2]  # header row + the |---|---| separator
        lines = lines[2:]

    fence = ""
    if lines and lines[0].lstrip().startswith("```"):
        fence = lines[0].strip()
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]

    pieces: list[str] = []
    current: list[str] = []

    # Repeating the header eats into the budget for actual rows. PostHog's
    # environment-variable table pads its header to ~800 characters, which left
    # a budget of ~2 and shattered the table into two-character fragments. So:
    # if the header would consume more than half the budget, drop it, and never
    # let the budget fall below a floor.
    header_len = sum(len(h) + 1 for h in header)
    if header_len > target // 2:
        header = []
        header_len = 0
    budget = max(settings.min_chunk_chars, target - header_len)

    def flush() -> None:
        if not current:
            return
        body = "\n".join(header + current) if header else "\n".join(current)
        if fence:
            body = f"{fence}\n{body}\n```"
        pieces.append(body)

    for line in lines:
        # Last resort: a single line longer than the whole budget. Minified JSON
        # and one-line config blobs do this - there is no newline to split on, so
        # we cut by character count. Ugly, but an un-embeddable chunk is worse
        # than an inelegantly cut one.
        if len(line) > budget:
            flush()
            current = []
            for i in range(0, len(line), budget):
                pieces.append(line[i : i + budget])
            continue

        if current and sum(len(x) + 1 for x in current) + len(line) > budget:
            flush()
            current = []
        current.append(line)
    flush()

    return pieces or [block[:target]]


def pack_blocks(blocks: list[str], target: int, ceiling: int) -> list[str]:
    """Greedily combine blocks up to `target` characters.

    Greedy is the right call here: it keeps neighbouring paragraphs together and
    never splits a block that fits. Anything over the ceiling goes to
    force_split rather than being emitted whole.
    """
    packed: list[str] = []
    current = ""

    for block in blocks:
        if len(block) > ceiling:
            if current:
                packed.append(current)
                current = ""
            packed.extend(force_split(block, target))
            continue

        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > target and current:
            packed.append(current)
            current = block
        else:
            current = candidate

    if current:
        packed.append(current)

    return packed


# ---------------------------------------------------------------------------
# Strategy 1: heading-based
# ---------------------------------------------------------------------------

def _sections(text: str) -> list[tuple[list[str], int, int]]:
    """Return (heading_path, start, end) for each section of the document.

    `heading_path` is a breadcrumb: an H3 under an H2 under an H1 yields all
    three. That trail is what makes a chunk understandable on its own - "Archive"
    means nothing; "Zulip help > Channels > Archive" means something.
    """
    matches = list(_HEADING.finditer(text))

    if not matches:
        return [([], 0, len(text))] if text.strip() else []

    sections: list[tuple[list[str], int, int]] = []

    # Text before the first heading (an intro paragraph) is real content.
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        sections.append(([], 0, matches[0].start()))

    stack: list[tuple[int, str]] = []  # (level, heading text)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()

        # Pop deeper-or-equal headings: an H2 ends any open H3s beneath it.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading))

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        if text[start:end].strip():
            sections.append(([h for _, h in stack], start, end))

    return sections


def chunk_by_heading(doc: RawDocument) -> list[tuple[str, int, int, list[str]]]:
    """Yield (text, char_start, char_end, heading_path) tuples."""
    out: list[tuple[str, int, int, list[str]]] = []
    text = doc.clean_text

    for heading_path, start, end in _sections(text):
        section = text[start:end].strip()
        if not section:
            continue

        # Offsets must point at the real document, not the stripped copy, or the
        # PDF page lookup and any future highlight feature will be off by a bit.
        offset = start + (end - start - len(text[start:end].lstrip()))

        if len(section) <= settings.max_chunk_chars:
            out.append((section, offset, offset + len(section), heading_path))
            continue

        # Section too long: pack its paragraphs down to target size.
        cursor = 0
        for piece in pack_blocks(split_blocks(section), settings.chunk_size, settings.max_chunk_chars):
            local = section.find(piece, cursor)
            if local == -1:  # packing joined blocks with "\n\n"; fall back
                local = cursor
            out.append((piece, offset + local, offset + local + len(piece), heading_path))
            cursor = local + len(piece)

    return out


# ---------------------------------------------------------------------------
# Strategy 2: fixed-size with overlap
# ---------------------------------------------------------------------------

def chunk_fixed(doc: RawDocument) -> list[tuple[str, int, int, list[str]]]:
    """Sliding window of `chunk_size` characters, stepping by size - overlap.

    Overlap exists to survive bad cuts. Without it, a window boundary landing
    mid-sentence means neither chunk contains the complete thought, and neither
    matches the question. With 150 characters of overlap, the sentence appears
    whole in at least one of them.

    The cost is duplication: the same sentence sits in two chunks, so the same
    fact can be retrieved twice. That is a real tradeoff, and it is one of the
    things Step 6 measures rather than assumes.
    """
    text = doc.clean_text
    size = settings.chunk_size
    step = max(1, size - settings.chunk_overlap)

    out: list[tuple[str, int, int, list[str]]] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        # Never cut mid-word: back off to the last whitespace, but only a little.
        # Searching too far back would shrink chunks unpredictably.
        if end < len(text):
            window = text.rfind(" ", start + int(size * 0.7), end)
            newline = text.rfind("\n", start + int(size * 0.7), end)
            boundary = max(window, newline)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            out.append((piece, start, end, _heading_at(text, start)))

        if end >= len(text):
            break
        start += step

    return out


def _heading_at(text: str, offset: int) -> list[str]:
    """Rebuild the heading breadcrumb in force at a character offset.

    Fixed-size chunks ignore structure, but citations still need to say which
    section a chunk came from. So we look backwards for the headings above it.
    """
    stack: list[tuple[int, str]] = []
    for match in _HEADING.finditer(text, 0, offset):
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
    return [h for _, h in stack]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_PREFIX = {"heading": "h", "fixed": "f"}


def chunk_document(doc: RawDocument, strategy: str | None = None) -> list[Chunk]:
    """Chunk one document. Requires metadata to already be attached."""
    strategy = strategy or settings.chunk_strategy
    if doc.meta is None:
        raise ValueError(f"{doc.doc_id} has no metadata - run copilot.ingest.metadata.enrich first")

    raw_pieces = chunk_by_heading(doc) if strategy == "heading" else chunk_fixed(doc)

    # Rescue undersized chunks by merging them backwards instead of deleting them.
    # A 60-character tail like "See also: Debug Services" is useless alone but
    # fine attached to the paragraph above it. Dropping content should be the
    # last resort, not the first.
    merged: list[tuple[str, int, int, list[str]]] = []
    for piece in raw_pieces:
        text, start, end, path = piece
        if len(text) < settings.min_chunk_chars and merged and merged[-1][3] == path:
            prev_text, prev_start, _, prev_path = merged[-1]
            merged[-1] = (f"{prev_text}\n\n{text}", prev_start, end, prev_path)
        else:
            merged.append(piece)

    chunks: list[Chunk] = []
    for i, (text, start, end, path) in enumerate(merged):
        # What survives here is a whole tiny document - a Zulip MDX partial, a
        # stub page. There is nothing to merge it into, and on its own it is a
        # context-free scrap that can outrank real answers on short queries.
        if len(text) < settings.min_chunk_chars:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}#{_PREFIX[strategy]}{i}",
                doc_id=doc.doc_id,
                text=text,
                strategy=strategy,
                index=i,
                char_start=start,
                char_end=end,
                heading_path=path,
                page=doc.page_for_offset(start),
                meta=doc.meta.model_copy(update={"section_heading": path[-1] if path else None}),
            )
        )

    return chunks


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from copilot.config import RAW_DIR
    from copilot.ingest.loaders import load_document
    from copilot.ingest.metadata import GitDateIndex, enrich

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    target = Path(sys.argv[1]).resolve()
    how = sys.argv[2] if len(sys.argv) > 2 else settings.chunk_strategy

    document = load_document(target, root=RAW_DIR)
    if document is None:
        print("Not loadable.")
        raise SystemExit(1)

    enrich(document, GitDateIndex())
    pieces = chunk_document(document, how)

    print(f"{document.doc_id}")
    print(f"{document.char_count} chars -> {len(pieces)} chunks using '{how}'\n")
    for chunk in pieces[:6]:
        trail = " > ".join(chunk.heading_path) or "(no heading)"
        print(f"--- {chunk.chunk_id}  [{chunk.char_count} chars]  {trail}")
        print(chunk.text[:280].replace("\n", " "))
        print()
