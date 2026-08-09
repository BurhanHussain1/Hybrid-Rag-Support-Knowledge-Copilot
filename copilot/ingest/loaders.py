"""Loaders: turn messy source files into clean text with headings preserved.

Every corpus source is dirty in a different way, and all of it would poison
retrieval if left in:

  Kubernetes  YAML frontmatter, ``<!-- overview -->`` HTML comments,
              Hugo shortcodes like ``{{< note >}}``
  Zulip       MDX ``import`` statements and JSX components such as
              ``<FlattenedSteps>`` and ``<EditIcon />``
  PostHog     frontmatter, inline JSX like ``<PrivateLink url="...">``,
              Cloudinary image URLs
  FastAPI     standard markdown, plus code-heavy reference pages

Two rules drive the whole cleaner:

1. **Headings survive.** ``## Debugging Pods`` is not decoration - it is the
   boundary the heading-based chunker splits on, and it becomes the
   ``section_heading`` metadata field the user sees in a citation.

2. **Code blocks survive untouched.** ``kubectl describe pods`` and
   ``CrashLoopBackOff`` are exactly the exact-match tokens BM25 exists to catch.
   Strip or mangle them and you have deleted the reason for building hybrid
   search in the first place.

Inspect any file's cleaned output from the command line:

    python -m copilot.ingest.loaders data/raw/zulip/.../change-your-email-address.mdx
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from copilot.config import RAW_DIR
from copilot.ingest.models import RawDocument

# Extensions we know how to read. Anything else is skipped rather than guessed
# at - a silently mis-parsed file is worse than a file that was never indexed.
SUPPORTED_EXTENSIONS = {".md", ".mdx", ".html", ".htm", ".pdf", ".txt"}

# Files that are corpus noise rather than documentation.
SKIP_FILENAMES = {"readme.md", "contributing.md", "code_of_conduct.md", "security.md", "license.md"}


# ---------------------------------------------------------------------------
# Protecting code from the cleaner
# ---------------------------------------------------------------------------
# The cleaning regexes below strip things that look like HTML tags. Code blocks
# are full of things that look like HTML tags: `<your-namespace>`, `List<T>`,
# shell redirects. So we lift every code block out first, replace it with an
# opaque placeholder, clean the prose, then put the code back exactly as it was.
#
# The placeholder uses \x00 (a null character) because it cannot appear in real
# text and no regex below will match it. Choosing a normal-looking token like
# "CODEBLOCK_1" risks colliding with actual document content.

_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def _protect(text: str) -> tuple[str, list[str]]:
    """Replace code spans with placeholders. Returns (text, saved_blocks)."""
    saved: list[str] = []

    def stash(match: re.Match) -> str:
        saved.append(match.group(0))
        return f"\x00{len(saved) - 1}\x00"

    text = _FENCED_CODE.sub(stash, text)
    text = _INLINE_CODE.sub(stash, text)
    return text, saved


def _restore(text: str, saved: list[str]) -> str:
    """Put the code spans back. Done last, so cleaning never touched them."""
    for i, block in enumerate(saved):
        text = text.replace(f"\x00{i}\x00", block)
    return text


# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------
# Order matters. Comments go before tags (an HTML comment contains characters
# that a tag regex would otherwise chew on), and images go before links (an
# image is a link with a `!` in front, so link-stripping would leave a stray `!`).

_MDX_IMPORT = re.compile(r"^\s*(?:import|export)\s+.*$", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Hugo shortcodes: {{< note >}}, {{% caution %}}, {{< /note >}}. Remove the
# marker but keep the sentence inside it - the note body is real content.
_HUGO_SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
# Kubernetes wraps code samples in tab shortcodes with NO markdown fence:
#
#     {{< tab name="Linux node" codelang="yaml" >}}
#     # The mount into the container is read-only.
#     apiVersion: v1
#     {{< /tab >}}
#
# Strip the shortcodes naively and that YAML lands in the prose, where its "#"
# comments look exactly like markdown headings - 41 phantom headings from this
# one file. Rather than teach every downstream stage about Hugo, we convert these
# into real fenced code blocks so the fence-aware logic already in the chunker
# handles them like any other code.
#
# Matched as an open/close pair (not two separate substitutions) so we can never
# emit an unbalanced fence, which would corrupt fence pairing for the whole file.
_HUGO_CODE_TAB = re.compile(
    r"\{\{<\s*tab\s+[^>]*codelang=\"([^\"]*)\"[^>]*>\}\}(.*?)\{\{<\s*/\s*tab\s*>\}\}",
    re.DOTALL,
)
# A tag must start with a letter, which keeps prose like "if x < y and y > z"
# intact. A naive `<[^>]+>` would silently eat that sentence.
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9._:-]*(?:\s[^<>]*?)?/?>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_BADGE_LINE = re.compile(r"^\s*\[!\[.*$", re.MULTILINE)
# Explicit heading anchors: "## The normal process { #the-normal-process }".
# FastAPI and Kubernetes both use these. Left in, the anchor id ends up in the
# section_heading shown to users and in the breadcrumb we embed - repeating the
# same words in slug form, which adds noise to the vector and reads as a bug.
_HEADING_ANCHOR = re.compile(r"^(#{1,6}\s+.*?)\s*\{\s*#[^}]*\}\s*$", re.MULTILINE)
# PyMdown "blocks" markers, used throughout the FastAPI docs to open and close
# admonitions:  "/// note" ... "///". The text inside is real content; only the
# markers are noise. (Code blocks are already protected, so a "///" comment in a
# code sample is safe.)
_PYMDOWN_BLOCK = re.compile(r"^\s*///.*$", re.MULTILINE)
_EXTRA_BLANKS = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)


def clean_markdown(text: str) -> str:
    """Normalize markdown or MDX into plain, readable text with headings kept."""
    text, saved = _protect(text)

    text = _MDX_IMPORT.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    # Must run before the generic shortcode stripper, which would otherwise
    # delete the tab markers and leave the code stranded in the prose.
    text = _HUGO_CODE_TAB.sub(lambda m: f"\n```{m.group(1)}\n{m.group(2).strip()}\n```\n", text)
    text = _HUGO_SHORTCODE.sub("", text)
    text = _BADGE_LINE.sub("", text)
    text = _MD_IMAGE.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_REF_LINK.sub(r"\1", text)
    text = _HEADING_ANCHOR.sub(r"\1", text)
    text = _PYMDOWN_BLOCK.sub("", text)

    text = _TRAILING_WS.sub("", text)
    text = _EXTRA_BLANKS.sub("\n\n", text)

    return _restore(text, saved).strip()


def _read_text(path: Path) -> str:
    """Read a file as UTF-8, always.

    Windows defaults to cp1252, which turns a UTF-8 curly apostrophe into the
    mojibake "theyâ€™ve". That corruption then gets embedded and indexed, and it
    is genuinely painful to trace back. errors="replace" means one bad byte
    cannot crash a 3,400-document ingestion run.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _first_heading(text: str) -> str | None:
    """First real heading, used as a fallback title.

    Skips anything inside a code fence, for the same reason the chunker does: a
    shell comment like `# Create the Role` is not a heading, and using one as a
    document title is both wrong and very visible in citations.
    """
    from copilot.ingest.chunking import find_headings  # imported here to avoid a cycle

    for match in find_headings(text):
        if len(match.group(1)) <= 2:  # H1 or H2 only
            return match.group(2).strip()
    return None


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Per-format loaders
# ---------------------------------------------------------------------------

def load_markdown(path: Path) -> tuple[str, str, dict, str | None]:
    """Markdown and MDX. Returns (raw, clean, frontmatter, title)."""
    import frontmatter  # imported lazily so `--help` does not pay for it

    raw = _read_text(path)

    try:
        post = frontmatter.loads(raw)
        meta, body = dict(post.metadata), post.content
    except Exception:
        # Malformed YAML in one file must not kill the whole run. We keep the
        # body and lose only that file's metadata.
        meta, body = {}, raw

    clean = clean_markdown(body)
    title = meta.get("title") or _first_heading(clean) or _title_from_filename(path)
    return raw, clean, meta, str(title)


def load_html(path: Path) -> tuple[str, str, dict, str | None]:
    from bs4 import BeautifulSoup

    raw = _read_text(path)
    soup = BeautifulSoup(raw, "lxml")

    # Navigation, scripts and styles are the same on every page. Indexed, they
    # would match every query weakly and add noise to every result.
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Re-mark headings as markdown so downstream chunking sees one consistent
    # format regardless of the original file type.
    for level in range(1, 7):
        for tag in soup.find_all(f"h{level}"):
            tag.replace_with(f"\n\n{'#' * level} {tag.get_text(strip=True)}\n\n")

    title = soup.title.get_text(strip=True) if soup.title else None
    clean = _EXTRA_BLANKS.sub("\n\n", soup.get_text("\n")).strip()
    return raw, clean, {}, title or _title_from_filename(path)


def load_pdf(path: Path) -> tuple[str, str, dict, str | None, list[tuple[int, int]]]:
    """PDFs, tracking where each page begins so citations can name a page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # scanned or image-only page; OCR is out of scope
        offsets.append((page_no, cursor))
        parts.append(text)
        cursor += len(text) + 2  # +2 for the "\n\n" join below

    clean = _EXTRA_BLANKS.sub("\n\n", "\n\n".join(parts)).strip()
    meta = {k.lstrip("/"): str(v) for k, v in (reader.metadata or {}).items()}
    title = meta.get("Title") or _title_from_filename(path)
    return clean, clean, meta, title, offsets


def load_plain_text(path: Path) -> tuple[str, str, dict, str | None]:
    raw = _read_text(path)
    clean = _EXTRA_BLANKS.sub("\n\n", raw).strip()
    return raw, clean, {}, _first_heading(clean) or _title_from_filename(path)


# ---------------------------------------------------------------------------
# Dispatcher and discovery
# ---------------------------------------------------------------------------

def make_doc_id(rel_path: str) -> str:
    """Human-readable, stable document ID.

    A hash like 'a3f9c2' would be shorter, but this ID ends up inside citations
    the user reads. 'k8s-website/tasks/debug/debug-pods' tells you what you are
    looking at; 'a3f9c2' tells you nothing and makes debugging retrieval a chore.
    Stability matters too: re-running ingestion must produce the same IDs, or
    every saved evaluation result silently becomes worthless.
    """
    no_ext = re.sub(r"\.(md|mdx|html?|pdf|txt)$", "", rel_path, flags=re.IGNORECASE)
    return no_ext.replace("\\", "/")


def load_document(path: Path, root: Path | None = None) -> RawDocument | None:
    """Load one file into a RawDocument, or return None if unsupported/empty."""
    root = root or RAW_DIR
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return None

    rel_path = path.relative_to(root).as_posix()
    page_offsets: list[tuple[int, int]] = []

    if suffix in {".md", ".mdx"}:
        raw, clean, meta, title = load_markdown(path)
        file_type = suffix.lstrip(".")
    elif suffix in {".html", ".htm"}:
        raw, clean, meta, title = load_html(path)
        file_type = "html"
    elif suffix == ".pdf":
        raw, clean, meta, title, page_offsets = load_pdf(path)
        file_type = "pdf"
    else:
        raw, clean, meta, title = load_plain_text(path)
        file_type = "txt"

    # A near-empty file is usually a stub or a redirect. Indexing it gives the
    # retriever a low-content chunk that can outrank real answers on short
    # queries, so we drop it here rather than debug it later.
    if len(clean) < 50:
        return None

    return RawDocument(
        doc_id=make_doc_id(rel_path),
        source_name=rel_path.split("/", 1)[0],
        rel_path=rel_path,
        abs_path=path,
        file_type=file_type,
        title=title,
        frontmatter=meta,
        raw_text=raw,
        clean_text=clean,
        page_offsets=page_offsets,
    )


def discover_files(root: Path | None = None) -> Iterator[Path]:
    """Walk the corpus, yielding loadable files.

    Yields lazily rather than building a list: on a large corpus this keeps
    memory flat and lets the caller show progress from the first file.
    """
    root = root or RAW_DIR
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if ".git" in path.parts:  # skip clone internals
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.name.lower() in SKIP_FILENAMES:
            continue
        yield path


def load_corpus(root: Path | None = None) -> Iterator[RawDocument]:
    """Load every document in the corpus, skipping anything that fails.

    One unreadable PDF in 3,400 files should cost you that file, not the run.
    """
    root = root or RAW_DIR
    for path in discover_files(root):
        try:
            doc = load_document(path, root)
        except Exception as exc:  # noqa: BLE001 - resilience beats strictness here
            print(f"  [warn] {path.name}: {type(exc).__name__}: {exc}")
            continue
        if doc is not None:
            yield doc


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    target = Path(sys.argv[1]).resolve()
    # Use the corpus root when the file lives inside it, so the printed doc_id
    # matches what a real ingestion run would produce.
    try:
        target.relative_to(RAW_DIR)
        preview_root = RAW_DIR
    except ValueError:
        preview_root = target.parent

    document = load_document(target, root=preview_root)
    if document is None:
        print("Not loadable (unsupported extension or too short).")
        raise SystemExit(1)

    print(f"title      : {document.title}")
    print(f"file_type  : {document.file_type}")
    print(f"frontmatter: {document.frontmatter}")
    print(f"chars      : {document.char_count} (raw {len(document.raw_text)})")
    print("-" * 70)
    print(document.clean_text[:2000])
