"""Metadata enrichment: the fields that make retrieval filterable and citations checkable.

The project brief asks every document to carry a source name, section heading,
last-updated date, document type, and access level. Those five fields are not
bookkeeping - each one buys a specific capability:

  source_name    group results, and show the user which product an answer came from
  doc_type       filter ("only troubleshooting guides") and diagnose failures by category
  last_updated   detect stale answers; this is what makes the evaluation set's
                 "outdated document trap" questions possible at all
  access_level   demonstrate metadata filtering, which every real deployment needs
  url            let a human open the source and check the citation themselves

Section heading is deliberately absent here: it is a property of a *chunk*, not
a document, so it is assigned during chunking in Step 1.3.

The hard part of this module is `last_updated`, and the interesting part is why.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from copilot.config import PROCESSED_DIR, RAW_DIR


class DocType(StrEnum):
    """What kind of document this is.

    These categories come from the support-corpus requirement in the brief. They
    are also how you will slice the evaluation report: "we retrieve troubleshooting
    guides well but fail on policy pages" is an actionable finding. "Retrieval is
    83%" is not.
    """

    FAQ = "faq"
    TROUBLESHOOTING = "troubleshooting"
    ONBOARDING = "onboarding"
    API_REFERENCE = "api_reference"
    RELEASE_NOTES = "release_notes"
    POLICY = "policy"
    TUTORIAL = "tutorial"
    CONCEPT = "concept"
    GUIDE = "guide"
    UNKNOWN = "unknown"


class AccessLevel(StrEnum):
    """Who is allowed to see this document.

    Honest disclosure: none of the source repositories carry an access level -
    they are all public. We synthesize it from document type so the retrieval
    layer has a real field to filter on, because access filtering is mandatory
    in any actual deployment and a portfolio project that ignores it looks naive.
    The rule is documented rather than hidden, which is the important part.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


# ---------------------------------------------------------------------------
# last_updated: one git pass per repository, not one per file
# ---------------------------------------------------------------------------
# The obvious implementation is `git log -1 --format=%aI -- <path>` for each
# document. That is correct and unusably slow: ~3,500 files x one process launch
# each is several minutes, every single run.
#
# This is the classic N+1 problem. The fix is the same as it is in databases:
# stop asking one question per item, ask one question and index the answer.
#
# `git log --name-only` walks history once and prints every commit with the files
# it touched. Git lists commits newest-first, so the FIRST time a path appears is
# its most recent change. One subprocess per repository, then dictionary lookups.

_COMMIT_MARKER = "\x01"  # a byte that cannot appear in a file path


def build_git_date_index(repo_dir: Path) -> dict[str, str]:
    """Map every path in a repo to its last-modified date (ISO 8601)."""
    result = subprocess.run(
        [
            "git", "-C", str(repo_dir),
            "log",
            f"--format={_COMMIT_MARKER}%aI",
            "--name-only",
            "--no-renames",  # follow the path as written, not through renames
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"  [warn] git log failed in {repo_dir.name}: {result.stderr.strip()[:120]}")
        return {}

    dates: dict[str, str] = {}
    commit_date: str | None = None

    for line in result.stdout.splitlines():
        if line.startswith(_COMMIT_MARKER):
            commit_date = line[1:].strip()
        elif line.strip() and commit_date:
            # setdefault, not assignment: newest-first order means the first
            # sighting is the newest, and later (older) commits must not overwrite it.
            dates.setdefault(line.strip(), commit_date)

    return dates


class GitDateIndex:
    """Lazily built, disk-cached last-modified dates for the whole corpus.

    Building the index costs 10-60 seconds per repository. Caching it to JSON
    means you pay that once instead of on every ingestion run - and you will run
    ingestion many times while tuning chunking.
    """

    def __init__(self, raw_dir: Path | None = None, cache_dir: Path | None = None):
        self.raw_dir = raw_dir or RAW_DIR
        self.cache_dir = cache_dir or PROCESSED_DIR
        self._indexes: dict[str, dict[str, str]] = {}

    def _cache_path(self, source_name: str) -> Path:
        return self.cache_dir / f"git_dates_{source_name}.json"

    def _index_for(self, source_name: str) -> dict[str, str]:
        if source_name in self._indexes:
            return self._indexes[source_name]

        cache = self._cache_path(source_name)
        if cache.exists():
            self._indexes[source_name] = json.loads(cache.read_text(encoding="utf-8"))
            return self._indexes[source_name]

        repo_dir = self.raw_dir / source_name
        print(f"  building git date index for {source_name} (one-time, ~10-60s)...")
        index = build_git_date_index(repo_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(index), encoding="utf-8")
        self._indexes[source_name] = index
        return index

    def lookup(self, source_name: str, rel_path: str) -> datetime | None:
        """rel_path is corpus-relative ('fastapi/docs/...'); git wants repo-relative."""
        index = self._index_for(source_name)
        repo_relative = rel_path.split("/", 1)[1] if "/" in rel_path else rel_path

        iso = index.get(repo_relative)
        if not iso:
            return None
        try:
            return datetime.fromisoformat(iso)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# doc_type: rules per source, most specific first
# ---------------------------------------------------------------------------
# Path-prefix rules, checked in order. A learned classifier would be overkill:
# these repositories organise content by folder, so the folder *is* the label.
# Free, deterministic, and auditable - and a deterministic labeller means a
# re-run cannot silently reshuffle your evaluation categories.

_TYPE_RULES: list[tuple[str, DocType]] = [
    # --- FastAPI ---
    ("fastapi/docs/en/docs/release-notes", DocType.RELEASE_NOTES),
    ("fastapi/docs/en/docs/tutorial", DocType.TUTORIAL),
    ("fastapi/docs/en/docs/advanced", DocType.GUIDE),
    ("fastapi/docs/en/docs/reference", DocType.API_REFERENCE),
    ("fastapi/docs/en/docs/how-to", DocType.GUIDE),
    ("fastapi/docs/en/docs/deployment", DocType.GUIDE),
    ("fastapi/docs/en/docs/learn", DocType.TUTORIAL),
    # --- Kubernetes ---
    ("k8s-website/content/en/docs/tasks/debug", DocType.TROUBLESHOOTING),
    ("k8s-website/content/en/docs/reference", DocType.API_REFERENCE),
    ("k8s-website/content/en/docs/concepts", DocType.CONCEPT),
    ("k8s-website/content/en/docs/tasks", DocType.GUIDE),
    # --- PostHog ---
    ("posthog/contents/handbook/people/onboarding", DocType.ONBOARDING),
    ("posthog/contents/handbook/people", DocType.POLICY),
    ("posthog/contents/handbook/company", DocType.POLICY),
    ("posthog/contents/handbook/growth", DocType.POLICY),
    ("posthog/contents/handbook/brand", DocType.POLICY),
    ("posthog/contents/handbook/engineering", DocType.GUIDE),
    ("posthog/contents/handbook", DocType.POLICY),
    ("posthog/contents/docs/api", DocType.API_REFERENCE),
    ("posthog/contents/tutorials", DocType.TUTORIAL),
    ("posthog/contents/docs", DocType.GUIDE),
    # --- Zulip ---
    ("zulip/api_docs", DocType.API_REFERENCE),
    ("zulip/starlight_help", DocType.FAQ),
    ("zulip/docs", DocType.GUIDE),
    # Catch-alls, last on purpose: every rule above is more specific, and
    # `startswith` matching means the first match wins. Without these, top-level
    # pages fall through to UNKNOWN, and an UNKNOWN bucket is where retrieval
    # failures hide from your evaluation report.
    ("fastapi/docs/en/docs", DocType.GUIDE),
    ("k8s-website/content/en/docs", DocType.CONCEPT),
    ("posthog/contents", DocType.GUIDE),
]

# Filename signals that override the folder rule. A page literally called
# "troubleshooting.md" is a troubleshooting page no matter where it lives.
_FILENAME_SIGNALS: list[tuple[str, DocType]] = [
    ("troubleshoot", DocType.TROUBLESHOOTING),
    ("debug", DocType.TROUBLESHOOTING),
    ("faq", DocType.FAQ),
    ("changelog", DocType.RELEASE_NOTES),
    ("release-note", DocType.RELEASE_NOTES),
    ("onboarding", DocType.ONBOARDING),
]


def classify_doc_type(rel_path: str) -> DocType:
    lowered = rel_path.lower()
    filename = lowered.rsplit("/", 1)[-1]

    for signal, doc_type in _FILENAME_SIGNALS:
        if signal in filename:
            return doc_type

    for prefix, doc_type in _TYPE_RULES:
        if lowered.startswith(prefix):
            return doc_type

    return DocType.UNKNOWN


# ---------------------------------------------------------------------------
# access_level
# ---------------------------------------------------------------------------

_CONFIDENTIAL_MARKERS = ("compensation", "equity", "legal", "security", "incident", "acquisition")


def assign_access_level(rel_path: str, doc_type: DocType) -> AccessLevel:
    """Synthesized, not read from the source. See AccessLevel's docstring."""
    lowered = rel_path.lower()

    if lowered.startswith("posthog/contents/handbook"):
        if any(marker in lowered for marker in _CONFIDENTIAL_MARKERS):
            return AccessLevel.CONFIDENTIAL
        return AccessLevel.INTERNAL

    if doc_type in {DocType.ONBOARDING, DocType.POLICY}:
        return AccessLevel.INTERNAL

    return AccessLevel.PUBLIC


# ---------------------------------------------------------------------------
# url: so a human can verify a citation in one click
# ---------------------------------------------------------------------------

_URL_RULES: list[tuple[str, str]] = [
    ("fastapi/docs/en/docs/", "https://fastapi.tiangolo.com/"),
    ("k8s-website/content/en/docs/", "https://kubernetes.io/docs/"),
    ("posthog/contents/handbook/", "https://posthog.com/handbook/"),
    ("posthog/contents/tutorials/", "https://posthog.com/tutorials/"),
    ("posthog/contents/docs/", "https://posthog.com/docs/"),
    ("zulip/starlight_help/src/content/docs/", "https://zulip.com/help/"),
    ("zulip/api_docs/", "https://zulip.com/api/"),
    ("zulip/docs/", "https://zulip.readthedocs.io/en/latest/"),
]
# Note: zulip/starlight_help/src/content/include/ is deliberately absent. Those
# are MDX partials embedded into other pages - they have no public URL of their
# own, and inventing one would produce a citation link that 404s. A missing URL
# is honest; a broken URL is worse than none.


def build_url(rel_path: str) -> str | None:
    """Reconstruct the public URL a document is published at.

    A citation that reads 'posthog/contents/handbook/people/onboarding' is
    checkable only by someone with the repo. A citation that links to
    posthog.com/handbook/people/onboarding is checkable by anyone in one click.
    Verifiability is the whole theme of this project, so it applies to the
    citation UI too, not just the verifier.
    """
    import re

    for prefix, base in _URL_RULES:
        if rel_path.startswith(prefix):
            tail = rel_path[len(prefix):]
            tail = re.sub(r"\.(md|mdx|html?|txt)$", "", tail, flags=re.IGNORECASE)
            tail = tail.removesuffix("/_index").removesuffix("/index")
            return f"{base}{tail}"
    return None


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def enrich(doc, git_index: GitDateIndex, *, now: datetime | None = None):
    """Attach metadata to a RawDocument, in place, and return it."""
    from copilot.ingest.models import DocumentMetadata

    now = now or datetime.now(timezone.utc)

    doc_type = classify_doc_type(doc.rel_path)
    last_updated = git_index.lookup(doc.source_name, doc.rel_path)

    age_days: int | None = None
    if last_updated is not None:
        # Git dates carry a timezone; if one ever arrives naive, assume UTC so
        # the subtraction cannot raise "can't subtract offset-naive and aware".
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        age_days = max(0, (now - last_updated).days)

    doc.meta = DocumentMetadata(
        source_name=doc.source_name,
        doc_type=doc_type,
        access_level=assign_access_level(doc.rel_path, doc_type),
        last_updated=last_updated,
        age_days=age_days,
        url=build_url(doc.rel_path),
        title=doc.title or doc.doc_id.rsplit("/", 1)[-1],
    )
    return doc
