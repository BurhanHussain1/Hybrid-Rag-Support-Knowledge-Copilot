"""BM25: keyword search over the same chunk IDs as the vector index.

BM25 scores a document by how often the query's words appear in it, adjusted for
how rare each word is across the corpus and how long the document is. It has no
idea what anything means. That is precisely why it is here.

Vector search and BM25 fail in opposite directions:

  question                          dense                sparse
  "my pods keep restarting"         finds "Debug Pods"   misses (no shared words)
  "CrashLoopBackOff"                fuzzy, unreliable    exact hit
  "--dry-run=client"                usually lost         exact hit

Error codes, CLI flags, API names, SKUs and product names are strings, not
concepts. An embedding blurs them into whatever they resemble. BM25 matches them
literally. Running both and merging is the whole idea of hybrid retrieval.

Both indexes are built over the *same chunk IDs*, read out of Qdrant rather than
re-parsed from JSONL, so the two rankings are guaranteed to describe the same
objects. If they drifted apart, fusion would be merging rankings of different
things and the bug would be nearly invisible.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from copilot.config import INDEX_DIR

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
# BM25 matches tokens literally, so tokenization decides what it can find. Three
# choices, each earning its keep on this corpus:

# 1. Keep dots, hyphens and underscores *inside* words. Splitting them would
#    destroy `--dry-run`, `posthog-js`, `app.routes` and `api_key` - exactly the
#    strings a support question quotes verbatim.
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")

# 2. Split CamelCase into parts as well as keeping the whole token, so
#    `CrashLoopBackOff` is findable by someone typing "crash loop back off".
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

# 3. A small stopword list. BM25 already down-weights common words, so this is a
#    modest speed and noise win rather than a necessity - and keeping it short
#    avoids removing words that matter in technical text ("no", "not", "all").
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of to in on at for with
by from as is are was were be been being it its do does did doing have has had
you your we our they their i me my he she his her them us so such via up out
""".split())


def tokenize(text: str) -> list[str]:
    """Text -> list of searchable tokens."""
    tokens: list[str] = []

    for match in _TOKEN.finditer(text):
        raw = match.group(0)
        lowered = raw.lower()

        if lowered not in _STOPWORDS and len(lowered) > 1:
            tokens.append(lowered)

        # Only bother splitting CamelCase when the token actually is CamelCase.
        if len(raw) > 3 and any(c.isupper() for c in raw[1:]) and not raw.isupper():
            parts = [p.lower() for p in _CAMEL.findall(raw)]
            if len(parts) > 1:
                tokens.extend(p for p in parts if len(p) > 1 and p not in _STOPWORDS)

    return tokens


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------

class BM25Index:
    """A BM25 index plus the chunk IDs its rows correspond to.

    rank_bm25 works on positions: you give it a list of tokenized documents and
    it returns a score per position. It knows nothing about our IDs, so we keep a
    parallel list and map position -> chunk_id ourselves. Keeping those two lists
    in the same order is the only invariant this class has to protect.
    """

    def __init__(self, chunk_ids: list[str], model, payloads: dict[str, dict] | None = None):
        self.chunk_ids = chunk_ids
        self.model = model
        self.payloads = payloads or {}

    # -- building ----------------------------------------------------------

    @classmethod
    def build(cls, records: list[tuple[str, str, dict]], *, quiet: bool = False) -> "BM25Index":
        """Build from (chunk_id, text, payload) triples."""
        from rank_bm25 import BM25Okapi

        chunk_ids = [chunk_id for chunk_id, _, _ in records]
        payloads = {chunk_id: payload for chunk_id, _, payload in records}

        if not quiet:
            print(f"  tokenizing {len(records)} chunks...")
        corpus = [tokenize(text) for _, text, _ in records]

        if not quiet:
            total = sum(len(doc) for doc in corpus)
            print(f"  {total} tokens, {total // max(len(corpus), 1)} per chunk average")
            print("  fitting BM25...")

        return cls(chunk_ids, BM25Okapi(corpus), payloads)

    # -- searching ---------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Top-scoring chunks for a query, highest score first."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.model.get_scores(tokens)

        # argsort over 26,000 floats is fine here. If the corpus grew by orders
        # of magnitude, a heap would be the better tool.
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "score": float(scores[i]),
                "payload": self.payloads.get(self.chunk_ids[i], {}),
            }
            for i in ranked
            if scores[i] > 0  # a zero score means no query term matched at all
        ]

    # -- persistence -------------------------------------------------------
    # Pickle, because rank_bm25 holds precomputed term-frequency tables that
    # would take ~30s to recompute on every API start. Pickle is unsafe to use on
    # files you did not create - this one is only ever written by index.py, and it
    # lives in gitignored data/indexes/, so it is never shared.

    @staticmethod
    def path(strategy: str, index_dir: Path | None = None) -> Path:
        return (index_dir or INDEX_DIR) / f"bm25_{strategy}.pkl"

    def save(self, strategy: str, index_dir: Path | None = None) -> Path:
        target = self.path(strategy, index_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "model": self.model, "payloads": self.payloads},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return target

    @classmethod
    def load(cls, strategy: str, index_dir: Path | None = None) -> "BM25Index":
        target = cls.path(strategy, index_dir)
        if not target.exists():
            raise FileNotFoundError(
                f"{target} not found. Run: python index.py --strategy {strategy} --rebuild"
            )
        with target.open("rb") as handle:
            state = pickle.load(handle)
        return cls(state["chunk_ids"], state["model"], state.get("payloads", {}))

    def __len__(self) -> int:
        return len(self.chunk_ids)
