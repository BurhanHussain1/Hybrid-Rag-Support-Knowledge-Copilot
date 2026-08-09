# Support Knowledge Copilot

A support assistant that answers questions from internal documentation, retrieves evidence using
**hybrid search**, generates answers with **inline citations**, and then **verifies that every
citation actually supports the claim attached to it**.

The last part is the point. Producing a plausible answer with plausible-looking sources is easy.
Proving each source says what the answer claims it says — and refusing when it does not — is the
part that makes a retrieval system trustworthy.

> **Headline result:** _pending Step 6._ This line will read something like
> "hybrid retrieval improved correct-source retrieval from X% to Y% on a 60-question evaluation set."
> It is written last, from real measurements, not estimated in advance.

---

## The assistant contract

Every request returns the same four things, whether the assistant is confident or not:

| Field | Meaning |
|---|---|
| `answer` | Grounded response, with `[chunk_id]` citations inline |
| `citations` | Each cited chunk, plus a **verdict**: does it support the claim, or not? |
| `confidence` | 0–1 score, with the component breakdown that produced it |
| `unverified` | Explicit "what I could not verify" list |

A system that is honest about uncertainty is more useful than one that is confident and wrong.

---

## Architecture

```
                    ┌──────────────────────────────────┐
  docs/ ──────────► │ INGEST                           │
  md · mdx · html   │  load → clean → metadata → chunk │
  pdf · txt         └────────────┬─────────────────────┘
                                 │  chunks share one ID space
                    ┌────────────┴─────────────┐
                    ▼                          ▼
            ┌───────────────┐          ┌───────────────┐
            │ Qdrant        │          │ BM25          │
            │ dense vectors │          │ sparse index  │
            │ (bge-small)   │          │ (rank_bm25)   │
            └───────┬───────┘          └───────┬───────┘
                    │                          │
                    └──────────┬───────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ RRF fusion           │  configurable weights
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Cross-encoder rerank │  top 20 → top 5
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Grounded generation  │  cite or refuse
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Citation verifier    │  claim ↔ evidence check
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Confidence scoring   │  → answer or refusal
                    └──────────────────────┘
```

**Why two indexes over one shared chunk-ID space:** dense and sparse search fail differently. Vector
search understands that "my pods keep restarting" relates to a page titled *Debug Running Pods*, but
it will happily miss the literal string `CrashLoopBackOff`. BM25 nails exact tokens — error codes,
API names, SKUs, flags — and understands nothing. Keeping both over the same IDs means the fusion
layer merges *rankings of the same objects*, so metadata, filtering, and citations stay consistent
no matter which retriever surfaced a chunk.

---

## Tech stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI |
| Embeddings | `BAAI/bge-small-en-v1.5`, local on CPU |
| Vector store | Qdrant (Docker) |
| Sparse search | BM25 via `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `gpt-4o-mini` |
| Dashboard | Streamlit |
| Orchestration | Docker Compose |

Embeddings run locally because chunking gets rewritten many times during this project, and each
rewrite means re-embedding every chunk. Local embeddings make that free, so experiments stay honest
instead of being avoided to save money.

---

## Quickstart

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 2. Secrets
copy .env.example .env          # then add your OpenAI key
# cp .env.example .env          # macOS/Linux

# 3. Corpus (~78 MB, not committed)
python scripts/download_corpus.py

# 4. Vector store
docker compose up -d            # Qdrant dashboard: http://localhost:6333/dashboard

# 5. Chunk the corpus
python ingest.py --strategy both --rebuild
python ingest.py --stats

# 6. Build the indexes (embeddings + BM25)
python index.py --strategy heading --rebuild

# 7. Try it
python search.py "why is my pod crashing" --compare
python search.py "pod stuck in pending state" --explain
```

---

## Corpus

3,396 documents from four real open-source projects — see [`data/CORPUS.md`](data/CORPUS.md) for
sources, licenses, and the metadata mapping.

The mix is chosen to make retrieval genuinely hard: PostHog restates the same procedure across its
handbook, docs, and tutorials (overlapping information), Kubernetes supplies exact-match tokens like
`CrashLoopBackOff` that defeat pure vector search, deprecation notices provide real outdated-document
traps, and four unrelated products guarantee plenty of questions the corpus honestly cannot answer.

---

## Roadmap

- [x] **Step 0** — Repository scaffolding, configuration, Docker, corpus script
- [x] **Step 1** — Ingestion: loaders, metadata extraction, chunking strategies, `ingest.py` CLI
      <br>→ 3,544 documents → **25,906 heading chunks** / **28,059 fixed chunks**, 100% with resolved `last_updated`
- [x] **Step 2** — Indexing: embeddings into Qdrant, BM25 over the same chunk IDs
      <br>→ 25,907 vectors (384-dim cosine) + 1.72M BM25 tokens over one shared ID space
- [x] **Step 3** — Hybrid retrieval: dense, sparse, RRF fusion, cross-encoder reranking
      <br>→ warm latency: dense 50ms, sparse 75ms, hybrid 120ms, +rerank 1.3s
- [ ] **Step 4** — Generation: grounded prompt, citation verification, confidence, refusal handling
- [ ] **Step 5** — FastAPI service exposing the assistant contract
- [ ] **Step 6** — Evaluation: golden set, retrieval/answer/citation/refusal metrics, `eval.py`
- [ ] **Step 7** — Streamlit dashboard with a dense-vs-hybrid comparison toggle
- [ ] **Step 8** — Full Docker Compose stack, case study, walkthrough

---

## Project structure

```
copilot/              importable package
  config.py           every tunable, loaded from .env
  ingest/             loaders, metadata, chunking
  retrieval/          dense, sparse, fusion, rerank
  generation/         answers, citation checks, confidence
  evaluation/         golden set, metrics, reports
scripts/
  download_corpus.py  reproduces the corpus from scratch
data/
  CORPUS.md           sources, licenses, metadata mapping
  raw/                downloaded docs (gitignored)
docker-compose.yml
requirements.txt
.env.example
```

## License

MIT — see [LICENSE](LICENSE). The corpus under `data/raw/` is third-party content under its own
terms and is not redistributed here.
