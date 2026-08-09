# Support Knowledge Copilot

A support assistant that answers questions from internal documentation, retrieves evidence using
**hybrid search**, generates answers with **inline citations**, and then **verifies that every
citation actually supports the claim attached to it**.

The last part is the point. Producing a plausible answer with plausible-looking sources is easy.
Proving each source says what the answer claims it says — and refusing when it does not — is the
part that makes a retrieval system trustworthy.

> **Headline result:** the full pipeline — hybrid retrieval plus cross-encoder reranking — found the
> correct source document for **93.9%** of questions, against **87.8%** for dense-only vector search
> and **75.5%** for BM25 alone, on a 62-question evaluation set. **79.6%** of the citations it
> produced survived independent verification, and it correctly decided whether to answer or refuse
> **93.5%** of the time.
>
> The interesting part is what fusion alone did: **RRF without reranking scored 85.7%, slightly
> *worse* than dense-only.** Merging the two retrievers is not automatically an improvement — the
> reranker is what converts extra candidates into better answers. Full numbers, per-question-type
> breakdown and every setting that produced them: [`reports/eval-full.md`](reports/eval-full.md).

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

# 7. Try it from the CLI
python search.py "why is my pod crashing" --compare
python ask.py "why is my pod stuck in pending"

# 8. Or run the API
python serve.py                 # docs at http://localhost:8000/docs
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Full contract: answer, verified citations, confidence breakdown, unverified list |
| `POST /search` | Retrieval only — no LLM call, for the dense-vs-hybrid comparison |
| `GET /health` | Reports Qdrant reachability, chunk count, BM25 presence, key configured |
| `GET /stats` | Every active retrieval setting, so a result can be reproduced |

A refusal is a normal `200` with `answered: false` — the question was valid and the
answer was honest, so nothing faulted.

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
- [x] **Step 4** — Generation: grounded prompt, citation verification, confidence, refusal handling
      <br>→ verifier scored 6/6 on hand-built cases; ~$0.0007 per answered question
- [x] **Step 5** — FastAPI service exposing the assistant contract
      <br>→ models preloaded at boot: 26.5s startup once, then `/search` 90ms and `/ask` ~8.5s
- [x] **Step 6** — Evaluation: golden set, retrieval/answer/citation/refusal metrics, `eval.py`
      <br>→ 63 questions across 5 types; rerank **93.9%** vs dense **87.8%**; report in `reports/`
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
