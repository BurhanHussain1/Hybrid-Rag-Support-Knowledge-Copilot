# Retrieval and Answer Evaluation

_Generated 2026-08-09 18:06 UTC · 64s total_

## Headline

**Dense retrieval found the correct source document for 87.5% of 8 questions.**

## How to read this

- **8 questions scored** out of 63 in the set.
- **8 questions were not human-verified.** These numbers are provisional until they are reviewed.
- Ground truth is at **document** level, not chunk level, so it survives re-chunking.
- `hit@k` is the metric that matters most in practice: the model needs only one correct source to answer well.
- `precision` is reported but not optimised for. A question is often answerable from documents the answer key does not list, so a "wrong" retrieval is frequently a valid unlisted source.
- Refusal questions are excluded from retrieval metrics (they have no expected document) and scored in the answer section instead.

## Retrieval (top-5)

| mode | hit@k | full recall | MRR | precision | mean latency |
|---|---|---|---|---|---|
| dense | **87.5%** | 87.5% | 0.750 | 53.5% | 4548ms |
| sparse | **50.0%** | 50.0% | 0.500 | 16.7% | 302ms |
| hybrid | **75.0%** | 75.0% | 0.650 | 44.2% | 231ms |
| rerank | **87.5%** | 87.5% | 0.775 | 52.9% | 2940ms |

**hit@k** — at least one correct document retrieved. **full recall** — every correct document retrieved (only differs for multi-document questions). **MRR** — 1/rank of the first correct document, so it rewards ranking the right thing *first*, not merely somewhere.

### hit@k by question type

| mode | simple_lookup (n=8) |
|---|---|
| dense | 87.5% |
| sparse | 50.0% |
| hybrid | 75.0% |
| rerank | 87.5% |

This table is the useful one. An aggregate score hides which *kind* of question fails, which is the only thing that tells you what to fix next.

## Configuration

Every setting that could change these numbers:

| setting | value |
|---|---|
| `chunk_strategy` | heading |
| `chunk_size` | 800 |
| `chunk_overlap` | 150 |
| `min_chunk_chars` | 200 |
| `embedding_model` | BAAI/bge-small-en-v1.5 |
| `reranker_model` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `llm_model` | gpt-4o-mini |
| `judge_model` | gpt-4o-mini (same as generator) |
| `dense_top_k` | 20 |
| `sparse_top_k` | 20 |
| `rrf_k` | 60 |
| `dense_weight` | 1.0 |
| `sparse_weight` | 1.0 |
| `rerank_top_n` | 20 |
| `final_top_k` | 5 |
| `min_confidence` | 0.35 |

## Question set

```
questions   63
verified    0
scorable    0

by category:
  ambiguous          4
  multi_doc         11
  no_answer         13
  outdated_trap     10
  simple_lookup     25

by status:
  draft             62
  rejected           1
```

## Known limitations

- The citation judge defaults to the same model that writes the answers, so it is predisposed to agree with itself. Set `JUDGE_MODEL` to a different model for an independent check, and spot-check a sample by hand.
- Questions were drafted with LLM assistance from real corpus chunks, then screened automatically and promoted by a human. Drafting from the corpus makes ground truth reliable; it does not make the questions representative of real user traffic.
- The ambiguous category is small, because the four corpora overlap little semantically. Its numbers carry correspondingly less weight.
