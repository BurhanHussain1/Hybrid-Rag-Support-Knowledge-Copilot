# Retrieval and Answer Evaluation

_Generated 2026-08-09 18:07 UTC · 605s total_

## Headline

**Rerank retrieval found the correct source document for 93.9% of questions, against 87.8% for dense-only** — a +6.1pp change across 49 questions.

## How to read this

- **62 questions scored** out of 63 in the set.
- **62 questions were not human-verified.** These numbers are provisional until they are reviewed.
- Ground truth is at **document** level, not chunk level, so it survives re-chunking.
- `hit@k` is the metric that matters most in practice: the model needs only one correct source to answer well.
- `precision` is reported but not optimised for. A question is often answerable from documents the answer key does not list, so a "wrong" retrieval is frequently a valid unlisted source.
- Refusal questions are excluded from retrieval metrics (they have no expected document) and scored in the answer section instead.

## Retrieval (top-5)

| mode | hit@k | full recall | MRR | precision | mean latency |
|---|---|---|---|---|---|
| dense | **87.8%** | 75.5% | 0.740 | 43.0% | 830ms |
| sparse | **75.5%** | 63.3% | 0.608 | 26.2% | 208ms |
| hybrid | **85.7%** | 75.5% | 0.709 | 39.8% | 246ms |
| rerank | **93.9%** | 79.6% | 0.799 | 44.5% | 2357ms |

**hit@k** — at least one correct document retrieved. **full recall** — every correct document retrieved (only differs for multi-document questions). **MRR** — 1/rank of the first correct document, so it rewards ranking the right thing *first*, not merely somewhere.

### hit@k by question type

| mode | ambiguous (n=3) | multi_doc (n=11) | outdated_trap (n=10) | simple_lookup (n=25) |
|---|---|---|---|---|
| dense | 100.0% | 100.0% | 90.0% | 80.0% |
| sparse | 100.0% | 90.9% | 50.0% | 76.0% |
| hybrid | 100.0% | 100.0% | 80.0% | 80.0% |
| rerank | 100.0% | 100.0% | 100.0% | 88.0% |

This table is the useful one. An aggregate score hides which *kind* of question fails, which is the only thing that tells you what to fix next.

## Answer quality (mode: rerank)

| metric | value |
|---|---|
| questions | 62 |
| refusal accuracy | **93.5%** |
| correct refusals | 11 |
| missed refusals (answered when it should not) | 2 |
| false refusals (refused when it could answer) | 2 |
| citation support rate | **79.6%** |
| unsupported citations caught | 30 of 112 |
| key-fact coverage | 45.9% |
| forbidden mentions | 0 |
| mean confidence when answering | 0.82 |
| mean confidence when refusing | 0.21 |

**Refusal accuracy is scored in both directions on purpose.** A system that refuses everything scores perfectly on unanswerable questions and is useless, so the same metric has to punish refusing answerable ones.

**Key-fact coverage is substring matching** and therefore a weak proxy — a correct answer phrased differently scores low. It is a signal, not a verdict.

### Where it went wrong

| question | type | expected | got | unsupported citations |
|---|---|---|---|---|
| q001 | simple_lookup | answer | answered | 1 |
| q002 | simple_lookup | answer | answered | 1 |
| q006 | simple_lookup | answer | answered | 2 |
| q007 | simple_lookup | answer | answered | 2 |
| q008 | simple_lookup | answer | answered | 2 |
| q011 | simple_lookup | answer | refused | 0 |
| q017 | simple_lookup | answer | answered | 1 |
| q024 | simple_lookup | answer | answered | 4 |
| q025 | simple_lookup | answer | answered | 4 |
| q027 | multi_doc | answer | answered | 1 |
| q028 | multi_doc | answer | answered | 1 |
| q029 | multi_doc | answer | answered | 1 |
| q030 | multi_doc | answer | answered | 1 |
| q031 | multi_doc | answer | answered | 4 |
| q033 | multi_doc | answer | answered | 1 |
| q034 | multi_doc | answer | answered | 1 |
| q035 | multi_doc | answer | refused | 0 |
| q041 | outdated_trap | answer | answered | 1 |
| q052 | no_answer | refuse | refused | 1 |
| q053 | no_answer | refuse | refused | 1 |

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

## Cost

174 LLM calls · 150,140 in + 13,029 out tokens · **$0.0303**

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
