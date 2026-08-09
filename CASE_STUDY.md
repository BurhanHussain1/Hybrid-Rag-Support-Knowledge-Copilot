# Support Knowledge Copilot — Case Study

A documentation assistant that retrieves evidence with hybrid search, answers with inline
citations, and then **checks whether each cited source actually supports the claim attached to it**.

---

## Result

On a 62-question evaluation set built from a 3,544-document corpus:

| Retrieval strategy | Correct source found (hit@5) | MRR | Mean latency |
|---|---|---|---|
| BM25 only | 75.5% | 0.608 | 208ms |
| RRF fusion (no reranking) | 85.7% | 0.709 | 246ms |
| Dense vectors only | 87.8% | 0.740 | 830ms |
| **Hybrid + cross-encoder reranking** | **93.9%** | **0.799** | 2,357ms |

Answer-level, on the same questions:

- **79.6% citation support rate** — 30 of 112 citations were caught not supporting their claim
- **93.5% refusal accuracy** — 11 correct refusals, 2 missed, 2 false
- **$0.03** for a complete evaluation run

### The finding worth leading with

**RRF fusion on its own scored 85.7% — two points *worse* than plain dense search.**

Merging two retrievers is not automatically an improvement. Fusion pulled BM25's candidates into
the result set, and some of them displaced better dense results. The cross-encoder reranker is what
converted those extra candidates into a real gain. Reported as hybrid-vs-dense alone, the honest
number is *negative*; the +6.1pp comes from fusion **and** reranking together.

---

## Architecture

```
3,544 docs ──► load ──► metadata ──► chunk ──► 25,907 chunks
   md/mdx/html/pdf         │                        │
                           │                  one shared ID space
                           │                   ┌────┴────┐
                           ▼                   ▼         ▼
                    doc_type, access,      Qdrant     BM25
                    last_updated (git),   384-dim    rank_bm25
                    public URL             cosine
                                              └────┬────┘
                                                   ▼
                                          RRF fusion (rank-based)
                                                   ▼
                                     cross-encoder rerank: 20 → 5
                                                   ▼
                                    grounded generation, [n] citations
                                                   ▼
                                    citation verifier (claim ↔ source)
                                                   ▼
                                    confidence score → answer or refuse
```

**Stack:** Python 3.12 · FastAPI · Qdrant · `bge-small-en-v1.5` (local CPU) ·
`rank_bm25` · `ms-marco-MiniLM-L-6-v2` cross-encoder · `gpt-4o-mini` · Streamlit · Docker Compose

**Corpus:** Kubernetes docs, FastAPI docs, PostHog docs + company handbook, Zulip help centre —
four real open-source documentation sets, chosen so that content genuinely overlaps, exact-match
tokens like `CrashLoopBackOff` exist, real deprecation notices provide stale-document traps, and
plenty of questions are honestly unanswerable.

---

## Retrieval strategy

**Two indexes over one shared chunk-ID space.** Dense and sparse search fail in opposite
directions, and the failures are not predictable per query:

| Query | Dense | Sparse |
|---|---|---|
| "my pods keep restarting over and over" | finds *Container restarts* | returns unrelated pages — 0/3 overlap |
| "kubectl describe pods" | ranks the reference page lower | exact hit |

Building both over the same IDs means fusion merges **rankings of the same objects**, so metadata,
filtering and citations stay consistent regardless of which retriever surfaced a chunk. BM25 is
built by reading chunks back out of Qdrant, which makes it structurally impossible for the two
indexes to cover different sets.

**Why RRF and not weighted score fusion.** Dense produces cosine similarities in 0–1; BM25 produces
unbounded scores (40+ observed). Min-max normalising each result list makes a chunk's score depend
on what else happened to be retrieved, so fusion weights mean something different for every query.
RRF discards scores and uses only rank, which is scale-free.

**`rrf_k` was measured, not chosen.** A single query suggested the default of 60 was too flat. A
sweep over the golden set showed:

| k | hybrid only | + reranker |
|---|---|---|
| 1 | 89.8% | 93.9% |
| 60 | 85.7% | 93.9% |
| 200 | 85.7% | 93.9% |

Small `k` does help pure fusion. **After reranking, every value gives the identical result** —
fusion only has to get the right document into the 20-candidate pool, and every `k` managed that.
The default stayed at 60: the k=1 gain applies only to a configuration that isn't shipped, and
4.1pp on 49 questions is about two questions. The sweep's value was discovering the knob doesn't
matter here.

---

## Chunking decisions

Two strategies, both built, with the strategy recorded on every chunk so they can be compared:

| | chunks | median size | text retained |
|---|---|---|---|
| heading-based | 25,907 | 603 chars | 0.92× source |
| fixed-size + 150 overlap | 28,059 | 795 chars | 1.20× source |

Heading-based splits where the author already decided a topic ends. Fixed-size is predictable and
works without headings, at the cost of 20% duplicated text — the same fact retrievable twice.

**Breadcrumbs are prepended before embedding.** A chunk reading *"Click Manage channel, then
Archive."* is meaningless alone. Embedded as *"Zulip help > Archive a channel\n\nClick Manage
channel…"* it becomes findable. The displayed text and the embedded text are deliberately different.

**Ground truth is document-level, never chunk-level.** Chunk IDs are positional (`#h3` = 4th heading
section), so changing chunk size renumbers everything. Since comparing chunking strategies is one of
the things being measured, chunk-level ground truth would break the moment the experiment ran.
`doc_id` derives from the file path and survives every re-chunk.

---

## Citation verification

The failure this catches: retrieval finds five relevant chunks, the model writes a fluent answer
with citations throughout, and one sentence is not stated in the source it cites. The answer looks
impeccably sourced and contains an unsupported claim.

Three design choices:

1. **Numbered sources, not chunk IDs.** The model cites `[3]`; the mapping back to
   `k8s-website/.../debug-pods#h3` is done in code. Asking a model to retype long paths inside prose
   produces unresolvable citations. Removing the opportunity to hallucinate beats detecting it after.
   Any number outside the provided range is a fabricated citation — caught with one dict lookup,
   dropped, and reported to the user.

2. **The judge sees one (claim, source) pair in isolation.** Not the answer, not the other sources.
   Given everything, it could justify a claim from a neighbouring source and mark the wrong citation
   supported. The prompt is biased toward "unsupported" when uncertain, because a verifier that
   waves things through produces a confidence number that looks meaningful and isn't.

3. **A free lexical check runs first.** If a claim quotes `` `kubectl drain` `` and that string is
   absent from the source, that's recorded as a signal. Cheap checks before expensive ones.

Validated against six hand-built pairs with known answers — 6/6, including *"A Pending Pod means the
image failed to pull"* (that's the `Waiting` state, not `Pending`) and a half-true claim correctly
graded `partial`.

**Confidence** combines citation support rate (0.40), retrieval strength (0.25), grounding rate
(0.20) and completeness (0.15), and the component breakdown is always returned. "0.42" tells a user
nothing; "0.42, because 2 of 4 citations failed" tells them how much to trust it. Each retrieval
mode's score is normalised separately — cosine, BM25, RRF sums and cross-encoder logits are not
comparable, and feeding them in raw would make confidence mean something different per mode.

---

## Evaluation methodology

**63 questions across five types**, drafted from real corpus chunks (so ground truth is known by
construction), screened automatically, then promoted by hand.

**Retrieval and answer quality are measured separately.** If you only score final answers, a wrong
answer doesn't tell you whether retrieval missed the document or the model misread it — and those
need different fixes. The retrieval pass makes zero LLM calls, so iterating on chunking and fusion
is free; only the answer pass costs money.

**Per-category results are the useful output:**

| mode | ambiguous (3) | multi_doc (11) | outdated_trap (10) | simple_lookup (25) |
|---|---|---|---|---|
| dense | 100% | 100% | 90% | 80% |
| sparse | 100% | 91% | **50%** | 76% |
| hybrid | 100% | 100% | 80% | 80% |
| **rerank** | 100% | 100% | **100%** | **88%** |

Two things an aggregate score would have hidden: **BM25 collapses on stale documents** (50% — old
docs use old terminology, so exact matching fails hardest where wording has drifted), and **simple
lookups are the weakest category**, not the hardest ones.

**Refusal accuracy is scored both directions.** A system that refuses everything scores 100% on
unanswerable questions and is useless, so the same metric punishes refusing answerable ones.

---

## Tradeoffs

| Decision | Gained | Paid |
|---|---|---|
| Local embeddings instead of an API | free re-embedding, so chunking experiments actually get run | ~20 min per full re-index on CPU |
| Cross-encoder reranking | +8.2pp over fusion alone | 2.3s per query vs 246ms |
| Fixed-size chunking | works without headings | 20% duplicated text |
| Storing chunk text in the Qdrant payload | citations render in one round trip | larger index on disk |
| Verifying every citation | caught 30 unsupported claims | one extra LLM call per citation |
| Same model generating and judging | no second provider to configure | the judge agrees with itself; disclosed in code, report and UI |

---

## Failure cases

**Hybrid retrieval can lose to dense alone.** For *"pod stuck in pending state"*, the correct
document was dense #1 and sparse #17; a weaker document was dense #6 and sparse #4. With `k=60`,
rank 1 and rank 6 score within 8% of each other, so "mediocre in both" beat "excellent in one".
Diagnosable in one command because every result carries its per-retriever ranks.

**Three questions where the expected document was never retrieved.** Retrieval returned documents
that arguably answer them. Those were deliberately **not** added to the answer key — fitting ground
truth to the system's output guarantees a pass and measures nothing. They are counted as failures.

**The ambiguous category has only 3–4 questions.** Cross-product neighbours are genuinely rare
because the four corpora barely overlap semantically. Reported with its sample size rather than
padded.

**Key-fact coverage is 45.9% and is a weak metric.** It's substring matching, so a correct answer
phrased differently scores low. Labelled a signal, not a verdict.

---

## Bugs worth recording

Each of these produced plausible-looking output and no error. All were found by reading real data,
not by reading code.

**Shallow clone destroyed every date.** `git clone --depth 1` yields one commit, so `git log`
reported the same `last_updated` for all 3,544 documents — the moment of cloning. Valid timestamps,
uniformly wrong, and they would have silently invalidated every outdated-document test. Fixed with a
blob-filtered history fetch (~4 min, 190 MB).

**Shell comments parsed as markdown headings.** `# Create the Role to read the credspec` inside a
```shell fence became an H1 — 1,150 phantom headings from 489 distinct comment strings. They split
chunks mid-example and injected shell comments into the breadcrumb that gets embedded.

**`\s` matches newlines.** `^(#{1,6})\s+(.+?)$` on a document containing a bare `##` line matched
*across the line break* and swallowed the next line as heading text, producing 78 chunks whose
section heading was literally ` ``` `. Fixed with `[ \t]+`.

**Giant tables were unsplittable.** A 65-row markdown table has no blank lines, so it arrived as one
29,686-character chunk. bge-small truncates at ~512 tokens, so 93% of it would have been silently
dropped from the embedding — a chunk that exists, looks fine, and is unfindable. The first fix then
shattered the table into 2-character fragments because the header row consumed the entire budget.

**Ten of thirteen "no-answer" questions were answerable.** The model was asked what the corpus
doesn't cover and guessed badly — it does cover privacy compliance, refunds, startup pricing and
Kubernetes security practices. A mislabelled no-answer question punishes the system for being
correct, so the refusal metric would have measured the labels. Fixed by over-generating candidates
and checking each against the live index, keeping only those retrieval genuinely fails on.

**And one heuristic of my own that was wrong.** A "question is too easy" check measured word overlap
with the source and flagged *"How do I import my Mattermost data into Zulip?"* at 100% — a perfectly
natural question. Naming the feature you're asking about isn't cheating. Replaced with a
consecutive-4-word-phrase check: 24 false flags dropped to 6 real ones.

---

## Performance work

**Sorting by length before batching: 9 → 32 chunks/sec.** A transformer processes a batch as one
rectangle, so every text is padded to the longest in its batch. Chunks range 200–1,719 characters
with a median of 603, so a random batch of 256 padded almost everything to ~1,700 — roughly
two-thirds of compute spent on padding. One `sort()` recovered it at zero quality cost.

**Models load at startup, not per request.** The CLI pays ~34s before answering. The API loads
everything and runs one real warmup query before accepting traffic: 26.5s once, then 90ms per
search.

**One git pass instead of 3,544.** Per-file `git log` calls would take ~5 minutes every run. One
`git log --name-only` pass per repository, parsed into a dictionary and cached to JSON, made it 20
seconds once.

---

## What I would do next

1. **Diversity in the final five.** Three of five reranked chunks sometimes come from the same
   section, so the model sees near-duplicates instead of five distinct sources. MMR or a
   per-document cap would help.
2. **An independent judge.** The verifier defaults to the same model that writes the answer.
   Pointing `JUDGE_MODEL` at a different model and measuring the disagreement rate would quantify
   how much the 79.6% is inflated by self-agreement.
3. **Compare chunking strategies end to end.** Both chunk sets exist and the harness supports it;
   only the heading set has been fully indexed.
4. **Calibrate the confidence weights.** They are chosen by judgement. With the golden set, they
   could be fitted against actual correctness instead.
5. **Grow the ambiguous category.** Three questions is too few to draw conclusions from.

---

## Reproducing

```bash
python scripts/download_corpus.py     # 3,544 docs, ~264 MB
python ingest.py --strategy both --rebuild
docker compose up -d qdrant
python index.py --strategy heading --rebuild
python eval.py --full --json          # writes reports/eval-full.md
```

Full report with every setting that produced these numbers: [`reports/eval-full.md`](reports/eval-full.md)
