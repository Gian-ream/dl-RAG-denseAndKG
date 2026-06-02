# Data Dictionary

Catalog of **every data artifact** in the project: for each file, the
schema (columns + type + meaning), what produces it and what consumes it.

Meant as a **quick reference**: when you don't remember what a file
contains (e.g. "what's in `n1`?"), look here instead of opening the code.

`data/` is gitignored (~14.7 GB). The schemas described here are derived
from the code that PRODUCES each file, not from manual inspection.

Type convention: parquet/DuckDB types are given where relevant;
`str`/`int`/`float`/`list` for JSONL and Python structures.

Snapshot: **2026-05-20**.

---

## Flow map (who produces what)

```
01_corpus    → psgs_w100_sentence.tsv
03_embedding → faiss_index/shard_XX.npy + shard_XX_ids.npy
02_filtering → qa_all_entities.jsonl
04_answer    → top100_subset.parquet, passage_entities_subset.parquet
05_curation  → top100_candidates.parquet, curation_chunks/*.parquet,
               curation_results.jsonl
06_apply     → queries_curated.jsonl, top100_curated.parquet,
               passage_entities_curated.parquet, query_embeddings_curated.npy

Layer 1   (hdt_export)  → edges.parquet
Layer 1.5 (node_stats)  → node_stats.parquet
Layer 1.6 (build_labels)→ labels.parquet
Layer 3   (build_n1)    → n1.parquet
utils/kg.py (KGScorer)  → kg.duckdb (tables n1 + edges)

07_kg_rerank → kg_pairs_raw.parquet (Phase A), kg_rerank_grid.parquet (Phase B)
08_llm_eval  → llm_eval/llm_eval_inputs.parquet, llm_eval/llm_responses_{cond}.jsonl
09_llm_judge → llm_eval/judgments_{cond}.jsonl, llm_eval/judgments_summary.parquet
```

---

## 1 · Corpus & retrieval (Step 0-4)

### psgs_w100_sentence.tsv
**Path**: `data/wikipedia_2018_sentence_aligned/psgs_w100_sentence.tsv`
**Produced by**: `01_corpus_preparation.ipynb`  |  **Rows**: ~42M passages (~25 GB)

| column | type | meaning |
|---|---|---|
| `id` | int | unique passage identifier |
| `text` | str | passage content (~100 words, sentence-aligned) |
| `title` | str | source Wikipedia article title |

**Consumed by**: `03_embedding.ipynb` (Contriever encoding).
**Notes**: sentence-aligned variant of the standard DPR `psgs_w100` corpus.

---

### FAISS shards — `shard_XX.npy` + `shard_XX_ids.npy`
**Path**: `data/faiss_index/shard_XX.npy`, `data/faiss_index/shard_XX_ids.npy`
**Produced by**: `03_embedding.ipynb`  |  **Shard**: ~5M vectors each

| file | content |
|---|---|
| `shard_XX.npy` | embedding matrix `(N, 768)` float32 — one Contriever vector per passage |
| `shard_XX_ids.npy` | array `(N,)` of the corresponding `passage_id`s, **same order** as the rows of `shard_XX.npy` |

**Consumed by**: `04_answer_preparation.ipynb` (retrieval), `07_kg_rerank.ipynb` (binding validation §2.4).
**Notes**: FAISS holds ONLY vectors — the row→passage_id mapping lives in `shard_XX_ids.npy`. Loaded with `mmap_mode="r"` to avoid saturating RAM (see GLOSSARIO → mmap).

---

### qa_all_entities.jsonl
**Path**: `data/NQ_question/qa_all_entities.jsonl`
**Produced by**: `02_nq_filtering.ipynb`  |  **Rows**: 31,372 queries (after token≤5 + entity-linking filter)

| field | type | meaning |
|---|---|---|
| `question` | str | NQ-open question text |
| `answers` | list[str] | gold answers (accepted variants) |
| `question_qids` | list[str] | Wikidata QIDs of entities recognized in the question (ReFiNed) |
| `answer_variant_qids` | list[list[str]] | QIDs for each answer variant |

**Consumed by**: `04_answer_preparation.ipynb` (1000-query subset selection).

---

### Curation intermediates (notebook 05)

Notebook `05_answer_curation.ipynb` finds the 344 queries whose top-100
passages have **no linkable entities** and picks replacement queries from a
candidate pool. These three artifacts are its internal/output products; the
final curated files are produced by `06_apply_curation.ipynb` (below).

#### top100_candidates.parquet
**Path**: `data/NQ_answer/top100_candidates.parquet`
**Produced by**: `05_answer_curation.ipynb` (§7)  |  **Rows**: 500,000 (5000 candidate queries × 100)

| column | type | meaning |
|---|---|---|
| `query_id` | int64 | **local** candidate-pool index (0..4999) — NOT the 0..999 of the final subset |
| `passage_id` | int64 | candidate passage id |
| `score` | double | Contriever inner product |
| `rank` | int32 | position 0..99 in the candidate's top-100 |
| `shard_id` | int64 | FAISS shard that held the passage |

**Consumed by**: `05` §8-10 (entity-linking + substitute selection).

#### curation_chunks/*.parquet
**Path**: `data/NQ_answer/curation_chunks/chunk_000.parquet` … `chunk_037.parquet` (**38 files**)
**Produced by**: `05_answer_curation.ipynb` (§8)  |  **Rows**: 375,943 total (~10K/chunk)

| column | type | meaning |
|---|---|---|
| `id` | int64 | passage id |
| `title` | string | Wikipedia article title |
| `text` | string | passage content |
| `qids` | list[str] | ReFiNed-linked Wikidata QIDs of the passage |

ReFiNed entity-linking of the candidate passages, chunked for resumability.
**Consumed by**: `05` §9 (concatenated into the candidate entity pool).

#### curation_results.jsonl
**Path**: `data/NQ_answer/curation_results.jsonl`
**Produced by**: `05_answer_curation.ipynb` (§10)  |  **Rows**: 344 (one per substituted query)

| field | type | meaning |
|---|---|---|
| `local_query_id` | int | index in the candidate pool (0..4999) of the chosen replacement |
| `original_query_id` | int | index in the full NQ pool (`qa_all_entities.jsonl`) of the dropped query |
| `n_unique_entities` | int | # unique question QIDs of the replacement |
| `question` | str | replacement question text |

The **substitution map** that drives curation: which 344 entity-poor queries
get replaced by which candidates.
**Consumed by**: `06_apply_curation.ipynb` (applies the swap).

---

### top100_curated.parquet
**Path**: `data/NQ_answer/top100_curated.parquet`
**Produced by**: `06_apply_curation.ipynb`  |  **Rows**: 100,000 (1000 queries × 100 candidates)

| column | type | meaning |
|---|---|---|
| `query_id` | int32 | **positional index** (0..999) into `queries_curated.jsonl`, NOT original_query_id |
| `passage_id` | int64 | candidate passage id (key into `passage_entities_curated`) |
| `score` | float | raw Contriever inner product (IP, NO L2 norm) |
| `rank` | int16 | position 0..99 in the dense retriever's top-100 |
| `shard_id` | int8 | FAISS shard (0..8) that held this passage's embedding |

**Consumed by**: `07_kg_rerank.ipynb`, `08_llm_eval.ipynb`.
**Notes**: `query_id` is a positional binding, verified by §2.4 of 07 (dot-product recompute, max|Δ|=2.15e-6). A `top100_subset.parquet` (pre-curation) also exists — **do not mix** (see GLOSSARIO → curated).

---

### passage_entities_curated.parquet
**Path**: `data/NQ_answer/passage_entities_curated.parquet`
**Produced by**: `06_apply_curation.ipynb`  |  **Rows**: 90,667 passages (pool of the top-100 of the 1000 post-curation queries)

| column | type | meaning |
|---|---|---|
| `id` | int | passage_id (key into `top100_curated.passage_id`) |
| `title` | str | Wikipedia article title |
| `text` | str | passage content (used for the LLM prompt in 08) |
| `qids` | list[str] | Wikidata QIDs of the entities in the passage (ReFiNed) |

**Consumed by**: `07_kg_rerank.ipynb` (loads `id, qids`), `08_llm_eval.ipynb` (loads `id, title, text`).

---

### queries_curated.jsonl
**Path**: `data/NQ_answer/queries_curated.jsonl`
**Produced by**: `06_apply_curation.ipynb`  |  **Rows**: 1000 queries (post-curation)

| field | type | meaning |
|---|---|---|
| `question` | str | question text |
| `answers` | list[str] | gold answers (variants) — used as gold by the judge in 09 |
| `question_qids` | list[str] | QIDs of the entities in the question |
| `answer_variant_qids` | list[list[str]] | QIDs per answer variant |
| `original_query_id` | str | original NQ id (join key for pre/post-curation) |
| `curated` | bool | True if this query is a substitute introduced by curation |

**Consumed by**: `07_kg_rerank.ipynb` (question_qids = seed entities), `08_llm_eval.ipynb` (question text), `09_llm_judge.ipynb` (answers = gold).
**Notes**: curation SUBSTITUTED 344/1000 queries (not just filtered). Positionally indexed 0..999 = `query_id` in `top100_curated`.

---

### query_embeddings_curated.npy
**Path**: `data/NQ_answer/query_embeddings_curated.npy`
**Produced by**: `06_apply_curation.ipynb`  |  **Shape**: `(1000, 768)` float32

Contriever embeddings of the 1000 post-curation queries, row `i` = query `i` in `queries_curated.jsonl`. Re-encoded for the 344 substitute queries.
**Consumed by**: `07_kg_rerank.ipynb` (§2.4 binding validation: `q_emb · p_emb` must match `top100_curated.score`).

---

## 2 · Knowledge graph (Layer 1-3)

### edges.parquet
**Path**: `data/db/edges.parquet`
**Produced by**: `scripts/pipeline/hdt_export_per_predicate.py` (Layer 1, WSL2 pyHDT)  |  **Rows**: 661,471,158

| column | type | meaning |
|---|---|---|
| `subject` | string | source entity QID of the edge |
| `predicate` | string | relation QID/PID (e.g. "P31") |
| `object` | string | destination entity QID of the edge |

All Q-Q `wdt:*` triples from the Wikidata HDT dump.
**Consumed by**: `node_stats.py`, `build_n1.py`, `utils/kg.py` (table `edges` for dist=3 queries).
**Notes**: no index — DuckDB uses hash joins. The on-disk parquet has **3 columns** (subject, predicate, object); `utils/kg.py` **projects away `predicate`** when loading into the DuckDB table `edges` (only subject/object are needed for reachability) — so the DuckDB table has 2 columns while the file has 3.

---

### node_stats.parquet
**Path**: `data/db/node_stats.parquet`
**Produced by**: `scripts/pipeline/node_stats.py` (Layer 1.5, polars streaming)

| column | type | meaning |
|---|---|---|
| `qid` | str | Wikidata entity |
| `out_degree` | uint32 | # edges originating FROM this entity (subject) |
| `in_degree` | uint32 | # edges pointing TO this entity (object) |
| `total_degree` | uint32 | `in_degree + out_degree` |

Rows sorted by `total_degree` DESC (hubs first).
**Consumed by**: `build_n1.py` (annotates `neighbor_degree`), hub-threshold diagnostics.

---

### labels.parquet
**Path**: `data/db/labels.parquet`
**Produced by**: `scripts/pipeline/build_labels.py` (Layer 1.6, WSL2 pyHDT)

| column | type | meaning |
|---|---|---|
| `qid` | str | Wikidata entity |
| `label` | str | `rdfs:label@en` (human-readable name) |

For human-readable inspection only (e.g. Q42 → "Douglas Adams"). NOT in the scoring critical path.

---

### n1.parquet  →  DuckDB table `n1`
**Path**: `data/n1/n1.parquet`
**Produced by**: `scripts/pipeline/build_n1.py` (Layer 3, DuckDB)  |  **Rows**: ~93M

| column | type | meaning |
|---|---|---|
| `qid` | string | source entity |
| `neighbor` | string | entity **1 hop** away from `qid` |
| `neighbor_degree` | uint32 | `total_degree` of the neighbor — used for the hub filter (threshold) |

Precomputed 1-hop adjacency list for the QIDs of `seeds ∪ passage_entities`.
**Consumed by**: `utils/kg.py` (KGScorer — all dist=1/2/3 reachability queries).
**Notes**: B-tree index on `qid`. At **dist=1**, `neighbor_degree` is never consulted → the cell is threshold-invariant (see GLOSSARIO → n1).

---

### kg.duckdb
**Path**: `data/kg.duckdb`
**Produced by**: `utils/kg.py` (KGScorer, one-time build ~5 min)

DuckDB database persisted on disk, holds 2 tables: **`n1`** (see above, + index `idx_n1_qid`) and **`edges`** (see above, no index). Can be opened `read_only=True` from multi-process workers (shared OS page cache).

---

## 3 · KG-rerank (notebook 07)

### kg_pairs_raw.parquet  (Phase A)
**Path**: `data/NQ_answer/kg_pairs_raw.parquet`
**Produced by**: `07_kg_rerank.ipynb` (Phase A full run)  |  **Rows**: ~1.8M (1000 queries × 100 passages × 18 cells, minus unreachable pairs)

| column | type | meaning |
|---|---|---|
| `query_id` | str | positional query index (0..999) |
| `passage_id` | str | passage id |
| `distance` | int | KG distance of the cell (1, 2, 3) |
| `threshold` | int | hub-degree cutoff of the cell (500..10000, or **0 = ∞**) |
| `connected_ratio` | float | fraction of query-entities with ≥1 reachable doc-entity (cr) |
| `purity_ratio` | float | fraction of doc-entities near a query-entity (pr) |
| `kg_score` | float | `cr · pr` |

**Consumed by**: `08_llm_eval.ipynb` (filters by cell → `kg_score` for the rerank).
**Notes**: `threshold == 0` encodes ∞ (no filter). See `DEFAULT_THRESHOLDS` in `utils/kg.py`.

---

### kg_rerank_grid.parquet  (Phase B)
**Path**: `data/NQ_answer/kg_rerank_grid.parquet`
**Produced by**: `07_kg_rerank.ipynb` (Phase B)  |  **Rows**: 18 (3 dist × 6 thr)

| column | type | meaning |
|---|---|---|
| `distance` | int | distance of the cell |
| `threshold` | int | threshold of the cell (0 = ∞) |
| `pct_jacc_at_5_lt_1` | float | % of queries with Jaccard@5 < 1 (rerank changed the top-5) |
| `pct_jacc_at_10_lt_1` | float | % of queries with Jaccard@10 < 1 |
| `mean_jacc_at_5` | float | mean Jaccard@5 (retrieval vs rerank top-5) |
| `mean_jacc_at_10` | float | mean Jaccard@10 |

Diagnostic of "how much" each cell reshuffles the top-K vs pure retrieval (α=0.5).

---

## 4 · LLM evaluation (notebook 08-09)

### llm_eval_inputs.parquet
**Path**: `data/NQ_answer/llm_eval/llm_eval_inputs.parquet`
**Produced by**: `08_llm_eval.ipynb`  |  **Rows**: 91,000 (91 conditions × 1000 queries)

| column | type | meaning |
|---|---|---|
| `condition` | str | `retrieval` or `alpha_X_dist{D}_thr{T}` |
| `query_id` | str | positional query index |
| `question` | str | question text |
| `passage_ids` | list[str] | top-5 passage_ids for that (query, condition), score-descending order |
| `user_message` | str | full raw-completion prompt (instruction + few-shot + Q + Docs + "Answer:") |
| `n_tokens` | int | # tokens of the prompt (pre-flight; >4081 ⇒ overflow skip) |

**Consumed by**: section 6 of 08 itself (inference loop).

---

### llm_responses_{condition}.jsonl
**Path**: `data/NQ_answer/llm_eval/llm_responses_{condition}.jsonl`  (**91 files**, one per condition)
**Produced by**: `08_llm_eval.ipynb` (Llama-2-7b base)  |  **Rows/file**: 1000

| field | type | meaning |
|---|---|---|
| `query_id` | str | positional query index |
| `question` | str | question text |
| `passage_ids` | list[str] | top-5 used in the prompt |
| `response` | str \| null | answer generated by Llama-2 (null if prompt overflowed) |
| `error` | str | *(overflow rows only)* marker `prompt_overflow_n_tokens_{N}` |

**Consumed by**: `09_llm_judge.ipynb`.
**Notes**: append-mode, resumable (skip already-done query_ids).

---

### judgments_{condition}.jsonl
**Path**: `data/NQ_answer/llm_eval/judgments_{condition}.jsonl`  (**91 files**)
**Produced by**: `09_llm_judge.ipynb` (Qwen2.5-7B-Instruct)  |  **Rows/file**: 1000

| field | type | meaning |
|---|---|---|
| `query_id` | str | positional query index |
| `condition` | str | condition name |
| `response` | str | the Llama-2 response being judged |
| `gold_answers` | list[str] | gold answers from `queries_curated.answers` |
| `judge_raw` | str | raw judge output ("YES"/"NO"/...) |
| `verdict_bool` | bool \| null | True=CORRECT, False=INCORRECT, null=unparseable |

**Consumed by**: sections 7-9 of 09 (aggregation, plot, McNemar).
**Notes**: append-mode, resumable.

---

### judgments_summary.parquet
**Path**: `data/NQ_answer/llm_eval/judgments_summary.parquet`
**Produced by**: `09_llm_judge.ipynb`  |  **Rows**: 91 (index = condition)

| column | type | meaning |
|---|---|---|
| `n_true` | int | # CORRECT verdicts |
| `n_total` | int | # judgments (≈1000) |
| `accuracy` | float | `n_true / n_total` |
| `retrieval_baseline_acc` | float | accuracy of the `retrieval` condition (constant) |
| `delta_vs_retrieval` | float | `accuracy − retrieval_baseline_acc` |
| `chi2` | float | McNemar statistic vs retrieval (NaN for the retrieval row) |
| `p_value` | float | McNemar p-value |
| `sig@.05` | bool | True if `p_value < 0.05` |

Final analysis output: per-condition accuracy comparison + significance.

---

*Last updated: 2026-05-20*