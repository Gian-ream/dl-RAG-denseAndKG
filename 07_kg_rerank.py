# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07 — KG-rerank diagnostic (Step 5 of research proposal)
#
# ## Big picture
#
# **Pipeline position**:
# `01_corpus → 02_filter → 03_embed → 04_answer → 05_curate → 06_apply → **07_kg_rerank**`
#
# **Goal**: identify **which grid cells deserve LLM-eval campaigns** by
# comparing KG-based rerank against pure dense retrieval, sweeping max
# distance (`1, 2, 3` hops) and hub-degree threshold (`500, 1000, 2000,
# 5000, 10000, ∞`). 18 grid cells; α = 0.5 fixed.
#
# **Logic**: a cell where rerank barely changes the top-K compared to
# retrieval (`%jaccard@K < 1` low) is **not worth** an LLM campaign — the
# LLM would judge the same passages as the dense retrieval. A cell that
# substantially reshuffles the top-K is a candidate for LLM-eval.
#
# ## I/O
#
# **Inputs** (`data/NQ_answer/`):
# - `queries_curated.jsonl`            — 1 query per line, with `question_qids`
# - `passage_entities_curated.parquet` — `id, title, text, qids`
# - `top100_curated.parquet`           — 100K `(query_id, passage_id, score, rank)` candidates
#
# **Outputs**:
# - `data/NQ_answer/kg_pairs_raw.parquet`   — Phase A: `cr/pr/kg_score` per
#   `(query_id, passage_id, distance, threshold)` — 1.8M rows
# - `data/NQ_answer/kg_rerank_grid.parquet` — Phase B: 18 cells with
#   `pct_jacc_at_{5,10}_lt_1`, `mean_jacc_at_{5,10}`

# %% [markdown]
# ## 1 · Setup

# %%
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import KGScorer


# %%
# Walk up from this file to the directory containing pyproject.toml. Same
# pattern as every other script/notebook in this repo — keeps paths stable
# regardless of where the kernel was launched from.
def find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("Could not find repo root (pyproject.toml)")


REPO_ROOT = find_repo_root()
NQ_DIR = REPO_ROOT / "data" / "NQ_answer"

QUERIES_PATH = NQ_DIR / "queries_curated.jsonl"
PASSAGES_PATH = NQ_DIR / "passage_entities_curated.parquet"
TOP100_PATH = NQ_DIR / "top100_curated.parquet"

PHASE_A_OUT = NQ_DIR / "kg_pairs_raw.parquet"
PHASE_B_OUT = NQ_DIR / "kg_rerank_grid.parquet"

print(f"REPO_ROOT: {REPO_ROOT}")
for p in [QUERIES_PATH, PASSAGES_PATH, TOP100_PATH]:
    flag = "OK" if p.exists() else "MISSING"
    print(f"  [{flag}] {p.relative_to(REPO_ROOT)}")

# %% [markdown]
# ## 2 · Load input data

# %% [markdown]
# ### 2.1 · Queries (`queries_curated.jsonl`)
#
# JSONL = one JSON object per line. Each line is independent — easy to
# stream and to grep. We materialize everything into a list because the
# file is small (~1000 queries).

# %%
# Peek the raw file: read just the first line, no parsing yet.
with QUERIES_PATH.open(encoding="utf-8") as f:
    sample_raw_line = f.readline()
print(sample_raw_line[:400])

# %%
# Bulk load. List index 0..N-1 corresponds to `query_id` in top100_curated
# (verified by the assert in section 2.3).
queries: list[dict] = [json.loads(line) for line in QUERIES_PATH.open(encoding="utf-8")]
print(f"queries loaded: {len(queries):,}")

# %%
queries[0]

# %%
# Transform: lookup query_id (int) -> list[str] of question_qids.
# Empty question_qids stays empty: those pairs will yield kg_score=0
# (zero-padding handled by kg_components_grid).
qids_by_query: list[list[str]] = [(q.get("question_qids") or []) for q in queries]
n_with_qids = sum(1 for qs in qids_by_query if qs)
print(f"queries with at least one question_qid: {n_with_qids:,} / {len(queries):,}")

# %% [markdown]
# ### 2.2 · Passage entities (`passage_entities_curated.parquet`)

# %%
# pq.read_table(path, columns=[...]) → pyarrow.Table
#   Reads ONLY the listed columns from disk (column pruning at parquet
#   level → faster than reading everything when we don't need title/text).
# .to_pandas() → pandas.DataFrame
#   Zero-copy bridge for primitive columns; list columns become object
#   dtype where each cell holds a Python list.
passages_df = pq.read_table(PASSAGES_PATH, columns=["id", "qids"]).to_pandas()
print(f"passages loaded: {len(passages_df):,}")

# %%
passages_df.head(1)

# %%
# Demo of df.itertuples(index=False) before using it in the bulk transform.
# itertuples yields one namedtuple per row. Faster than iterrows() (which
# wraps each row in a Series) and gives attribute access.
sample_row = next(passages_df.itertuples(index=False))
print(f"type:       {type(sample_row).__name__}")
print(f"row.id:     {sample_row.id}  (type: {type(sample_row.id).__name__})")
print(f"row.qids:   {list(sample_row.qids)[:3]}...  (type: {type(sample_row.qids).__name__})")

# %%
# Bulk transform: dict comprehension over itertuples → passage_id -> list[QID]
qids_by_passage: dict[int, list[str]] = {
    int(row.id): list(row.qids) for row in passages_df.itertuples(index=False)
}
n_pass_with_qids = sum(1 for v in qids_by_passage.values() if v)
print(f"passages with at least one qid: {n_pass_with_qids:,}")

# %% [markdown]
# ### 2.3 · Candidate pairs (`top100_curated.parquet`)

# %%
# pq.read_table(path) without `columns=` → reads ALL columns from disk.
top100 = pq.read_table(TOP100_PATH).to_pandas()
print(f"top100 rows: {len(top100):,}")

# %%
top100.head(1)

# %%
# Diagnostics: dtypes, query_id range, missing passage_ids
print("dtypes:")
print(top100.dtypes)

qid_min, qid_max = top100["query_id"].min(), top100["query_id"].max()
assert qid_min >= 0 and qid_max < len(queries), (
    f"query_id range [{qid_min}, {qid_max}] incompatible with "
    f"len(queries) = {len(queries)}"
)
print(f"\nquery_id range: [{qid_min}, {qid_max}]  (within [0, {len(queries) - 1}])")

top100_pids = set(top100["passage_id"].astype(int).unique())
missing_pids = top100_pids - set(qids_by_passage.keys())
print(f"distinct passage_ids in top100: {len(top100_pids):,}")
print(f"  missing from passage_entities (D=[]): {len(missing_pids):,}")

# %% [markdown]
# ### 2.4 · End-to-end validation — recompute similarity for ALL pairs
#
# Strongest possible integrity check: for **every** (query, passage) pair
# in `top100_curated`, recompute the similarity from stored embeddings and
# verify it matches the stored `score`. Validates simultaneously:
#
# - **Positional binding**: `query_id=X ↔ queries[X] ↔ query_embeddings_curated[X]`
# - **Score integrity**: scores in `top100_curated` are the actual retriever
#   similarities, not corrupted by the curation step
#
# **Similarity formula** (verified by reading the encoding pipeline):
# - Passage encoding (`03_embedding.ipynb`): `mean_pooling` over Contriever
#   outputs, **no L2 normalization**. Norms ≈ 0.9-1.1 by model nature.
# - Query encoding (`04_answer_preparation.ipynb`): identical pipeline.
# - FAISS index: `IndexFlatIP` ⇒ stored score is **raw dot product**
#   `q_emb · p_emb`, not cosine.
#
# So the recompute is plain `np.dot(q, p)`, no normalization.

# %%
# Load query embeddings (one row per query, positional order = queries_curated).
QUERY_EMB_PATH = NQ_DIR / "query_embeddings_curated.npy"
FAISS_INDEX_DIR = REPO_ROOT / "data" / "faiss_index"

query_embs = np.load(QUERY_EMB_PATH)
print(f"query_embeddings_curated.npy shape: {query_embs.shape}")
print(f"  norm of [0]: {np.linalg.norm(query_embs[0]):.4f}  "
      f"(≠1 confirms NO L2 normalization — score is raw dot product)")

assert query_embs.shape[0] == len(queries), (
    f"query embedding count {query_embs.shape[0]} != len(queries) {len(queries)} "
    f"→ embedding file out of sync with queries jsonl"
)

# %%
# Per-shard processing: load each shard's embeddings ONCE (mmap), resolve
# all (query, passage) lookups inside the shard via vectorized searchsorted,
# then compute dot products in a single matmul-like operation.
#
# We collect per-shard diffs and aggregate at the end.
diffs_per_shard: list[np.ndarray] = []
n_total = len(top100)

for sid in sorted(top100["shard_id"].unique()):
    rows_in_shard = top100[top100["shard_id"] == sid]
    n_in_shard = len(rows_in_shard)

    emb_path = FAISS_INDEX_DIR / f"shard_{int(sid):02d}.npy"
    ids_path = FAISS_INDEX_DIR / f"shard_{int(sid):02d}_ids.npy"

    # mmap_mode="r" → file mappato; carichiamo in RAM solo le righe accedute.
    # Senza mmap caricheremmo ~1.6 GB di shard ad ogni iterazione.
    embs_shard = np.load(emb_path, mmap_mode="r")
    ids_shard = np.load(ids_path)

    # passage_id → position lookup vectorizzato (vs dict Python lento da costruire
    # su 5M righe). argsort + searchsorted è O((M + N) log M) totale.
    sorted_idx = np.argsort(ids_shard, kind="stable")
    sorted_ids = ids_shard[sorted_idx]

    pids_to_find = rows_in_shard["passage_id"].values
    pos_in_sorted = np.searchsorted(sorted_ids, pids_to_find)
    # Sanity: tutti i passage_id devono essere stati trovati nella loro shard
    assert (sorted_ids[pos_in_sorted] == pids_to_find).all(), (
        f"shard {sid}: some passage_ids declared in top100 are missing from "
        f"the shard's id list"
    )
    pid_positions = sorted_idx[pos_in_sorted]  # posizioni nello shard originale

    # Fancy indexing su mmap → triggera la lettura di SOLO n_in_shard × 768 floats
    # (≈30 MB per shard); np.asarray forza la materializzazione in RAM continua.
    p_embs = np.asarray(embs_shard[pid_positions])           # (n_in_shard, 768)
    q_embs = query_embs[rows_in_shard["query_id"].values]    # (n_in_shard, 768)

    # Row-wise dot product vectorizzato: elementwise product + sum sull'asse-768.
    # Equivalente a fare np.dot per ogni riga ma in una sola call BLAS.
    recomputed = (q_embs * p_embs).sum(axis=1)               # (n_in_shard,)

    stored = rows_in_shard["score"].values.astype(np.float32)
    diffs = np.abs(stored - recomputed.astype(np.float32))
    diffs_per_shard.append(diffs)

    print(f"shard {int(sid):02d}: {n_in_shard:>6,} rows   "
          f"max|Δ|={diffs.max():.2e}   mean|Δ|={diffs.mean():.2e}")

    # Cleanup esplicito per liberare le mmap views e gli array intermedi
    del embs_shard, ids_shard, sorted_idx, sorted_ids, p_embs, q_embs, recomputed

# %%
# Global aggregation + assertion
all_diffs = np.concatenate(diffs_per_shard)
TOLERANCE = 1e-4
n_violations = int((all_diffs > TOLERANCE).sum())

print(f"\nGlobal stats across {n_total:,} (query, passage) pairs:")
print(f"  max  |Δ score|: {all_diffs.max():.2e}")
print(f"  mean |Δ score|: {all_diffs.mean():.2e}")
print(f"  p99  |Δ score|: {np.percentile(all_diffs, 99):.2e}")
print(f"  violations (|Δ| > {TOLERANCE:.0e}): {n_violations:,}")

assert n_violations == 0, (
    f"{n_violations} pairs have recomputed score diverging from stored by "
    f"> {TOLERANCE:.0e}. Either the positional binding "
    f"(query_id ↔ queries[X] ↔ query_embeddings_curated[X]) is wrong, or "
    f"scores were corrupted by the curation step. Investigate before Phase A."
)
print(f"\n✓ ALL {n_total:,} (query, passage) similarities reproducible "
      f"from stored embeddings → binding & scores valid")

# %% [markdown]
# ## 3 · Init `KGScorer`
#
# Opens `data/kg.duckdb` in read-write with idempotent auto-build:
# - File exists with `n1` + `edges` tables → init ~1s
# - Missing → builds from `data/n1/n1.parquet` + `data/db/edges.parquet` (~5 min, one-time)

# %%
t0 = time.perf_counter()
scorer = KGScorer()
print(f"\nKGScorer init: {time.perf_counter() - t0:.1f}s")

# %% [markdown]
# ## 4 · Phase A smoke test — OLD API on query 0 × 100 passages
#
# We pick query_id=0 and ALL its 100 candidate passages from the retriever.
# This single-query layout is the input we'll reuse in section 4b for the
# per-query batched API comparison (apples-to-apples timing).
#
# Old API (`kg_components_grid_batch`) treats each (query, passage) pair
# independently → 6 SQL calls per pair × 100 pairs = 600 SQL total.
#
# **Estimated time**: ~7-8 sec/pair × 100 ≈ 12-15 min.

# %%
SMOKE_QUERY_ID = 0
smoke_top100 = top100[top100["query_id"] == SMOKE_QUERY_ID].sort_values("rank")
print(f"smoke pairs to score: {len(smoke_top100)}")
print(f"  query_id:        {SMOKE_QUERY_ID}")
print(f"  question:        {queries[SMOKE_QUERY_ID].get('question', '<no text>')[:80]}")
print(f"  question_qids Q: {qids_by_query[SMOKE_QUERY_ID]}")

# %%
smoke_top100.head(1)

# %%
# Demo: build ONE tuple (query_id, Q, passage_id, D) before the bulk list.
# Each tuple holds: stringified ids (for SQL match) plus the entity lists
# pulled from the lookups built in section 2.
_demo_row = next(smoke_top100.itertuples(index=False))
_demo_pair = (
    str(int(_demo_row.query_id)),
    qids_by_query[int(_demo_row.query_id)],
    str(int(_demo_row.passage_id)),
    qids_by_passage.get(int(_demo_row.passage_id), []),
)
print("one pair tuple:")
print(f"  query_id   = {_demo_pair[0]}")
print(f"  Q          = {_demo_pair[1]}")
print(f"  passage_id = {_demo_pair[2]}")
print(f"  D ({len(_demo_pair[3])}) = {_demo_pair[3][:5]}{'...' if len(_demo_pair[3]) > 5 else ''}")

# %%
# Bulk: build the list of 100 pairs (all sharing query_id=0)
smoke_pairs: list[tuple[str, list[str], str, list[str]]] = [
    (
        str(int(row.query_id)),
        qids_by_query[int(row.query_id)],
        str(int(row.passage_id)),
        qids_by_passage.get(int(row.passage_id), []),
    )
    for row in smoke_top100.itertuples(index=False)
]
print(f"smoke_pairs built: {len(smoke_pairs)}")

# %%
# scorer.kg_components_grid_batch(pairs, verbose=True)
#   Input:  list of tuples (query_id: str, Q: list[QID], passage_id: str, D: list[QID])
#   Output: pd.DataFrame with columns
#     [query_id, passage_id, distance, threshold, connected_ratio, purity_ratio, kg_score]
#     Exactly len(pairs) * 18 rows (3 distances × 6 thresholds per pair).
#   Internals: for each pair, runs 6 SQL queries on data/kg.duckdb (one per
#     threshold). Each query returns reachable (q, d) pairs at min hop
#     distance ≤ 3. Python derives cr/pr/kg_score per (distance, threshold).
#   verbose=True → prints "processed N/M pairs" every 50 pairs.
print("Running OLD API (kg_components_grid_batch)...")
t0 = time.perf_counter()
df_smoke_old = scorer.kg_components_grid_batch(smoke_pairs, verbose=True)
old_elapsed = time.perf_counter() - t0
print(f"\nOLD: {old_elapsed:.1f}s ({old_elapsed / len(smoke_pairs) * 1000:.0f} ms/pair)")

# %%
df_smoke_old.head(1)

# %%
# Diagnostics: shape + extrapolation with OLD API
print(f"df_smoke_old.shape: {df_smoke_old.shape}  (expected: ({len(smoke_pairs) * 18}, 7))")
n_queries_full = top100["query_id"].nunique()
est_full_min_old = old_elapsed * n_queries_full / 60
print(f"\nExtrapolated Phase A full run, OLD API:")
print(f"  {n_queries_full:,} queries × {old_elapsed:.0f}s/query ≈ {est_full_min_old:,.0f} min "
      f"(~{est_full_min_old / 60:.1f} h)")

# %% [markdown]
# ## 4b · Per-query smoke — NEW API + equivalence check
#
# `kg_components_grid_per_query` exploits a structural redundancy: the 100
# passages of one query share the same `Q`. Instead of `_reachable_pairs_min_dist`
# being called 600 times (100 pair × 6 threshold), it's called **6 times**
# — one per threshold, with `D_union = ⋃ D_i` (~200-1500 entities, dedup).
# Per-passage filtering happens in Python on the cached reach map (free).
#
# **Same input data as section 4** — apples-to-apples timing.
#
# **Expected speedup**: 50-100x vs OLD API.

# %%
# Reorganize the SAME 100 pairs into the (passages_list) format the new API expects.
# The new API takes (query_id, Q, list[(passage_id, D)]) — Q is shared across
# all passages, no need to repeat it per-pair.
smoke_passages: list[tuple[str, list[str]]] = [
    (str(int(row.passage_id)), qids_by_passage.get(int(row.passage_id), []))
    for row in smoke_top100.itertuples(index=False)
]
Q = qids_by_query[SMOKE_QUERY_ID]
print(f"|Q|             = {len(Q)}")
print(f"|passages|      = {len(smoke_passages)}")
print(f"Q               = {Q}")

# Quick stat: |D_union| (the bound on the SQL result set per threshold)
D_union_size = len({qid for _, D in smoke_passages for qid in D})
print(f"|D_union|       = {D_union_size}  (bound on per-threshold SQL result)")

# %%
# scorer.kg_components_grid_per_query(query_id, Q, passages, verbose=True)
#   Input:  query_id (str), Q (list[QID]), passages (list[(passage_id, D)])
#   Output: pd.DataFrame with SAME columns as kg_components_grid_batch
#     [query_id, passage_id, distance, threshold, connected_ratio, purity_ratio, kg_score]
#     Same row count: len(passages) * 18.
#   Internals: 1 SQL per threshold (6 total), with D = D_union (∪ D_i).
#     verbose=True → log per (query, threshold): timing + |reach| + Q's qids.
print("Running NEW API (kg_components_grid_per_query)...")
t0 = time.perf_counter()
df_smoke_new = scorer.kg_components_grid_per_query(
    query_id=str(SMOKE_QUERY_ID),
    Q=Q,
    passages=smoke_passages,
    verbose=True,
)
new_elapsed = time.perf_counter() - t0
print(f"\nNEW: {new_elapsed:.1f}s")

# %%
df_smoke_new.head(1)

# %%
print(f"df_smoke_new.shape: {df_smoke_new.shape}  (expected: ({len(smoke_pairs) * 18}, 7))")

# %% [markdown]
# ### Equivalence check — same numbers in both DataFrames?
#
# Both APIs should produce identical (cr, pr, kg_score) for every
# (query, passage, distance, threshold). Order may differ → sort first,
# then compare row-by-row.

# %%
# Sort both by (passage_id, distance, threshold) for row-by-row alignment.
sort_cols = ["passage_id", "distance", "threshold"]
df_old_sorted = df_smoke_old.sort_values(sort_cols).reset_index(drop=True)
df_new_sorted = df_smoke_new.sort_values(sort_cols).reset_index(drop=True)

# Max abs diff per metric column. Should all be < 1e-9 (float noise only).
for col in ("connected_ratio", "purity_ratio", "kg_score"):
    diff = (df_old_sorted[col] - df_new_sorted[col]).abs().max()
    print(f"  max |Δ {col:<16}| = {diff:.2e}")

# Hard assert: equivalence within float precision
assert (df_old_sorted["connected_ratio"] - df_new_sorted["connected_ratio"]).abs().max() < 1e-9
assert (df_old_sorted["purity_ratio"]    - df_new_sorted["purity_ratio"]   ).abs().max() < 1e-9
assert (df_old_sorted["kg_score"]        - df_new_sorted["kg_score"]       ).abs().max() < 1e-9
print("\n✓ OUTPUT EQUIVALENT (all metric diffs < 1e-9)")

# %% [markdown]
# ### Speedup + extrapolation to full Phase A

# %%
speedup = old_elapsed / new_elapsed
print(f"OLD: {old_elapsed:>8.1f}s   ({old_elapsed / len(smoke_pairs) * 1000:>6.0f} ms/pair)")
print(f"NEW: {new_elapsed:>8.1f}s   ({new_elapsed / len(smoke_pairs) * 1000:>6.0f} ms/pair-equivalent)")
print(f"\nSpeedup: {speedup:>5.1f}x")

est_full_min_new = new_elapsed * n_queries_full / 60
print(f"\nExtrapolation to {n_queries_full:,} queries (full Phase A):")
print(f"  OLD: ~{est_full_min_old:>8,.0f} min  (~{est_full_min_old / 60:>5.1f} h)")
print(f"  NEW: ~{est_full_min_new:>8,.0f} min  (~{est_full_min_new / 60:>5.1f} h)")

# %% [markdown]
# ## 5 · Phase A — Full run (per-query batched)
#
# Uses `kg_components_grid_per_query`: groups `top100` by `query_id` and
# calls the new API once per query. ~32x faster than per-pair batching
# (verified in section 4b smoke). Estimated ~8 hours for the full 1000
# queries on this machine.
#
# Pre-check: skip if `kg_pairs_raw.parquet` already exists (idempotent).
#
# **Knob `MAX_QUERIES`**: positive int for a staged run (e.g. first 50
# queries) before committing to the full ~8h run. `None` = all queries.
#
# **No checkpointing**: a crash means starting over. Acceptable risk for
# a one-off ~8h run; if it becomes recurring we'll add incremental save.

# %%
MAX_QUERIES: int | None = None  # None = all queries; int to subsample

if PHASE_A_OUT.exists():
    print(f"Phase A output already present: {PHASE_A_OUT}")
    print(f"  size: {PHASE_A_OUT.stat().st_size / 1024 / 1024:.1f} MB")
    # pd.read_parquet(path) → DataFrame, schema auto-detected.
    # Default engine is pyarrow; compression codec is decoded transparently.
    df_phase_a = pd.read_parquet(PHASE_A_OUT)
    print(f"  rows: {len(df_phase_a):,}")
else:
    # df.groupby(col, sort=True) → DataFrameGroupBy object.
    #   sort=True (default) → groups iterated in ascending key order.
    #   .groups → dict mapping each group key to the row indices in that group.
    # We materialize the keys here so we can subsample (MAX_QUERIES) and
    # report progress against a known total.
    grouped = top100.groupby("query_id", sort=True)
    all_query_ids: list[int] = list(grouped.groups.keys())
    if MAX_QUERIES is not None:
        all_query_ids = all_query_ids[:MAX_QUERIES]

    n_queries = len(all_query_ids)
    n_pairs_total = sum(len(grouped.get_group(q)) for q in all_query_ids)
    print(f"Total queries to process: {n_queries:,}")
    print(f"Total pairs implied:      {n_pairs_total:,}")

    # Per-query DataFrames; concat at the end. Concatenating once is
    # cheaper than appending to a single DataFrame inside the loop.
    sub_dfs: list[pd.DataFrame] = []
    t_total_start = time.perf_counter()

    for idx, query_id in enumerate(all_query_ids):
        # Pull the rows of top100 belonging to THIS query (typically 100).
        q_rows = grouped.get_group(query_id)

        # Build per-query input for the new API: list[(passage_id_str, D)].
        # Q is the same for all 100 candidates, so we pass it once outside.
        passages_list: list[tuple[str, list[str]]] = [
            (
                str(int(row.passage_id)),
                qids_by_passage.get(int(row.passage_id), []),
            )
            for row in q_rows.itertuples(index=False)
        ]
        Q = qids_by_query[int(query_id)]

        # Call the per-query batched API. verbose=False here: 1000 queries
        # × 6 threshold log lines = 6000 lines, too noisy. We do our own
        # progress report every N queries below.
        sub = scorer.kg_components_grid_per_query(
            query_id=str(int(query_id)),
            Q=Q,
            passages=passages_list,
            verbose=False,
        )
        sub_dfs.append(sub)

        # Progress report every 10 queries: elapsed + ETA + total estimate.
        if (idx + 1) % 10 == 0 or (idx + 1) == n_queries:
            elapsed_min = (time.perf_counter() - t_total_start) / 60
            est_total_min = elapsed_min / (idx + 1) * n_queries
            eta_min = est_total_min - elapsed_min
            print(
                f"  [{idx + 1:>4}/{n_queries}]  "
                f"elapsed {elapsed_min:>6.1f} min  "
                f"ETA {eta_min:>6.1f} min  "
                f"(total est ~{est_total_min:>6.1f} min, "
                f"~{est_total_min / 60:.1f} h)",
                flush=True,
            )

    # Single concat → one allocation. ignore_index=True renumbers rows
    # 0..N-1 (pd.concat preserves original indices by default, which would
    # produce duplicate indices since each sub_df is 0..1799).
    df_phase_a = pd.concat(sub_dfs, ignore_index=True)
    elapsed_min = (time.perf_counter() - t_total_start) / 60
    print(f"\nPhase A complete in {elapsed_min:.1f} min ({elapsed_min / 60:.1f} h)")
    print(f"  rows: {len(df_phase_a):,}  (expected: {n_pairs_total * 18:,})")

    # df.to_parquet(path, compression=...) → writes a single parquet file.
    # snappy is the fast default; "zstd" gives smaller files at higher CPU cost.
    df_phase_a.to_parquet(PHASE_A_OUT, compression="snappy")
    print(f"Saved: {PHASE_A_OUT}  ({PHASE_A_OUT.stat().st_size / 1024 / 1024:.1f} MB)")

# %%
df_phase_a.head(1)

# %%
print(f"df_phase_a.shape: {df_phase_a.shape}")

# %% [markdown]
# ## 6 · Phase B — Jaccard@5 / Jaccard@10 grid
#
# For each cell `(distance, threshold)`:
# 1. For each query, compute `final_score = 0.5·dense_norm + 0.5·kg_score`
#    over all ~100 candidates.
# 2. Re-sort by `final_score` descending → top-K rerank set.
# 3. Compare with top-K retrieval set (original `rank` order) via Jaccard.
# 4. Aggregate over all queries: `pct_jacc_at_K_lt_1`, `mean_jacc_at_K`.
#
# **Dense score normalization**: per-query min-max → `dense_norm ∈ [0, 1]`,
# same range as `kg_score`. Degenerate case (all scores equal) → fallback 0.5.

# %%
# Add string-typed id columns for joining with df_phase_a (KGScorer stores str).
top100["query_id_s"] = top100["query_id"].astype(str)
top100["passage_id_s"] = top100["passage_id"].astype(str)


def min_max_per_group(s: pd.Series) -> pd.Series:
    """Max-only scaling: divide by per-group max, no shift.
    Preserves the relative shape (rank-1 → 1.0, others → score/max).
    """
    hi = s.max()
    if hi < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return s / hi


# Demo of groupby(col)[t].transform(fn) before the bulk apply.
# transform() applies fn to each group's values of `t` and BROADCASTS the
# result back so the output has the SAME index/length as the original df.
# Compare with .apply(): apply collapses to one value per group; transform
# keeps the row alignment. Perfect for "add a per-group normalized column".
_demo_q = top100[top100["query_id"] == 0]["score"]
_demo_q_norm = min_max_per_group(_demo_q)
print(f"query_id=0  raw scores [first 5]:    {_demo_q.head().round(3).tolist()}")
print(f"query_id=0  normalized [first 5]:    {_demo_q_norm.head().round(3).tolist()}")
print(f"query_id=0  raw range:        [{_demo_q.min():.3f}, {_demo_q.max():.3f}]")
print(f"query_id=0  normalized range: [{_demo_q_norm.min():.3f}, {_demo_q_norm.max():.3f}]")

# %%
# Bulk: apply per-query min-max via groupby.transform
top100["dense_norm"] = top100.groupby("query_id")["score"].transform(min_max_per_group)

# %%
top100.head(1)

# %%
print(f"dense_norm range: [{top100['dense_norm'].min():.3f}, {top100['dense_norm'].max():.3f}]")


# %%
def jaccard(a: set, b: set) -> float:
    """Jaccard set similarity, robust to empty sets."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# %% [markdown]
# Now we walk through the metric computation on **one cell** of the grid
# (e.g. `distance=2, threshold=5000`) before wrapping it in a function and
# looping over all 18 cells.

# %%
# Pick one slice of df_phase_a: rows for distance=2, threshold=5000
_demo_cell = df_phase_a[
    (df_phase_a["distance"] == 2) & (df_phase_a["threshold"] == 5000)
]
print(f"_demo_cell rows: {len(_demo_cell):,}  (expected: ~{len(top100):,})")
_demo_cell.head(1)

# %%
# Demo of base.merge(other, left_on, right_on, how, suffixes):
#   - SQL-style join: matches rows where base[left_on] == other[right_on]
#     (the join keys can be DIFFERENT column names on each side)
#   - how="left": keep ALL rows of `base`; missing right-side columns → NaN
#   - suffixes=("", "_kg"): both sides have same-named columns
#     (`query_id`, `passage_id`) → suffix the right-side ones with "_kg"
_demo_merged = top100.merge(
    _demo_cell[["query_id", "passage_id", "kg_score"]],
    left_on=["query_id_s", "passage_id_s"],
    right_on=["query_id", "passage_id"],
    how="left",
    suffixes=("", "_kg"),
)
# fillna(0.0): passages absent from _demo_cell (empty D, or all unreachable)
# get NaN from the LEFT join → coerce to 0.0 for the arithmetic below.
_demo_merged["kg_score"] = _demo_merged["kg_score"].fillna(0.0)
_demo_merged["final_score"] = 0.5 * _demo_merged["dense_norm"] + 0.5 * _demo_merged["kg_score"]
_demo_merged.head(1)

# %%
# Demo of nsmallest/nlargest on ONE query (query_id=0):
#   df.nsmallest(k, col) → DataFrame of the k rows with the smallest `col`
#     values, already ordered ascending. O(N log k), faster than sort + head.
#   df.nlargest(k, col)  → analogous for largest values, descending.
_one_q = _demo_merged[_demo_merged["query_id_s"] == "0"]
_top5_retrieval = set(_one_q.nsmallest(5, "rank")["passage_id_s"])
_top5_rerank = set(_one_q.nlargest(5, "final_score")["passage_id_s"])
print(f"query 0 — retrieval top-5: {_top5_retrieval}")
print(f"query 0 — rerank    top-5: {_top5_rerank}")
print(f"query 0 — Jaccard@5      : {jaccard(_top5_retrieval, _top5_rerank):.3f}")


# %%
def compute_cell_metrics(
    df_cell: pd.DataFrame,
    base: pd.DataFrame,
    k_values: tuple[int, ...] = (5, 10),
) -> dict:
    """Compute Jaccard@K metrics for ONE grid cell, aggregated across queries.

    df_cell:  Phase A rows for fixed (distance, threshold).
    base:     top100 plus dense_norm (for join + reranking).
    """
    merged = base.merge(
        df_cell[["query_id", "passage_id", "kg_score"]],
        left_on=["query_id_s", "passage_id_s"],
        right_on=["query_id", "passage_id"],
        how="left",
        suffixes=("", "_kg"),
    )
    merged["kg_score"] = merged["kg_score"].fillna(0.0)
    merged["final_score"] = 0.5 * merged["dense_norm"] + 0.5 * merged["kg_score"]

    metrics: dict = {}
    for k in k_values:
        per_query = []
        # groupby(col, sort=False) → (group_key, group_df) iterator.
        # sort=False skips the costly key sort (we aggregate per-group anyway).
        for _, g in merged.groupby("query_id_s", sort=False):
            top_retrieval = set(g.nsmallest(k, "rank")["passage_id_s"])
            top_rerank = set(g.nlargest(k, "final_score")["passage_id_s"])
            per_query.append(jaccard(top_retrieval, top_rerank))
        arr = np.array(per_query)
        # (arr < scalar) → boolean numpy array; .mean() on bools = fraction True.
        metrics[f"pct_jacc_at_{k}_lt_1"] = float((arr < 1.0).mean())
        metrics[f"mean_jacc_at_{k}"] = float(arr.mean())
    return metrics


# %%
# Loop over the 18 cells. df.groupby([c1, c2]) yields ((v1, v2), sub_df) tuples.
# Expect ~5-30s per cell (groupby on ~1000 queries inside compute_cell_metrics).
cell_results: list[dict] = []
for (dist, thr), df_cell in df_phase_a.groupby(["distance", "threshold"]):
    t0 = time.perf_counter()
    m = compute_cell_metrics(df_cell, top100, k_values=(5, 10))
    m["distance"] = int(dist)
    m["threshold"] = int(thr) if thr != 0 else 0
    cell_results.append(m)
    print(
        f"dist={dist} thr={thr:>5}: "
        f"%j5<1={m['pct_jacc_at_5_lt_1']:.3f} "
        f"mean_j5={m['mean_jacc_at_5']:.3f} "
        f"%j10<1={m['pct_jacc_at_10_lt_1']:.3f} "
        f"mean_j10={m['mean_jacc_at_10']:.3f}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

df_grid = pd.DataFrame(cell_results)[
    [
        "distance", "threshold",
        "pct_jacc_at_5_lt_1", "mean_jacc_at_5",
        "pct_jacc_at_10_lt_1", "mean_jacc_at_10",
    ]
]
df_grid.to_parquet(PHASE_B_OUT, compression="snappy")
print(f"\nSaved: {PHASE_B_OUT}")

# %%
df_grid.head(1)

# %%
print(f"df_grid.shape: {df_grid.shape}  (expected: (18, 6))")

# %% [markdown]
# ## 7 · Grid visualization
#
# Pivot `distance × threshold` for each of the 4 metrics. Cells with high
# `pct_jacc_at_K_lt_1` AND low `mean_jacc_at_K` are the priority candidates
# for LLM-eval — rerank changes things often, and substantially.

# %%
# df.pivot(index, columns, values) → wide-form DataFrame (reshape long→wide)
#   - One row per unique value of `index`
#   - One column per unique value of `columns`
#   - Cell = the `values` column at that (index, columns) intersection
#   Here: 18 long rows → 3 (distances) × 6 (thresholds) wide table.
df_grid.pivot(index="distance", columns="threshold", values="pct_jacc_at_5_lt_1")

# %%
df_grid.pivot(index="distance", columns="threshold", values="mean_jacc_at_5")

# %%
df_grid.pivot(index="distance", columns="threshold", values="pct_jacc_at_10_lt_1")

# %%
df_grid.pivot(index="distance", columns="threshold", values="mean_jacc_at_10")

# %% [markdown]
# ## 8 · α sweep — pre-screening for LLM-eval campaign
#
# The Phase B grid above used α=0.5 fixed. Before committing LLM budget
# to the campaign, sweep α ∈ {0.1, 0.2, ..., 0.9} on ONE cell of the
# grid (`distance=3, threshold=10000` — best recall/precision trade-off
# from section 6) and answer:
#
# 1. **How much does each α deviate from pure retrieval?** (`mean_jacc@5` +
#    `%j@5<1` vs the dense baseline)
# 2. **Which α values give equivalent reranks?** (pairwise Jaccard@5
#    matrix between all α pairs)
#
# **Formula**: `final_score(α) = (1 − α) · dense_norm + α · kg_score`
#   - α=0   → pure retrieval (dense only, no KG influence)
#   - α=0.5 → equal weight (= the Phase B configuration above)
#   - α=1   → pure KG (no dense influence)
#
# **Output use**: pairs (α_i, α_j) with high jaccard ⇒ they produce
# essentially the same top-5 ⇒ ridondanti, scegline uno per LLM-eval.
# Pairs with low jaccard ⇒ rerank distinti ⇒ entrambi candidati. Riduce
# 9 condizioni potenziali a un set rappresentativo (es. 3-4) sotto
# budget LLM realistico.

# %%
ALPHA_CELL_DIST = 3
ALPHA_CELL_THR = 10000

# Pull the (query, passage) → kg_score for the chosen cell.
df_cell_alpha = df_phase_a[
    (df_phase_a["distance"] == ALPHA_CELL_DIST)
    & (df_phase_a["threshold"] == ALPHA_CELL_THR)
]
print(f"cell selected: dist={ALPHA_CELL_DIST}, thr={ALPHA_CELL_THR}")
print(f"df_cell_alpha shape: {df_cell_alpha.shape}  "
      f"(expected: ({top100['query_id'].nunique() * 100}, 7))")

# %%
df_cell_alpha.head(1)

# %%
# Merge with top100 → per-pair (query_id_s, passage_id_s, score, dense_norm, kg_score).
# Stesso pattern LEFT join di compute_cell_metrics (sezione 6); fillna(0) sui pair
# senza kg_score (passage senza entità o non raggiunti dalla SQL).
base_alpha = top100[["query_id_s", "passage_id_s", "score", "dense_norm"]].merge(
    df_cell_alpha[["query_id", "passage_id", "kg_score"]],
    left_on=["query_id_s", "passage_id_s"],
    right_on=["query_id", "passage_id"],
    how="left",
    suffixes=("", "_kg"),
)
base_alpha["kg_score"] = base_alpha["kg_score"].fillna(0.0)
print(f"base_alpha shape: {base_alpha.shape}")

# %%
base_alpha.head(1)

# %%
# Genera 9 colonne kg_final_alpha_X dove X = α · 10 (suffisso intero).
# α = peso del segnale KG, (1-α) = peso dense_norm. α=0.1 → kg_final_alpha_1.
ALPHAS = [round(0.1 * i, 1) for i in range(1, 10)]  # [0.1, 0.2, ..., 0.9]
for alpha in ALPHAS:
    col = f"kg_final_alpha_{int(round(alpha * 10))}"
    base_alpha[col] = (1 - alpha) * base_alpha["dense_norm"] + alpha * base_alpha["kg_score"]

print(f"alpha columns added: "
      f"{[c for c in base_alpha.columns if c.startswith('kg_final_alpha')]}")

# %%
base_alpha.head(1)

# %%
# Per-query top-5 set per ogni "setting" (retrieval baseline + 9 alpha).
# Cache come dict {label → {query_id_s → set(passage_id_s)}} per pairwise lookup veloce.
# Costo: 10 × groupby(1000 query) × nlargest(5) ≈ pochi secondi.
top5_by_setting: dict = {}

# Retrieval baseline: ordina per `score` raw (= IP Contriever) decrescente.
# Equivalente a nsmallest("rank") ma più simmetrico col loop alpha sotto.
top5_by_setting["retrieval"] = {
    qid: set(g.nlargest(5, "score")["passage_id_s"])
    for qid, g in base_alpha.groupby("query_id_s", sort=False)
}

# Top-5 per ogni alpha
for alpha in ALPHAS:
    col = f"kg_final_alpha_{int(round(alpha * 10))}"
    top5_by_setting[alpha] = {
        qid: set(g.nlargest(5, col)["passage_id_s"])
        for qid, g in base_alpha.groupby("query_id_s", sort=False)
    }

print(f"top-5 sets cached for {len(top5_by_setting)} settings "
      f"(retrieval + {len(ALPHAS)} alphas)")


# %%
def jaccard_aggregate(setting_a: dict, setting_b: dict) -> tuple[float, float]:
    """Aggrega Jaccard@5 su tutte le query tra due dict di top-5.

    setting_a, setting_b : dict {query_id_s → set(passage_id_s)}
    Returns: (mean_jaccard, pct_jaccard_lt_1)
    """
    common_qids = set(setting_a.keys()) & set(setting_b.keys())
    arr = np.array([jaccard(setting_a[q], setting_b[q]) for q in common_qids])
    return float(arr.mean()), float((arr < 1.0).mean())


# %% [markdown]
# ### 8.1 · α vs retrieval baseline
#
# Per ogni α, quanto perturba il top-5 rispetto al puro dense ranking?
# - mean_jacc@5 vicino a 1 → poca differenza (rerank ≈ retrieval)
# - mean_jacc@5 vicino a 0 → top-5 stravolto

# %%
results_vs_retrieval = []
for alpha in ALPHAS:
    mean_j, pct_lt1 = jaccard_aggregate(
        top5_by_setting[alpha], top5_by_setting["retrieval"]
    )
    results_vs_retrieval.append({
        "alpha": alpha,
        "mean_jacc@5": mean_j,
        "%j@5<1": pct_lt1,
    })
df_alpha_vs_retrieval = pd.DataFrame(results_vs_retrieval)
df_alpha_vs_retrieval

# %% [markdown]
# ### 8.2 · Pairwise α vs α matrix (upper triangular)
#
# Per ogni coppia (α_i, α_j) con i < j: `mean_jacc@5` tra i loro top-5.
# Valori vicini a 1 ⇒ rerank quasi identici ⇒ uno dei due ridondante per
# LLM-eval. Valori bassi ⇒ rerank distinti ⇒ entrambi candidati.

# %%
# Costruisci la matrice 9×9. Riempiamo solo upper triangular (j > i):
# la matrice è simmetrica per definizione, la diagonale è 1.0 (ogni
# setting confrontato con sé stesso). Lower triangular lasciato a NaN
# per leggibilità.
matrix_jacc = pd.DataFrame(
    np.full((len(ALPHAS), len(ALPHAS)), np.nan),
    index=ALPHAS,
    columns=ALPHAS,
)
for i, a1 in enumerate(ALPHAS):
    matrix_jacc.iloc[i, i] = 1.0
    for j, a2 in enumerate(ALPHAS):
        if j > i:
            mean_j, _ = jaccard_aggregate(top5_by_setting[a1], top5_by_setting[a2])
            matrix_jacc.iloc[i, j] = mean_j

print("mean_jaccard@5 between α_i (rows) and α_j (cols), upper triangular:")
print(matrix_jacc.round(3).to_string(na_rep="—"))

# %% [markdown]
# **Lettura matrice**:
# - Step "smooth": tipicamente α adiacenti (es. 0.4 vs 0.5) hanno jaccard alto
#   (~0.85+); α distanti (es. 0.1 vs 0.9) hanno jaccard basso.
# - **Strategia di pruning**: scegli un sottoinsieme di α dove ogni coppia
#   ha jaccard < soglia (es. 0.7), così copri scenari distinti senza buttare
#   budget LLM su run quasi-equivalenti.
# - Esempio possibile: se 0.3 vs 0.5 ha jaccard=0.85 (simili) e 0.5 vs 0.7
#   ha jaccard=0.6 (diversi), butti 0.3, tieni {0.5, 0.7}. Più altri α
#   distanti per coprire estremi.