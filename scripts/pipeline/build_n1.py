"""Layer 3 N1 — Precompute 1-hop neighborhoods for query and passage entities.

OBJECTIVE
=========
For each QID in {seeds ∪ passage_entities}, store the set of immediate
neighbors (1-hop) plus their degree, in a flat parquet table. This is
the cornerstone of architecture B (vedi §4.10): instead of full
BFS-3-waves precompute, store only N1 and answer reachability queries
on-demand at scoring time via meet-in-the-middle.

INPUTS
======
- data/db/edges.parquet (Layer 1): 661.5M Q-Q wdt:* triples
- data/db/node_stats.parquet (Layer 1.5): qid → total_degree
- data/NQ_answer/queries_curated.jsonl: question_qids + answer_variant_qids
- data/NQ_answer/passage_entities_curated.parquet: passage entities

OUTPUT
======
data/n1/n1.parquet
    Schema (long format, one row per (qid, neighbor) directed-pair):
        qid              (str)        # entity in {seeds ∪ passage_entities}
        neighbor         (str)        # neighbor of qid (in or out direction)
        neighbor_degree  (uint64)     # total_degree of neighbor (from node_stats)

ALGORITHM
=========
1. Collect target QIDs: union of seeds + passage entities (~140k)
2. Build N1 via DuckDB:
     UNION (dedupe) of:
       - SELECT subject AS qid, object AS neighbor FROM edges WHERE subject IN targets
       - SELECT object  AS qid, subject AS neighbor FROM edges WHERE object  IN targets
     LEFT JOIN node_stats on neighbor for degree
3. Write to parquet (snappy)

WHY DUCKDB AND NOT HDT
======================
Layer 1 already extracted edges into a clean parquet. DuckDB JOINs these
natively, much faster than HDT search_triples per QID. We use
edges.parquet as the single source of truth for the graph from Layer 3
onwards.

WHY LONG FORMAT
===============
- Easy to filter by threshold at scoring time: WHERE neighbor_degree <= t
- No nested types — loadable everywhere (pandas/polars/duckdb)
- Group_by qid for set operations at scoring time

WHY UNION (NOT UNION ALL)
=========================
The same (qid, neighbor) pair can appear multiple times if the two are
linked by more than one predicate (e.g., qid is both subject AND object
of relations to neighbor). For boolean reachability we just need
set-membership, so DEDUPE in the source data.

EXPECTED RUNTIME
================
~30-60 min on a single DuckDB process. The heavy lifting is the JOIN
on 661M-row edges × 140k targets, with column-pruning + predicate
pushdown via Parquet metadata.

HOW TO RUN
==========
From Windows venv (DuckDB and PyArrow already installed):
    .venv\\Scripts\\python.exe scripts\\build_n1.py
"""

import json
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def _find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not find repo root above {start}")


REPO_ROOT = _find_repo_root()
EDGES_PATH = REPO_ROOT / "data" / "db" / "edges.parquet"
NODE_STATS_PATH = REPO_ROOT / "data" / "db" / "node_stats.parquet"
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
PASSAGE_ENTITIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

OUT_DIR = REPO_ROOT / "data" / "n1"
OUT_PATH = OUT_DIR / "n1.parquet"


def collect_target_qids() -> set[str]:
    """Union of seed QIDs (queries) and passage entity QIDs.

    Both sources contribute because at scoring time we need N1 for both
    sides of each (q, d) pair where q ∈ seeds and d ∈ passage_entities.

    DATA SHAPES
    -----------
    - queries_curated.jsonl: one JSON per line with `question_qids` (list[str])
      and `answer_variant_qids` (list[list[str]]). We collect from both —
      the seed pool for KG retrieval (cf. §6.1 item 9: ~1500 distinct).
    - passage_entities_curated.parquet: column `qids` (list[str]), one row
      per passage. We flatten and collect distinct (~138k).
    """
    qids: set[str] = set()

    # --- Seeds from queries jsonl ---
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qids.update(obj.get("question_qids") or [])
            for variant in obj.get("answer_variant_qids") or []:
                qids.update(variant or [])

    # --- Passage entities from parquet (only the qids column) ---
    table = pq.read_table(PASSAGE_ENTITIES_PATH, columns=["qids"])
    for chunk in table.column("qids").chunks:
        flat = chunk.flatten()
        qids.update(flat.to_pylist())

    return qids


def main() -> None:
    assert EDGES_PATH.exists(), f"edges.parquet not found at {EDGES_PATH}"
    assert NODE_STATS_PATH.exists(), f"node_stats.parquet not found at {NODE_STATS_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Repo root:  {REPO_ROOT}")
    print(f"Edges:      {EDGES_PATH}")
    print(f"Node stats: {NODE_STATS_PATH}")
    print(f"Output:     {OUT_PATH}")
    print()

    # --- Phase 1: collect target QIDs ---
    print("Collecting target QIDs (seeds ∪ passage entities)...", flush=True)
    t0 = time.perf_counter()
    targets = collect_target_qids()
    print(f"  {len(targets):,} distinct target QIDs in {time.perf_counter() - t0:.1f}s",
          flush=True)

    # --- Phase 2: register targets in DuckDB ---
    db = duckdb.connect(":memory:")
    targets_table = pa.table({"qid": sorted(targets)})
    db.register("targets", targets_table)
    print()

    # --- Phase 3: build N1 via UNION of two JOINs ---
    # COPY ... TO streams the result directly to parquet without materializing
    # in RAM — important because N1 can be hundreds of millions of rows.
    print("Building N1 via DuckDB JOIN on edges.parquet (~30-60 min)...", flush=True)
    t0 = time.perf_counter()
    db.execute(f"""
        COPY (
            WITH n1_directed AS (
                SELECT t.qid AS qid, e.object AS neighbor
                FROM '{EDGES_PATH.as_posix()}' e
                INNER JOIN targets t ON e.subject = t.qid

                UNION

                SELECT t.qid AS qid, e.subject AS neighbor
                FROM '{EDGES_PATH.as_posix()}' e
                INNER JOIN targets t ON e.object = t.qid
            )
            SELECT
                n.qid,
                n.neighbor,
                COALESCE(ns.total_degree, 0) AS neighbor_degree
            FROM n1_directed n
            LEFT JOIN '{NODE_STATS_PATH.as_posix()}' ns ON n.neighbor = ns.qid
        ) TO '{OUT_PATH.as_posix()}' (FORMAT 'parquet', COMPRESSION 'snappy')
    """)
    elapsed = time.perf_counter() - t0
    print(f"  done in {elapsed/60:.1f} min", flush=True)

    # --- Phase 4: sanity stats ---
    print()
    out_mb = OUT_PATH.stat().st_size / (1024 ** 2)
    print(f"Output: {OUT_PATH}  ({out_mb:.1f} MB)")

    rows, distinct_qids, distinct_neighbors = db.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT qid),
            COUNT(DISTINCT neighbor)
        FROM '{OUT_PATH.as_posix()}'
    """).fetchone()
    print(f"  total rows:           {rows:,}")
    print(f"  distinct qids w/ N1:  {distinct_qids:,}  (targets: {len(targets):,})")
    print(f"  distinct neighbors:   {distinct_neighbors:,}")

    n_targets_missing = len(targets) - distinct_qids
    if n_targets_missing > 0:
        print(f"  ! {n_targets_missing:,} target qids have no edges in edges.parquet")
        print(f"    (expected for some isolated/nonsense QIDs from ReFiNed)")

    # --- Phase 5: distribution of |N1| sizes ---
    # This tells us at-a-glance the scale we'll work with at scoring time.
    print("\n=== Distribution of |N1| (per qid) ===")
    stats = db.execute(f"""
        WITH per_qid AS (
            SELECT qid, COUNT(*) AS n1_size
            FROM '{OUT_PATH.as_posix()}'
            GROUP BY qid
        )
        SELECT
            MIN(n1_size),
            quantile_cont(n1_size, 0.50),
            quantile_cont(n1_size, 0.90),
            quantile_cont(n1_size, 0.99),
            MAX(n1_size),
            AVG(n1_size)
        FROM per_qid
    """).fetchone()
    metric_labels = ["min", "p50", "p90", "p99", "max", "avg"]
    for lbl, val in zip(metric_labels, stats):
        print(f"  {lbl:>5}: {int(val):>13,}")

    # Top-10 largest N1 (likely hub-seeds: countries, languages, etc.)
    print("\n=== Top 10 qids by |N1| (likely hub seeds/entities) ===")
    top = db.execute(f"""
        SELECT qid, COUNT(*) AS n1_size
        FROM '{OUT_PATH.as_posix()}'
        GROUP BY qid
        ORDER BY n1_size DESC
        LIMIT 10
    """).fetchall()
    print(f"  {'qid':>10}  {'|N1|':>13}")
    print(f"  {'-'*10}  {'-'*13}")
    for qid, sz in top:
        print(f"  {qid:>10}  {sz:>13,}")


if __name__ == "__main__":
    main()
