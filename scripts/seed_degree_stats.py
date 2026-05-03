"""Diagnostic — distribution of total_degree over the BFS seed pool.

OBJECTIVE
=========
Layer 3 BFS (build_n3.py) uses the dataset's question_qids and
answer_variant_qids as seeds. The expansion always proceeds from a seed
regardless of its degree, so a hub-seed (e.g., Q21=England with 50k+
incoming edges) makes BFS infeasible.

Before changing the BFS algorithm or accepting metric degradation, we
need to know HOW MANY seeds are actually hubs.

This script:
  1. Loads the seed list from queries_curated.jsonl (same as build_n3.py)
  2. Joins with node_stats.parquet for total_degree
  3. Optionally joins with labels.parquet for human-readable names
  4. Prints distribution + per-threshold counts + top-N hub-seeds

INPUTS
======
- data/NQ_answer/queries_curated.jsonl
- data/db/node_stats.parquet
- data/db/labels.parquet (optional, for top-N hub names)

USAGE
=====
From Windows venv (duckdb already installed):
    .venv\\Scripts\\python.exe scripts\\seed_degree_stats.py
"""

import json
from pathlib import Path

import duckdb


def _find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not find repo root above {start}")


REPO_ROOT = _find_repo_root()
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
NODE_STATS_PATH = REPO_ROOT / "data" / "db" / "node_stats.parquet"
LABELS_PATH = REPO_ROOT / "data" / "db" / "labels.parquet"


def load_seeds() -> list[str]:
    """Same logic as build_n3.py — union of question_qids + answer_variant_qids."""
    qids: set[str] = set()
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qids.update(obj.get("question_qids") or [])
            for variant in obj.get("answer_variant_qids") or []:
                qids.update(variant or [])
    return sorted(qids)


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Queries:   {QUERIES_PATH}")
    print(f"Node stats: {NODE_STATS_PATH}")
    print()

    # --- Phase 1: Load seeds ---
    seeds = load_seeds()
    print(f"Distinct seeds from queries_curated.jsonl: {len(seeds):,}")

    # --- Phase 2: Build a DuckDB session and join with parquet ---
    con = duckdb.connect(":memory:")
    # Register seeds as a virtual table for the JOIN
    import pyarrow as pa
    seeds_table = pa.table({"qid": seeds})
    con.register("seeds", seeds_table)

    # Coverage check: how many seeds have an entry in node_stats?
    coverage = con.execute(f"""
        SELECT
            COUNT(*) AS n_with_data,
            (SELECT COUNT(*) FROM seeds) - COUNT(*) AS n_missing
        FROM seeds s
        INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
    """).fetchone()
    n_with_data, n_missing = coverage
    print(f"  with data in node_stats:    {n_with_data:>6,}")
    print(f"  missing (degree = 0):       {n_missing:>6,}  "
          f"({100*n_missing/len(seeds):.1f}%)")

    # --- Phase 3: Distribution stats ---
    print("\n=== Distribution of total_degree over seeds (with data) ===")
    stats = con.execute(f"""
        WITH seed_deg AS (
            SELECT s.qid, n.total_degree
            FROM seeds s
            INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
        )
        SELECT
            MIN(total_degree)            AS min,
            quantile_cont(total_degree, 0.05) AS p05,
            quantile_cont(total_degree, 0.25) AS p25,
            quantile_cont(total_degree, 0.50) AS p50,
            quantile_cont(total_degree, 0.75) AS p75,
            quantile_cont(total_degree, 0.90) AS p90,
            quantile_cont(total_degree, 0.95) AS p95,
            quantile_cont(total_degree, 0.99) AS p99,
            MAX(total_degree)            AS max,
            AVG(total_degree)            AS avg
        FROM seed_deg
    """).fetchone()
    labels_row = ["min", "p05", "p25", "p50 (median)", "p75", "p90", "p95", "p99",
                  "max", "avg"]
    for label, val in zip(labels_row, stats):
        print(f"  {label:>15}: {int(val):>13,}")

    # --- Phase 4: How many seeds at each threshold ---
    print("\n=== Seeds above hub-banning threshold candidates ===")
    print(f"  {'threshold':>12}  {'n_seeds_over':>15}  {'%_over':>10}  "
          f"{'%_remaining':>15}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*10}  {'-'*15}")
    for threshold in [500, 1000, 2000, 5000, 10000, 50000, 100000, 500000]:
        n_over = con.execute(f"""
            SELECT COUNT(*) FROM seeds s
            INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
            WHERE n.total_degree > {threshold}
        """).fetchone()[0]
        pct_over = 100 * n_over / len(seeds)
        pct_remaining = 100 - pct_over
        print(f"  {threshold:>12,}  {n_over:>15,}  {pct_over:>9.1f}%  "
              f"{pct_remaining:>14.1f}%")

    # --- Phase 5: Top-N hub-seeds with labels (if available) ---
    print("\n=== Top 30 highest-degree seeds ===")
    if LABELS_PATH.exists():
        top = con.execute(f"""
            SELECT s.qid, n.total_degree, l.label_en
            FROM seeds s
            INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
            LEFT JOIN '{LABELS_PATH.as_posix()}' l ON s.qid = l.qid
            ORDER BY n.total_degree DESC
            LIMIT 30
        """).fetchall()
        print(f"  {'qid':>10}  {'total_degree':>13}  label")
        print(f"  {'-'*10}  {'-'*13}  {'-'*40}")
        for qid, deg, lbl in top:
            label_str = lbl if lbl else "(no label)"
            print(f"  {qid:>10}  {deg:>13,}  {label_str}")
    else:
        top = con.execute(f"""
            SELECT s.qid, n.total_degree
            FROM seeds s
            INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
            ORDER BY n.total_degree DESC
            LIMIT 30
        """).fetchall()
        print(f"  {'qid':>10}  {'total_degree':>13}")
        print(f"  {'-'*10}  {'-'*13}")
        for qid, deg in top:
            print(f"  {qid:>10}  {deg:>13,}")
        print("\n  (labels.parquet not found — run build_labels.py for names)")

    # --- Phase 6: Bottom-N for sanity (low-degree seeds) ---
    print("\n=== Bottom 10 lowest-degree seeds (sanity check) ===")
    bot = con.execute(f"""
        SELECT s.qid, n.total_degree
        FROM seeds s
        INNER JOIN '{NODE_STATS_PATH.as_posix()}' n ON s.qid = n.qid
        ORDER BY n.total_degree ASC
        LIMIT 10
    """).fetchall()
    print(f"  {'qid':>10}  {'total_degree':>13}")
    print(f"  {'-'*10}  {'-'*13}")
    for qid, deg in bot:
        print(f"  {qid:>10}  {deg:>13,}")

    # --- Phase 7: Query-level taintedness (using ONLY question_qids) ---
    # A query is "tainted" if at least one of its question entities is a hub.
    # We use question_qids only — answer_variant_qids represent ground-truth
    # answers we don't have at retrieval time. This phase is independent from
    # Phases 2-6, which used the full seed union (question + answer variants).
    print("\n=== Query-level taintedness (question_qids only) ===")

    query_pairs: list[tuple[int, str]] = []  # flat (query_idx, qid) pairs
    n_with = 0
    n_without = 0
    qids_per_query: list[int] = []
    distinct_qqids: set[str] = set()

    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for query_idx, line in enumerate(f):
            obj = json.loads(line)
            q_qids = obj.get("question_qids") or []
            if q_qids:
                n_with += 1
                qids_per_query.append(len(q_qids))
                for qid in q_qids:
                    query_pairs.append((query_idx, qid))
                    distinct_qqids.add(qid)
            else:
                n_without += 1

    n_total = n_with + n_without
    print(f"  total queries:           {n_total:>5,}")
    print(f"  with question_qids:      {n_with:>5,}")
    print(f"  without question_qids:   {n_without:>5,}  (excluded — no seed possible)")
    print(f"  distinct question_qids:  {len(distinct_qqids):>5,}  "
          f"(vs {len(seeds):,} union with answer-variants)")
    if qids_per_query:
        avg_q = sum(qids_per_query) / len(qids_per_query)
        print(f"  qids per query:  min={min(qids_per_query)}  "
              f"avg={avg_q:.1f}  max={max(qids_per_query)}")

    if query_pairs:
        # Register the (query_id, qid) pairs as a virtual table for the JOIN.
        qs_table = pa.table({
            "query_id": [p[0] for p in query_pairs],
            "qid": [p[1] for p in query_pairs],
        })
        con.register("query_seeds", qs_table)

        # For each threshold, classify queries:
        #   clean   = no hub seed                      → KG-score full
        #   mixed   = some hubs AND some non-hubs      → KG-score partial
        #   all-hub = every seed is a hub              → KG-score=0 inevitable
        # %lost-if-drop = % queries we'd lose by keeping only `clean`.
        print(f"\n  {'thresh':>8}  {'clean':>7}  {'mixed':>7}  {'all-hub':>8}  "
              f"{'%clean':>7}  {'%mixed':>7}  {'%all-hub':>9}  "
              f"{'%lost-if-drop':>14}")
        print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*7}  "
              f"{'-'*9}  {'-'*14}")

        for threshold in [500, 1000, 2000, 5000, 10000]:
            row = con.execute(f"""
                WITH classified AS (
                    SELECT
                        qs.query_id,
                        SUM(CASE WHEN COALESCE(n.total_degree, 0) > {threshold}
                                 THEN 1 ELSE 0 END) AS n_hub,
                        COUNT(*) AS n_total
                    FROM query_seeds qs
                    LEFT JOIN '{NODE_STATS_PATH.as_posix()}' n ON qs.qid = n.qid
                    GROUP BY qs.query_id
                )
                SELECT
                    SUM(CASE WHEN n_hub = 0 THEN 1 ELSE 0 END) AS clean,
                    SUM(CASE WHEN n_hub > 0 AND n_hub < n_total
                             THEN 1 ELSE 0 END) AS mixed,
                    SUM(CASE WHEN n_hub > 0 AND n_hub = n_total
                             THEN 1 ELSE 0 END) AS all_hub
                FROM classified
            """).fetchone()
            clean, mixed, all_hub = row
            total = clean + mixed + all_hub
            pct_c = 100 * clean / total
            pct_m = 100 * mixed / total
            pct_a = 100 * all_hub / total
            pct_lost = 100 * (mixed + all_hub) / total
            print(f"  {threshold:>8,}  {clean:>7,}  {mixed:>7,}  {all_hub:>8,}  "
                  f"{pct_c:>6.1f}%  {pct_m:>6.1f}%  {pct_a:>8.1f}%  "
                  f"{pct_lost:>13.1f}%")


if __name__ == "__main__":
    main()
