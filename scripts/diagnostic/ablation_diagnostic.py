"""Layer 3 ablation — Multi-threshold analysis on N1.

OBJECTIVE
=========
For each threshold t in [500, 1000, 2000, 5000, 10000, ∞]:
  - Classify each query as clean / mixed / all-hub based on question_qids
    (ONLY question_qids — answer_variant_qids are ground-truth labels not
     available at retrieval time, see §4.9).
  - Compute mean |N1_filtered(q)| for both seeds and passage entities at
    that threshold, so we can see how aggressive filtering shrinks the
    neighborhoods.
  - Identify which queries are "invalidated" — all question_qids have
    degree > t, so the KG-rerank degrades to dense-only fallback for
    those queries.

This complements `seed_degree_stats.py` (which only looked at degree
distribution) by also reporting the ACTUAL mean N1 sizes after
filtering — what really matters for the final scoring metric.

INPUTS
======
- data/n1/n1.parquet (Layer 3, from build_n1.py)
- data/db/node_stats.parquet (degrees for question_qids)
- data/db/labels.parquet (optional, for human-readable labels in output)
- data/NQ_answer/queries_curated.jsonl
- data/NQ_answer/passage_entities_curated.parquet (passage entity universe)

OUTPUTS
=======
- data/n1/ablation_summary.parquet
    One row per threshold:
        threshold              (uint32)   # 0 encodes ∞
        threshold_label        (str)      # "500", "1k", ..., "inf"
        n_queries_total        (uint32)
        n_queries_clean        (uint32)
        n_queries_mixed        (uint32)
        n_queries_all_hub      (uint32)
        pct_invalidated        (float32)  # 100 * (mixed + all_hub) / total
        mean_n1_size_seeds     (float64)
        mean_n1_size_passages  (float64)

- data/n1/ablation_invalidated_per_t.jsonl
    One JSON per line, one entry per threshold:
        {
            "threshold": 5000,
            "threshold_label": "5k",
            "n_invalidated": 174,
            "queries": [
                {"query_idx": 7, "question_qids": ["Q30"],
                 "degrees": [2353407], "max_degree": 2353407,
                 "labels": ["United States of America"]},
                ...
            ]
        }

HOW TO RUN
==========
After build_n1.py has produced data/n1/n1.parquet:
    .venv\\Scripts\\python.exe scripts\\ablation_diagnostic.py
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
N1_PATH = REPO_ROOT / "data" / "n1" / "n1.parquet"
NODE_STATS_PATH = REPO_ROOT / "data" / "db" / "node_stats.parquet"
LABELS_PATH = REPO_ROOT / "data" / "db" / "labels.parquet"
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
PASSAGE_ENTITIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

OUT_DIR = REPO_ROOT / "data" / "n1"
SUMMARY_OUT = OUT_DIR / "ablation_summary.parquet"
INVALIDATED_OUT = OUT_DIR / "ablation_invalidated_per_t.jsonl"


# Thresholds for ablation. 0 encodes ∞ (no filter) since we use unsigned
# integers in the parquet schema.
THRESHOLDS = [500, 1000, 2000, 5000, 10000, 0]
THRESHOLD_LABELS = {500: "500", 1000: "1k", 2000: "2k",
                    5000: "5k", 10000: "10k", 0: "inf"}


def load_query_seeds() -> list[tuple[int, list[str]]]:
    """Read queries_curated.jsonl and return [(query_idx, question_qids), ...].

    Only `question_qids` are returned — `answer_variant_qids` are NOT used
    in the ablation because they represent ground-truth answers not
    available at retrieval time (see §4.9 for the methodological argument).
    """
    out = []
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            obj = json.loads(line)
            q_qids = obj.get("question_qids") or []
            out.append((idx, q_qids))
    return out


def load_labels(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Load labels.parquet into {qid: label_en} dict, if available.

    Used to enrich the invalidated-queries report with human-readable
    names. If labels.parquet doesn't exist, returns empty dict and the
    output records will have `labels: [None, ...]`.
    """
    if not LABELS_PATH.exists():
        print("  labels.parquet not found — skipping labels in invalidated reports")
        return {}
    rows = con.execute(
        f"SELECT qid, label_en FROM '{LABELS_PATH.as_posix()}' "
        f"WHERE label_en IS NOT NULL"
    ).fetchall()
    return dict(rows)


def main() -> None:
    assert N1_PATH.exists(), (
        f"n1.parquet not found at {N1_PATH}\n"
        "Run scripts/pipeline/build_n1.py first (Layer 3)."
    )
    assert NODE_STATS_PATH.exists(), f"node_stats.parquet not found at {NODE_STATS_PATH}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Repo root:  {REPO_ROOT}")
    print(f"N1:         {N1_PATH}")
    print(f"Node stats: {NODE_STATS_PATH}")
    print(f"Queries:    {QUERIES_PATH}")
    print(f"Output dir: {OUT_DIR}")
    print()

    db = duckdb.connect(":memory:")

    # --- Phase 1: load question_qids per query ---
    queries = load_query_seeds()
    n_total = len(queries)
    n_with_seeds = sum(1 for _, qs in queries if qs)
    n_no_seeds = n_total - n_with_seeds
    print(f"Loaded {n_total:,} queries  "
          f"({n_with_seeds:,} with question_qids, {n_no_seeds:,} without)")

    # --- Phase 2: load degrees for all (query_idx, qid) pairs ---
    # Build a flat (query_idx, qid) Arrow table to JOIN with node_stats.
    pairs = [(idx, qid) for idx, qids in queries for qid in qids]
    if not pairs:
        print("  No question_qids found — nothing to classify")
        return
    pairs_table = pa.table({
        "query_idx": [p[0] for p in pairs],
        "qid": [p[1] for p in pairs],
    })
    db.register("query_pairs", pairs_table)

    deg_per_pair = db.execute(f"""
        SELECT qp.query_idx, qp.qid, COALESCE(ns.total_degree, 0) AS deg
        FROM query_pairs qp
        LEFT JOIN '{NODE_STATS_PATH.as_posix()}' ns ON qp.qid = ns.qid
    """).fetchall()

    # Group by query_idx → [(qid, deg), ...]
    by_query: dict[int, list[tuple[str, int]]] = {}
    for q_idx, qid, deg in deg_per_pair:
        by_query.setdefault(q_idx, []).append((qid, deg))

    # --- Phase 3: register seed and passage qid sets for N1 stats ---
    seed_qids = sorted({q for _, qs in queries for q in qs})
    seeds_table = pa.table({"qid": seed_qids})
    db.register("seed_qids", seeds_table)
    print(f"  distinct question_qids: {len(seed_qids):,}")

    passage_table = pq.read_table(PASSAGE_ENTITIES_PATH, columns=["qids"])
    passage_qids: set[str] = set()
    for chunk in passage_table.column("qids").chunks:
        passage_qids.update(chunk.flatten().to_pylist())
    pass_table = pa.table({"qid": sorted(passage_qids)})
    db.register("passage_qids", pass_table)
    print(f"  distinct passage QIDs:  {len(passage_qids):,}")

    # --- Phase 4: load labels for invalidated reports ---
    labels = load_labels(db)
    if labels:
        print(f"  labels loaded:          {len(labels):,}")

    # --- Phase 5: per-threshold ablation ---
    print("\n=== Multi-threshold ablation ===")
    print(f"  {'thresh':>8}  {'clean':>7}  {'mixed':>7}  {'all-hub':>8}  "
          f"{'%inval':>8}  {'mean_n1_seeds':>14}  {'mean_n1_pass':>13}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*13}")

    summary_rows = []
    invalidated_records = []

    for t in THRESHOLDS:
        t_label = THRESHOLD_LABELS[t]
        # `t == 0` encodes ∞: no filter, no query is "all_hub" by definition
        deg_filter_sql = "TRUE" if t == 0 else f"neighbor_degree <= {t}"

        # --- Classify queries at threshold t ---
        n_clean = n_mixed = n_all_hub = 0
        invalidated_at_t = []

        for q_idx, qid_degs in by_query.items():
            if t == 0:
                # No filter: all queries are "clean" (no hubs to ban)
                n_clean += 1
                continue

            n_hub = sum(1 for _, d in qid_degs if d > t)
            n_local = len(qid_degs)

            if n_hub == 0:
                n_clean += 1
            elif n_hub == n_local:
                n_all_hub += 1
                invalidated_at_t.append({
                    "query_idx": q_idx,
                    "question_qids": [q for q, _ in qid_degs],
                    "degrees": [d for _, d in qid_degs],
                    "max_degree": max(d for _, d in qid_degs),
                    "labels": [labels.get(q) for q, _ in qid_degs],
                })
            else:
                n_mixed += 1

        # --- Mean |N1_filtered| for seeds and passages at threshold t ---
        # Computed once per threshold via DuckDB on n1.parquet, restricted
        # to the relevant qid universe (seeds vs passages) and filtered
        # by neighbor_degree.
        mean_n1_seeds = db.execute(f"""
            WITH per_seed AS (
                SELECT n.qid, COUNT(*) AS n1_size
                FROM '{N1_PATH.as_posix()}' n
                INNER JOIN seed_qids s ON n.qid = s.qid
                WHERE {deg_filter_sql}
                GROUP BY n.qid
            )
            SELECT COALESCE(AVG(n1_size), 0) FROM per_seed
        """).fetchone()[0]

        mean_n1_pass = db.execute(f"""
            WITH per_pass AS (
                SELECT n.qid, COUNT(*) AS n1_size
                FROM '{N1_PATH.as_posix()}' n
                INNER JOIN passage_qids p ON n.qid = p.qid
                WHERE {deg_filter_sql}
                GROUP BY n.qid
            )
            SELECT COALESCE(AVG(n1_size), 0) FROM per_pass
        """).fetchone()[0]

        n_inv = n_mixed + n_all_hub  # mixed counts as "partially invalidated"
        pct_inv = 100 * n_inv / n_total if n_total > 0 else 0.0

        # Print row
        t_disp = t_label
        print(f"  {t_disp:>8}  {n_clean:>7,}  {n_mixed:>7,}  {n_all_hub:>8,}  "
              f"{pct_inv:>7.1f}%  {mean_n1_seeds:>14,.1f}  {mean_n1_pass:>13,.1f}")

        summary_rows.append({
            "threshold": t,
            "threshold_label": t_label,
            "n_queries_total": n_total,
            "n_queries_clean": n_clean,
            "n_queries_mixed": n_mixed,
            "n_queries_all_hub": n_all_hub,
            "pct_invalidated": pct_inv,
            "mean_n1_size_seeds": float(mean_n1_seeds),
            "mean_n1_size_passages": float(mean_n1_pass),
        })

        invalidated_records.append({
            "threshold": t,
            "threshold_label": t_label,
            "n_invalidated": len(invalidated_at_t),
            "queries": invalidated_at_t,
        })

    # --- Phase 6: write outputs ---
    print(f"\nWriting outputs...")

    # Summary parquet — one row per threshold
    summary_table = pa.table({
        "threshold": pa.array([r["threshold"] for r in summary_rows],
                              type=pa.uint32()),
        "threshold_label": [r["threshold_label"] for r in summary_rows],
        "n_queries_total": pa.array([r["n_queries_total"] for r in summary_rows],
                                    type=pa.uint32()),
        "n_queries_clean": pa.array([r["n_queries_clean"] for r in summary_rows],
                                    type=pa.uint32()),
        "n_queries_mixed": pa.array([r["n_queries_mixed"] for r in summary_rows],
                                    type=pa.uint32()),
        "n_queries_all_hub": pa.array([r["n_queries_all_hub"] for r in summary_rows],
                                      type=pa.uint32()),
        "pct_invalidated": pa.array([r["pct_invalidated"] for r in summary_rows],
                                    type=pa.float32()),
        "mean_n1_size_seeds": [r["mean_n1_size_seeds"] for r in summary_rows],
        "mean_n1_size_passages": [r["mean_n1_size_passages"] for r in summary_rows],
    })
    pq.write_table(summary_table, SUMMARY_OUT, compression="snappy")
    print(f"  {SUMMARY_OUT}  ({SUMMARY_OUT.stat().st_size / 1024:.1f} KB)")

    # Invalidated JSONL — one record per threshold (line)
    with INVALIDATED_OUT.open("w", encoding="utf-8") as f:
        for record in invalidated_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {INVALIDATED_OUT}  ({INVALIDATED_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()