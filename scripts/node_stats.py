"""Layer 1.5 — Compute degree statistics per QID from edges.parquet.

OBJECTIVE
=========
For every Q-entity that appears in any triple of edges.parquet, compute:
    in_degree    = number of triples where this QID is the OBJECT
    out_degree   = number of triples where this QID is the SUBJECT
    total_degree = in_degree + out_degree

Output is consumed by Layer 3 (BFS hub-banning): if a node's total_degree
exceeds a threshold (default 5000), the BFS treats it as terminal —
includes it in the visited set but does not enumerate its neighbors.

WHY THIS LAYER EXISTS
=====================
Without precomputed degrees, Layer 3 would have to call HDT
search_triples twice per visited node just to decide "is this a hub?".
For ~50k visited nodes per seed × ~1500 seeds, that's 150 million extra
HDT lookups — wasted hours. Storing degrees once means O(1) hash lookup
in Python at BFS time.

INPUT
=====
data/db/edges.parquet
    Columns: subject (str), predicate (str), object (str)
    All values stripped of Wikidata URI prefix:
        "Q42" instead of "http://www.wikidata.org/entity/Q42"
        "P31" instead of "http://www.wikidata.org/prop/direct/P31"
    Approx 1.5-2 billion rows after the Q-Q filter applied by hdt_export.py.

OUTPUT
======
data/db/node_stats.parquet
    Columns:
        qid           (str)    — the Q-entity ID, e.g. "Q42"
        in_degree     (uint64) — how many edges point TO this entity
        out_degree    (uint64) — how many edges originate FROM this entity
        total_degree  (uint64) — in + out
    Rows sorted by total_degree DESC (hubs first — easy to inspect with
    `polars.read_parquet(...).head(20)`).

API USED
========
polars (pl):
    pl.scan_parquet(path)
        Returns a LazyFrame. No data is read; only the schema is inspected.
    LazyFrame.group_by(col).agg(...)
        Streaming groupby. Polars uses a hash aggregation that spills to
        disk if the unique-key set doesn't fit in RAM.
    LazyFrame.join(other, on, how="full", coalesce=True)
        FULL OUTER JOIN. coalesce=True merges the join key columns so we
        get a single 'qid' column instead of 'qid' + 'qid_right'.
    pl.col(c).fill_null(value)
        Replaces nulls produced by the outer join (when a QID appears
        only on one side) with 0.
    LazyFrame.collect(streaming=True)
        Triggers execution with the streaming engine — keeps memory bounded.
    DataFrame.write_parquet(path, compression="snappy")
        Writes a single parquet file. Snappy is fast and good enough.

WHY POLARS NOT DUCKDB
=====================
Polars is already in pyproject.toml; using it avoids one extra dependency.
On 1.5-2B rows of (str, str, str), polars' streaming aggregation handles
the workload comfortably (peak ~1-3 GB RAM for the hash tables).

EXPECTED RUNTIME
================
~5-15 minutes on the full edges.parquet, depending on disk speed (the
file lives on /mnt/c, so 9P bridge overhead applies).

HOW TO RUN
==========
From Windows venv (polars already installed):
    .venv\\Scripts\\python.exe scripts\\node_stats.py
"""

import time
from pathlib import Path
import polars as pl


# ============================================================================
# Path resolution
# ============================================================================
# We use the same walk-up-to-pyproject.toml trick as all other scripts in
# this repo. This makes the script runnable from any working directory and
# from any clone location.

def _find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        # Fallback for notebook context where __file__ is undefined
        start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"Could not find repo root (pyproject.toml) above {start}")


REPO_ROOT = _find_repo_root()
EDGES_PATH = REPO_ROOT / "data" / "db" / "edges.parquet"
OUT_PATH = REPO_ROOT / "data" / "db" / "node_stats.parquet"


# ============================================================================
# Main pipeline
# ============================================================================
# Three logical phases, all expressed declaratively as a single polars
# query that the engine executes as one streaming pass:
#
#   1. Out-degree branch  : GROUP BY subject, COUNT
#   2. In-degree branch   : GROUP BY object, COUNT
#   3. Outer join + total : combine, fill nulls, sum

def main() -> None:
    # Pre-condition: edges.parquet must already exist.
    # We refuse to silently auto-run hdt_export.py — that's an 8-hour job
    # the user must initiate explicitly (and from WSL, not Windows).
    assert EDGES_PATH.exists(), (
        f"\nedges.parquet not found at: {EDGES_PATH}\n"
        "Run scripts/hdt_export.py from WSL first (Layer 1).\n"
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    edges_size_gb = EDGES_PATH.stat().st_size / (1024 ** 3)
    print(f"Input:  {EDGES_PATH}  ({edges_size_gb:.2f} GB)")
    print(f"Output: {OUT_PATH}")
    print()

    t_start = time.perf_counter()

    # --- Lazy scan: only metadata is read here, no actual rows ---
    # Polars looks at the parquet footer to discover schema and row groups.
    edges = pl.scan_parquet(EDGES_PATH)

    # --- Out-degree branch ---
    # For each distinct subject, count how many rows it's the subject of.
    # We rename "subject" to "qid" so this branch's schema matches the
    # in-degree branch — needed for the FULL OUTER JOIN below.
    out_degrees = (
        edges
        .group_by("subject")
        .agg(pl.len().alias("out_degree"))  # pl.len() = count of rows in group
        .rename({"subject": "qid"})
    )

    # --- In-degree branch ---
    # For each distinct object, count how many rows it's the object of.
    in_degrees = (
        edges
        .group_by("object")
        .agg(pl.len().alias("in_degree"))
        .rename({"object": "qid"})
    )

    # --- FULL OUTER JOIN ---
    # A QID may appear ONLY as subject (no incoming edges) or ONLY as object
    # (no outgoing edges) or in both. The full outer join captures all three
    # cases. fill_null(0) handles the missing-direction case.
    #
    # how="full" requests outer join. coalesce=True merges the two join-key
    # columns into a single 'qid' column (otherwise polars would name them
    # 'qid' and 'qid_right' separately).
    stats = (
        out_degrees
        .join(in_degrees, on="qid", how="full", coalesce=True)
        .with_columns([
            pl.col("in_degree").fill_null(0),
            pl.col("out_degree").fill_null(0),
        ])
        .with_columns(
            (pl.col("in_degree") + pl.col("out_degree")).alias("total_degree")
        )
        # Sort hubs to the top: makes spot-checking and Top-N inspection trivial.
        .sort("total_degree", descending=True)
    )

    # --- Trigger execution ---
    # collect(streaming=True) runs the entire plan with polars' streaming
    # engine. On 1.5-2B rows, this is what keeps memory bounded — we never
    # materialize the full edges table or the intermediate hash tables.
    print("Running streaming aggregation...")
    result = stats.collect(streaming=True)
    elapsed = time.perf_counter() - t_start
    print(f"  done in {elapsed:.1f}s")
    print(f"  total distinct QIDs: {len(result):,}")

    # --- Sanity printout: top hubs ---
    # Useful for verifying the result before writing. We expect Q5 (human)
    # near the top with millions of in-degree.
    print("\nTop-10 hubs by total_degree:")
    print(f"  {'qid':>10}  {'in_degree':>14}  {'out_degree':>12}  {'total':>14}")
    for row in result.head(10).iter_rows(named=True):
        print(f"  {row['qid']:>10}  {row['in_degree']:>14,}  "
              f"{row['out_degree']:>12,}  {row['total_degree']:>14,}")

    # --- Write parquet ---
    result.write_parquet(OUT_PATH, compression="snappy")
    out_mb = OUT_PATH.stat().st_size / (1024 ** 2)
    print(f"\nWrote {OUT_PATH}  ({out_mb:.1f} MB)")


if __name__ == "__main__":
    main()