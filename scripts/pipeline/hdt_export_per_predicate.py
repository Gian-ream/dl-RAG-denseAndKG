"""Layer 1 (per-predicate version) — Export Wikidata HDT Q-Q wdt:* triples.

CONTEXT
=======
This is the corrected replacement for hdt_export.py. The previous wildcard
approach `doc.search_triples("", "", "")` had two problems on this dump:
  1. OVERSHOOT: yielded 8.85B+ triples vs the dump's 3.65B header count,
     producing labels/descriptions/reifications well past the wdt:* core.
  2. SILENT GAPS: missed Q-Q wdt:* matches anyway. verify_completeness.py
     confirmed e.g. 15,781 missing on P2860 alone (out of 285M).

THIS APPROACH
=============
Iterate PER PREDICATE: for each Pxxx in [P1, P15000], call
    iter, count = doc.search_triples("", "wdt:Pxxx", "")
This API is bounded and deterministic. Sanity verified earlier:
    iter_total == hdt_count   (3/3 predicates: P7481, P3174, P2860)

For each predicate's iterator, filter Q-Q (both subject and object pure
Q-entities) and write to parquet, same schema as before.

WHY NOT SLOWER
==============
P2860 (285M triples) iterated in ~690s = 413k triples/sec, comparable
to the wildcard. But per-predicate has zero overshoot — we only iterate
what's relevant. Total estimate for all wdt:* predicates: ~1-1.5h.

P-ID RANGE
==========
We brute-force P1...P15000. Wikidata P-IDs are dense in the low range
(P1-P9000 are mostly populated) and sparse above. P15000 is a generous
upper bound as of 2026; predicates beyond that are extremely rare.
For each missing P-ID, search_triples returns count=0 (instant, ~1ms).

OUTPUT
======
data/db/edges_v2.parquet
    Schema:
        subject   (str)  — "Q42" (URI prefix stripped)
        predicate (str)  — "P31" (URI prefix stripped)
        object    (str)  — "Q5"  (URI prefix stripped)
    Row order: predicate-major (P1's triples first, then P2's, ...).
    Within a predicate: the natural HDT iteration order (subject-major).

The OLD `edges.parquet` is NOT touched — kept as backup until promotion.

USAGE
=====
From WSL with the dl-rag-wsl venv:
    python scripts/pipeline/hdt_export_per_predicate.py 2>&1 | tee /tmp/hdt_export_v2.log
"""

import time
from pathlib import Path
from hdt import HDTDocument
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Path resolution (same idiom as other scripts)
# ============================================================================

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
HDT_PATH = REPO_ROOT / "data" / "Wikidata_service" / "latest-all-06-Jan-2022.hdt"
OUT_PATH = REPO_ROOT / "data" / "db" / "edges_v2.parquet"


# ============================================================================
# URI prefixes — same as hdt_export.py
# ============================================================================

WD_ENTITY = "http://www.wikidata.org/entity/"
WDT_DIRECT = "http://www.wikidata.org/prop/direct/"
WD_ENTITY_LEN = len(WD_ENTITY)


def is_q_entity_uri(uri: str) -> bool:
    """True iff URI is a pure Q\\d+ Wikidata entity (no statement nodes)."""
    if not uri.startswith(WD_ENTITY):
        return False
    last = uri[WD_ENTITY_LEN:]
    if not last.startswith("Q"):
        return False
    return "-" not in last and last[1:].isdigit()


# ============================================================================
# Tuning knobs
# ============================================================================

# Same as hdt_export.py: 1M rows ≈ 25 MB raw, ~10-15 MB snappy → good row group size.
BATCH_SIZE = 1_000_000

# P-ID enumeration upper bound. Very generous — Wikidata properties top
# out around P12000 in 2026. Each empty P-ID costs ~1ms (header lookup
# returns count=0 instantly), so the slack is cheap.
MAX_PID = 15_000

# Log a per-predicate progress line if HDT has at least this many triples.
# Smaller predicates (which are most of them by count) are still iterated
# and counted in the totals, just not individually logged — keeps the
# console readable.
LOG_PRED_THRESHOLD = 100_000


# ============================================================================
# Main pipeline
# ============================================================================

def main() -> None:
    assert HDT_PATH.exists(), f"HDT file not found: {HDT_PATH}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"HDT path: {HDT_PATH}")
    print(f"Output:   {OUT_PATH}")
    print()

    print("Loading HDT (uses pre-built index; ~30s)...")
    t0 = time.perf_counter()
    doc = HDTDocument(str(HDT_PATH))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  total triples in dump (header): {doc.total_triples:,}")
    print()

    # Output schema — matches edges.parquet exactly so downstream scripts
    # (node_stats.py, build_n3.py) work unchanged.
    schema = pa.schema([
        ("subject", pa.string()),
        ("predicate", pa.string()),
        ("object", pa.string()),
    ])

    # Streaming writer: row groups are flushed to disk on each write_table.
    # Keep buffer in memory until BATCH_SIZE rows accumulated.
    writer = pq.ParquetWriter(OUT_PATH, schema, compression="snappy")

    buf_s: list[str] = []
    buf_p: list[str] = []
    buf_o: list[str] = []

    # Counters used for progress + final report
    written = 0          # Q-Q rows flushed to parquet so far
    n_iter_total = 0     # total triples iterated (across all predicates)
    n_preds_with_data = 0
    n_preds_logged = 0

    print(f"Iterating P1...P{MAX_PID:,} ...\n")
    print(f"  {'pred':>8}  {'hdt_count':>13}  {'q_q_in_pred':>13}  "
          f"{'pred_time':>9}  {'pred_rate':>10}  "
          f"{'cum_written':>14}  {'elapsed':>8}")
    print(f"  {'-'*8}  {'-'*13}  {'-'*13}  {'-'*9}  {'-'*10}  {'-'*14}  {'-'*8}")

    t_start = time.perf_counter()

    try:
        for pid_n in range(1, MAX_PID + 1):
            pid = f"P{pid_n}"
            uri = WDT_DIRECT + pid

            # Header lookup — instant. count=0 means predicate not in dump.
            iter_, hdt_count = doc.search_triples("", uri, "")
            if hdt_count == 0:
                continue
            n_preds_with_data += 1

            # Iterate this predicate's triples. Bounded by hdt_count.
            t_pred = time.perf_counter()
            n_iter_pred = 0
            n_qq_pred = 0
            for s, _p, o in iter_:
                n_iter_pred += 1
                n_iter_total += 1
                # Filter: both subject and object pure Q-entities.
                # Predicate is fixed by the search, no need to re-check.
                if is_q_entity_uri(s) and is_q_entity_uri(o):
                    buf_s.append(s[WD_ENTITY_LEN:])
                    buf_p.append(pid)  # already without prefix
                    buf_o.append(o[WD_ENTITY_LEN:])
                    n_qq_pred += 1

                    if len(buf_s) >= BATCH_SIZE:
                        # Flush full batch — frees memory and persists progress.
                        table = pa.Table.from_arrays(
                            [pa.array(buf_s), pa.array(buf_p), pa.array(buf_o)],
                            schema=schema,
                        )
                        writer.write_table(table)
                        written += len(buf_s)
                        buf_s.clear()
                        buf_p.clear()
                        buf_o.clear()

            elapsed_pred = time.perf_counter() - t_pred
            elapsed_total = time.perf_counter() - t_start

            # Log only "interesting" predicates to keep the console readable.
            # All predicates contribute to totals regardless.
            if hdt_count >= LOG_PRED_THRESHOLD:
                rate = n_iter_pred / elapsed_pred if elapsed_pred > 0 else 0
                # cum_written = persisted + still in buffer
                cum = written + len(buf_s)
                print(f"  {pid:>8}  {hdt_count:>13,}  {n_qq_pred:>13,}  "
                      f"{elapsed_pred:>7.1f}s  "
                      f"{rate:>9,.0f}/s  "
                      f"{cum:>14,}  "
                      f"{elapsed_total/60:>6.1f}m")
                n_preds_logged += 1

        # Flush any remaining tail rows (last batch < BATCH_SIZE).
        if buf_s:
            table = pa.Table.from_arrays(
                [pa.array(buf_s), pa.array(buf_p), pa.array(buf_o)],
                schema=schema,
            )
            writer.write_table(table)
            written += len(buf_s)
            buf_s.clear()
            buf_p.clear()
            buf_o.clear()

    finally:
        # Always close the writer so the parquet footer is written and
        # the file is readable even if we got interrupted mid-loop.
        writer.close()

    elapsed = time.perf_counter() - t_start
    out_gb = OUT_PATH.stat().st_size / (1024 ** 3)

    print()
    print(f"Done in {elapsed/3600:.2f}h ({elapsed/60:.1f}m)")
    print(f"  predicates with data: {n_preds_with_data:,}  "
          f"(of which {n_preds_logged:,} logged above)")
    print(f"  total HDT iterations: {n_iter_total:,}")
    print(f"  Q-Q wdt:* written:    {written:,}")
    print(f"  output: {OUT_PATH}  ({out_gb:.2f} GB)")
    print()
    print("Compare with old edges.parquet (661,000,000 rows expected).")
    print("If new count is comparable (≥ old, allowing for Phase 1 deltas),")
    print("promote: `mv data/db/edges_v2.parquet data/db/edges.parquet`")


if __name__ == "__main__":
    main()