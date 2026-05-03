"""Verify that edges.parquet covers all wdt:* Q-Q triples in the HDT dump.

CONTEXT
=======
hdt_export.py used pyHDT's wildcard `search_triples("", "", "")`. On this
Wikidata dump, that iterator overshoots the header-reported `total_triples`
(3.65B) by 2.4x — yielding a long tail of label/description/reification
triples that fail our Q-Q wdt:* filter. We Ctrl+C'd at 8.85B yields with
661M rows written. The crucial question: are those 661M rows COMPLETE for
the wdt:* Q-Q subset, or did the iterator skip something?

WHY THIS WORKS
==============
HDT's PER-PREDICATE search (`search_triples("", "wdt:Pxxx", "")`) is a
different code path from the full-wildcard search. It uses the predicate
index in BitmapTriples, which:
  - returns an EXACT count from the header in O(1)
  - iterates a bounded set of triples (no overshoot)

So if for every predicate Pxxx the count we got from HDT matches the
count of rows in our parquet (subject to Q-Q filter), the parquet is
complete by construction.

THREE PHASES
============
Phase 1 — Sanity test (5 min):
    Pick 3 predicates from our parquet (small, median, largest count).
    For each, fully iterate via HDT and verify:
      - HDT-reported count == actual iteration count
      - Q-Q triples in iteration == count in parquet
    If both checks pass → predicate-restricted iteration is reliable.

Phase 2 — HDT predicate inventory (~30 sec):
    Enumerate P1...P12000, get HDT count for each via search_triples.
    Records every predicate with > 0 triples in the dump.
    Note: pyHDT has no API to enumerate the predicate dictionary
    directly, so we brute-force the P-ID range. Cheap (each lookup is ~1ms).

Phase 3 — Classification (~1 sec):
    For every wdt:* predicate found in HDT (Phase 2), compare with parquet:
      - perfect  : HDT_count == parquet_count
                   (predicate is fully Q-valued — all its triples are Q-Q)
      - partial  : HDT_count >  parquet_count
                   (some objects are literals → ambiguous; could be legit
                    or could mean we missed Q-Q triples — needs Phase 4
                    follow-up if delta is suspicious)
      - missing  : HDT_count >  0 BUT parquet_count == 0
                   (predicate not in our parquet at all — strong signal
                    of missing data IF the predicate is wikibase-item)

OUTPUT
======
Console report with summary + top-N for each category.
JSON dump at data/db/verification.json for follow-up analysis.

USAGE
=====
From WSL with the dl-rag-wsl venv (pyHDT + pyarrow installed):
    python scripts/verify_completeness.py
"""

import json
import time
from pathlib import Path
from collections import Counter

from hdt import HDTDocument
import pyarrow.parquet as pq
import pyarrow.compute as pc


# ============================================================================
# Path resolution
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
PARQUET_PATH = REPO_ROOT / "data" / "db" / "edges.parquet"
OUT_PATH = REPO_ROOT / "data" / "db" / "verification.json"


# ============================================================================
# URI prefixes
# ============================================================================

WD_ENTITY = "http://www.wikidata.org/entity/"
WDT_DIRECT = "http://www.wikidata.org/prop/direct/"
WD_ENTITY_LEN = len(WD_ENTITY)


def is_q_entity_uri(uri: str) -> bool:
    """Same filter as hdt_export.py — pure Q\\d+ entity, no statement nodes."""
    if not uri.startswith(WD_ENTITY):
        return False
    last = uri[WD_ENTITY_LEN:]
    if not last.startswith("Q"):
        return False
    return "-" not in last and last[1:].isdigit()


# ============================================================================
# Parquet → predicate counts
# ============================================================================

def parquet_predicate_counts() -> Counter:
    """Read the predicate column and return Counter(predicate -> count).

    Uses pyarrow.compute.value_counts which streams a single column
    efficiently. For 661M rows this is ~10-20s, well within memory.
    """
    print(f"Reading parquet predicate column from {PARQUET_PATH}...")
    t0 = time.perf_counter()
    table = pq.read_table(PARQUET_PATH, columns=["predicate"])
    # value_counts returns a StructArray with fields (values, counts).
    vc = pc.value_counts(table.column("predicate"))
    counts: Counter[str] = Counter()
    for entry in vc.to_pylist():
        counts[entry["values"]] = entry["counts"]
    elapsed = time.perf_counter() - t0
    print(f"  {len(counts):,} distinct predicates, "
          f"{sum(counts.values()):,} total rows  ({elapsed:.1f}s)")
    return counts


# ============================================================================
# Phase 1 — Sanity test
# ============================================================================

def phase1_sanity(doc: HDTDocument, parquet_counts: Counter) -> bool:
    """For 3 predicates of varying frequency, fully iterate via HDT and
    verify (a) HDT count matches actual iteration count, (b) the Q-Q
    subset in iteration matches our parquet count.

    Returns True if all 3 sanity checks pass.
    """
    print("\n=== PHASE 1: Sanity ===")
    if not parquet_counts:
        print("  no predicates in parquet — skipping")
        return False

    sorted_preds = sorted(parquet_counts.items(), key=lambda x: x[1])
    # Pick smallest, median, largest
    test_preds = [
        sorted_preds[0],
        sorted_preds[len(sorted_preds) // 2],
        sorted_preds[-1],
    ]

    all_ok = True
    for pid, parquet_n in test_preds:
        uri = WDT_DIRECT + pid
        iter_, hdt_n = doc.search_triples("", uri, "")

        t0 = time.perf_counter()
        iter_total = 0
        iter_qq = 0
        for s, _, o in iter_:
            iter_total += 1
            if is_q_entity_uri(s) and is_q_entity_uri(o):
                iter_qq += 1
        elapsed = time.perf_counter() - t0

        ok_total = (iter_total == hdt_n)
        ok_qq = (iter_qq == parquet_n)
        status = "OK" if (ok_total and ok_qq) else "MISMATCH"
        all_ok = all_ok and ok_total and ok_qq

        print(f"  {pid:>8}  "
              f"hdt_count={hdt_n:>13,}  "
              f"iter_total={iter_total:>13,}  "
              f"iter_qq={iter_qq:>13,}  "
              f"parquet={parquet_n:>13,}  "
              f"({elapsed:>5.1f}s)  {status}")

    print(f"\n  Phase 1 result: {'OK — predicate counts are reliable' if all_ok else 'FAILED'}")
    return all_ok


# ============================================================================
# Phase 2 — HDT predicate inventory
# ============================================================================

def phase2_inventory(doc: HDTDocument, max_pid: int = 15000) -> dict[str, int]:
    """Get HDT count for every wdt:Pxxx predicate where xxx in [1, max_pid].

    pyHDT does not expose a way to enumerate the predicate dictionary
    directly, so we brute-force the integer ID range. P-IDs in Wikidata
    are dense in the low range and sparse in the high range; max_pid=15000
    covers all known properties as of 2026.

    Each search_triples call is ~1ms (returns iterator + count, we use
    only the count). Total ~15s for 15k calls.
    """
    print(f"\n=== PHASE 2: HDT predicate inventory (P1...P{max_pid}) ===")
    counts: dict[str, int] = {}
    t0 = time.perf_counter()
    for n in range(1, max_pid + 1):
        pid = f"P{n}"
        uri = WDT_DIRECT + pid
        _, count = doc.search_triples("", uri, "")
        if count > 0:
            counts[pid] = count
        if n % 2000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  scanned up to P{n}, "
                  f"{len(counts):,} non-empty so far  ({elapsed:.1f}s)")
    elapsed = time.perf_counter() - t0
    total = sum(counts.values())
    print(f"\n  Found {len(counts):,} non-empty wdt:* predicates  "
          f"({total:,} total triples in HDT)  ({elapsed:.1f}s)")
    return counts


# ============================================================================
# Phase 3 — Classification
# ============================================================================

def phase3_classify(
    hdt_counts: dict[str, int],
    parquet_counts: Counter,
) -> dict:
    """Compare HDT counts vs parquet counts per predicate.

    Returns a dict with keys 'perfect', 'partial', 'missing',
    'parquet_only_warning' (parquet has rows for a predicate that HDT
    reports as 0 — this would be impossible and indicates a bug).
    """
    print("\n=== PHASE 3: Classification ===")
    perfect: list[tuple[str, int]] = []        # HDT == parquet
    partial: list[tuple[str, int, int]] = []    # HDT > parquet
    missing: list[tuple[str, int]] = []         # HDT > 0, parquet == 0
    parquet_only: list[tuple[str, int]] = []    # impossible

    for pid, hdt_n in hdt_counts.items():
        pq_n = parquet_counts.get(pid, 0)
        if pq_n == 0:
            missing.append((pid, hdt_n))
        elif pq_n == hdt_n:
            perfect.append((pid, hdt_n))
        elif pq_n < hdt_n:
            partial.append((pid, hdt_n, pq_n))
        else:
            # Parquet has more rows than HDT → indicates either duplicates
            # in parquet or HDT count bug. Should never happen.
            parquet_only.append((pid, pq_n))

    # Predicates in parquet that HDT says have 0 triples — also impossible
    for pid, pq_n in parquet_counts.items():
        if pid not in hdt_counts:
            parquet_only.append((pid, pq_n))

    print(f"  perfect (HDT == parquet):  {len(perfect):>6,} predicates")
    print(f"  partial (HDT >  parquet):  {len(partial):>6,} predicates")
    print(f"  missing (parquet == 0):    {len(missing):>6,} predicates")
    if parquet_only:
        print(f"  parquet-only (anomaly):    {len(parquet_only):>6,} predicates "
              f"⚠ INVESTIGATE")

    # Top-20 missing by HDT count — high-impact gaps if real
    if missing:
        missing_sorted = sorted(missing, key=lambda x: x[1], reverse=True)
        print("\n  Top 20 'missing' predicates (parquet=0) by HDT count:")
        for pid, hdt_n in missing_sorted[:20]:
            print(f"    {pid:>8}  HDT={hdt_n:>13,}")

    # Top-20 partial by absolute delta (HDT - parquet)
    if partial:
        partial_sorted = sorted(partial, key=lambda x: x[1] - x[2], reverse=True)
        print("\n  Top 20 'partial' predicates by delta (HDT - parquet):")
        for pid, hdt_n, pq_n in partial_sorted[:20]:
            print(f"    {pid:>8}  HDT={hdt_n:>13,}  "
                  f"parquet={pq_n:>13,}  delta={hdt_n - pq_n:>13,}")

    return {
        "perfect": perfect,
        "partial": partial,
        "missing": missing,
        "parquet_only_anomaly": parquet_only,
        "summary": {
            "perfect_count": len(perfect),
            "partial_count": len(partial),
            "missing_count": len(missing),
            "anomaly_count": len(parquet_only),
            "total_hdt_triples_for_wdt_predicates": sum(hdt_counts.values()),
            "total_parquet_rows": sum(parquet_counts.values()),
        },
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    assert HDT_PATH.exists(), f"HDT not found: {HDT_PATH}"
    assert PARQUET_PATH.exists(), f"parquet not found: {PARQUET_PATH}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"HDT:     {HDT_PATH}")
    print(f"Parquet: {PARQUET_PATH}")

    parquet_counts = parquet_predicate_counts()

    print(f"\nOpening HDT (uses pre-built index; ~30s)...")
    t0 = time.perf_counter()
    doc = HDTDocument(str(HDT_PATH))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    sanity_ok = phase1_sanity(doc, parquet_counts)
    if not sanity_ok:
        print("\n⚠ Phase 1 failed — predicate-restricted counts are NOT reliable. "
              "Phase 2/3 results would be untrustworthy. Aborting.")
        return

    hdt_counts = phase2_inventory(doc)
    report = phase3_classify(hdt_counts, parquet_counts)

    # Persist for follow-up
    with open(OUT_PATH, "w") as f:
        # JSON-friendly: tuples → lists
        out = {
            "summary": report["summary"],
            "perfect": [[p, n] for p, n in report["perfect"]],
            "partial": [[p, h, q] for p, h, q in report["partial"]],
            "missing": [[p, n] for p, n in report["missing"]],
            "parquet_only_anomaly": [[p, n] for p, n in report["parquet_only_anomaly"]],
        }
        json.dump(out, f, indent=2)
    print(f"\nReport written to {OUT_PATH}")

    # Final verdict
    s = report["summary"]
    if s["partial_count"] == 0 and s["missing_count"] == 0 and s["anomaly_count"] == 0:
        print("\n✓ VERDICT: parquet is COMPLETE for wdt:* Q-Q triples.")
        print("  Safe to proceed with Layer 1.5 (node_stats) and Layer 3 (build_n3).")
    else:
        print("\n⚠ VERDICT: parquet may have gaps. Inspect verification.json:")
        if s["partial_count"]:
            print(f"  - {s['partial_count']} predicates have parquet < HDT (could be "
                  f"legitimate non-Q objects or actual missed data).")
        if s["missing_count"]:
            print(f"  - {s['missing_count']} predicates absent from parquet "
                  f"(critical if they are wikibase-item type).")
        if s["anomaly_count"]:
            print(f"  - {s['anomaly_count']} predicates show impossible parquet>HDT "
                  f"or parquet without HDT entry. INVESTIGATE.")


if __name__ == "__main__":
    main()