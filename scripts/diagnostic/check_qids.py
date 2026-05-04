"""Sanity check: verify all QIDs from ReFiNed entity linking are pure Q-entities.

Pre-flight check before launching the long HDT export. Loads QIDs from the
dataset files (queries + passages), checks that each one matches ^Q\\d+$
(no statement nodes Q42-uuid, no P-ids, no full URIs left in by mistake).

Run from Windows venv (uses polars from pyproject.toml).
"""

import json
import re
from collections import Counter
from pathlib import Path
import polars as pl


def _find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError(f"Could not find repo root above {start}")


REPO_ROOT = _find_repo_root()
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
PASSAGES_PATH = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

# Pure Q-entity: starts with Q, followed by digits only, nothing else
QID_RE = re.compile(r"^Q\d+$")


def collect_query_qids() -> tuple[list[str], list[str]]:
    """Return (question_qids_flat, answer_qids_flat) from queries jsonl.

    answer_variant_qids is List[List[str]] — flatten one level.
    """
    q_qids: list[str] = []
    a_qids: list[str] = []
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q_qids.extend(obj.get("question_qids", []) or [])
            for variant in obj.get("answer_variant_qids", []) or []:
                a_qids.extend(variant or [])
    return q_qids, a_qids


def collect_passage_qids() -> list[str]:
    """Return flat list of QIDs from passage_entities_curated.parquet."""
    df = pl.read_parquet(PASSAGES_PATH)
    # 'qids' is List[String] — explode to flat column then materialize
    return df.select(pl.col("qids").explode()).drop_nulls().to_series().to_list()


def report(name: str, qids: list[str]) -> set[str]:
    """Print stats for one source, return set of distinct non-conforming QIDs."""
    total = len(qids)
    distinct = set(qids)
    bad = [q for q in distinct if not QID_RE.match(q)]
    print(f"\n[{name}]")
    print(f"  total occurrences: {total:,}")
    print(f"  distinct QIDs:     {len(distinct):,}")
    print(f"  non-conforming:    {len(bad):,}")
    if bad:
        # Show up to 10 examples and a frequency-like sample
        print(f"  examples (up to 10): {bad[:10]}")
    return set(bad)


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Queries:   {QUERIES_PATH}")
    print(f"Passages:  {PASSAGES_PATH}")

    q_qids, a_qids = collect_query_qids()
    p_qids = collect_passage_qids()

    bad_q = report("question_qids", q_qids)
    bad_a = report("answer_variant_qids", a_qids)
    bad_p = report("passage_qids", p_qids)

    # Aggregate
    all_distinct = set(q_qids) | set(a_qids) | set(p_qids)
    all_bad = bad_q | bad_a | bad_p
    print(f"\n[aggregate]")
    print(f"  distinct QIDs across all sources: {len(all_distinct):,}")
    print(f"  distinct non-conforming:          {len(all_bad):,}")

    # Inspect what kind of "bad" we have, if any
    if all_bad:
        print("\n[bad QID classification]")
        kinds = Counter()
        for q in all_bad:
            if "-" in q and q.startswith("Q"):
                kinds["statement_node (Q\\d+-...)"] += 1
            elif q.startswith("P"):
                kinds["property_id (Pxxx)"] += 1
            elif q.startswith("http"):
                kinds["full_uri"] += 1
            elif q.startswith("Q"):
                kinds["other_Q_form"] += 1
            else:
                kinds["non_Q_prefix"] += 1
        for k, v in kinds.most_common():
            print(f"  {k}: {v}")
    else:
        print("\nAll QIDs match ^Q\\d+$ — nessun outlier. Safe to proceed.")


if __name__ == "__main__":
    main()