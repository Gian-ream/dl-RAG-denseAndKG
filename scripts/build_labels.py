"""Layer 1.6 — Lookup English labels for dataset QIDs via HDT.

OBJECTIVE
=========
Given the set of QIDs that appear in our dataset (queries + passages),
fetch each one's English label from the Wikidata HDT dump (using the
rdfs:label predicate) and write a parquet table mapping qid -> label_en.

These labels are used for human-readable inspection: when looking at
queries, passages, or top retrieval results, having "Q42 = Douglas Adams"
makes audit and debugging much faster than raw QIDs.

WHO USES THIS
=============
- Layer 3 (build_n3.py) loads labels.parquet as a WARM CACHE for hub
  label lookups. When BFS encounters a hub QID not in the cache, it
  falls back to a direct HDT search and memoizes — so banned_hubs.parquet
  is self-contained without needing this script to run twice.
- Manual inspection / debugging notebooks.

NO TWO-PHASE WORKFLOW
=====================
A previous design ran build_labels.py twice (once before Layer 3, once
after, to extend with banned_hubs QIDs). That created a chicken-egg
problem: banned_hubs.parquet is produced BY Layer 3, so on first run
its labels would all be None. We removed this dependency by having
build_n3.py do its own HDT lookups for hubs not in the cache.

WHY NOT LABEL EVERYTHING
========================
edges.parquet contains ~100-200M distinct QIDs. Labeling all of them
would take 50+ hours at ~2-3 ms per HDT lookup. We only need labels
for entities we'll actually display: the ~138k dataset QIDs from
queries + passages.

INPUT
=====
- HDT file at data/Wikidata_service/latest-all-06-Jan-2022.hdt
- Dataset QIDs from:
    data/NQ_answer/queries_curated.jsonl       (question_qids + answer_variant_qids)
    data/NQ_answer/passage_entities_curated.parquet  (column: qids List[str])

OUTPUT
======
data/db/labels.parquet
    Columns:
        qid       (str)
        label_en  (str, nullable)

    label_en is null for QIDs that have no English-tagged label in the
    HDT dump. This is rare but happens for very obscure entities.

API USED
========
hdt.HDTDocument:
    HDTDocument(path: str)
        Open the HDT file. Mmap-based; the .hdt.index.v1-1 must be
        adjacent (we already have it from prior step).
    HDTDocument.search_triples(s_uri, p_uri, o_uri) -> (iter, count)
        s_uri/p_uri/o_uri are full URIs as strings, or "" for wildcard.
        Returns a streaming iterator over matching triples plus the
        cardinality count (cheap, no enumeration).
        We use it as: search_triples("<wd:Q42>", "<rdfs:label>", "")
        to enumerate labels of Q42 in all languages.

pyarrow:
    pyarrow.parquet.read_table(path, columns=...)
        Read selected columns from a parquet file efficiently.
    pyarrow.Table.from_arrays + pyarrow.parquet.write_table
        Build and write a parquet table directly from Python lists.

LABEL ENCODING IN HDT
=====================
Wikidata labels are stored as triples like:
    <wd:Q5>  <rdfs:label>  "human"@en
    <wd:Q5>  <rdfs:label>  "essere umano"@it

When pyHDT yields the object string, it includes the surrounding double
quotes AND the language tag suffix:
    obj == '"human"@en'

We filter for objects ending with '@en' and strip:
    obj[1:-4]  → 'human'
                ^   ^
                |   └── len('"@en') = 4
                └── leading '"'

EXPECTED RUNTIME
================
~5-10 min for 138k QIDs. Each lookup is ~2-3 ms cold, faster as more
of the .hdt label section gets paged into the OS cache.

HOW TO RUN
==========
From WSL with the dl-rag-wsl venv (pyHDT and pyarrow installed):
    python scripts/build_labels.py
"""

import json
import time
from pathlib import Path
from hdt import HDTDocument
import pyarrow as pa
import pyarrow.parquet as pq


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
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
PASSAGES_PATH = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"
OUT_PATH = REPO_ROOT / "data" / "db" / "labels.parquet"


# ============================================================================
# URI prefixes (full form used by HDT)
# ============================================================================

WD_ENTITY = "http://www.wikidata.org/entity/"            # subject prefix: wd:Qxxxx
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"  # the label predicate


# ============================================================================
# HDT lookup helper — the core operation of this script
# ============================================================================

def get_label_en(doc: HDTDocument, qid: str) -> str | None:
    """Return the English label for a Wikidata QID, or None if missing.

    Args:
        doc: an opened HDTDocument
        qid: the QID without prefix, e.g. "Q42"

    Returns:
        Plain-text English label (e.g., "Douglas Adams") or None.

    HOW IT WORKS
    ------------
    1. Build the full subject URI by prefixing.
    2. Call search_triples with subject=this URI, predicate=rdfs:label,
       object=wildcard ("").
    3. Iterate over results. Each result is a triple (s, p, o) where
       o is a literal string like '"human"@en' or '"essere umano"@it'.
    4. Pick the first object ending in '@en' and strip the wrapping.

    Most QIDs have an @en label, so this loop usually exits on the first
    iteration. Worst case it iterates through all languages (~50-200) for
    QIDs without @en — still fast.
    """
    s_uri = f"{WD_ENTITY}{qid}"

    # search_triples returns (iterator, count). We discard count (we don't
    # need to know the total upfront).
    iter_, _ = doc.search_triples(s_uri, RDFS_LABEL, "")

    for _, _, obj in iter_:
        # obj is something like '"human"@en'.
        # endswith('@en') is the cheapest possible filter for language.
        if obj.endswith("@en"):
            # Strip leading '"' and trailing '"@en' (4 chars).
            return obj[1:-4]
    return None


# ============================================================================
# Seed collection — gather the QIDs that need labels
# ============================================================================

def collect_dataset_qids() -> set[str]:
    """Read queries_curated.jsonl + passage_entities_curated.parquet and
    return the union of all QIDs.

    DATA SHAPES
    -----------
    queries_curated.jsonl: one JSON object per line.
        {
          "question": "...",
          "answers": ["..."],
          "question_qids": ["Q42", ...],          # list of strings
          "answer_variant_qids": [["Q544", ...]], # list of lists
          ...
        }

    passage_entities_curated.parquet:
        Schema: id (int64), title (str), text (str), qids (list[str])
        We only need the qids column.

    Why use pyarrow not polars: pyarrow is already a dependency for this
    script (we use it to write output), so reusing it for read avoids
    pulling in polars on the WSL venv.
    """
    qids: set[str] = set()

    # --- queries jsonl: native Python json + line iteration ---
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qids.update(obj.get("question_qids") or [])
            for variant in obj.get("answer_variant_qids") or []:
                qids.update(variant or [])

    # --- passages parquet: read only the 'qids' column to save memory ---
    table = pq.read_table(PASSAGES_PATH, columns=["qids"])
    qids_col = table.column("qids")  # ChunkedArray of List<string>

    # ChunkedArray comes in chunks (one per row group of the source file).
    # Each chunk is a ListArray; .flatten() unnests it into a flat string array.
    for chunk in qids_col.chunks:
        flat = chunk.flatten()
        qids.update(flat.to_pylist())

    return qids


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    assert HDT_PATH.exists(), f"HDT not found: {HDT_PATH}"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: figure out what to label ---
    print("Collecting QIDs to label...")
    all_qids = collect_dataset_qids()
    print(f"  dataset QIDs (queries + passages): {len(all_qids):>7,}")

    # --- Phase 2: open HDT (cheap with index already built) ---
    print("\nOpening HDT...")
    t0 = time.perf_counter()
    doc = HDTDocument(str(HDT_PATH))
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # --- Phase 3: lookup loop ---
    # We sort QIDs for reproducibility and locality of reference: lookups
    # for QIDs that are numerically adjacent tend to hit nearby pages of
    # the .hdt and .index files, improving cache utilization.
    print("\nLooking up labels...")
    qid_list = sorted(all_qids)
    labels: list[str | None] = [None] * len(qid_list)

    t0 = time.perf_counter()
    for i, qid in enumerate(qid_list):
        labels[i] = get_label_en(doc, qid)

        # Periodic progress with rate + ETA
        if (i + 1) % 10_000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = len(qid_list) - i - 1
            eta_min = remaining / rate / 60
            print(f"  {i+1:>7,} / {len(qid_list):,}  "
                  f"rate {rate:>5,.0f}/s  ETA {eta_min:.1f} min")

    elapsed = time.perf_counter() - t0
    n_with_label = sum(1 for label in labels if label is not None)
    print(f"\n  done in {elapsed/60:.1f} min")
    print(f"  with @en label: {n_with_label:,} / {len(qid_list):,} "
          f"({100*n_with_label/len(qid_list):.1f}%)")

    # --- Phase 4: write parquet ---
    # pa.Table.from_arrays takes parallel lists/arrays + column names.
    # We pass labels as-is; pyarrow handles the None values as nulls.
    table = pa.Table.from_arrays(
        [pa.array(qid_list), pa.array(labels)],
        names=["qid", "label_en"],
    )
    pq.write_table(table, OUT_PATH, compression="snappy")
    out_kb = OUT_PATH.stat().st_size / 1024
    print(f"\nWrote {OUT_PATH}  ({out_kb:.1f} KB)")


if __name__ == "__main__":
    main()