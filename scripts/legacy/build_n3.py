"""Layer 3 — Build N3 (3-hop neighborhoods) for seed QIDs via BFS with hub-banning.

OBJECTIVE
=========
For each seed QID (entities from question_qids + answer_variant_qids of
the dataset), perform a Breadth-First Search up to depth 3 over the
Wikidata graph (loaded from HDT) and record:

    1. hop_sets:  (origin_qid, neighbor_qid, min_distance)
       For every node reachable within 3 hops, the SHORTEST distance at
       which we first saw it.

    2. banned_hubs: (origin_qid, hub_qid, hub_label, hub_degree, first_seen_dist)
       For every "hub" node (total_degree > THRESHOLD) encountered as a
       potential transit, we record it but DO NOT expand through it.
       This prevents BFS explosion when hitting categorical nodes like
       Q5 (human, 9.7M incoming P31 edges).

WHY HUB-BANNING (NOT PREDICATE-DIRECTION FILTER)
================================================
Two valid strategies exist for taming the explosion:
  (a) Block specific predicate+direction pairs (e.g., "incoming P31")
  (b) Block specific node IDs once they're identified as too connected

We chose (b) because it's more transparent: banned_hubs.parquet is an
audit trail showing exactly which concepts we treat as terminals. The
threshold is a single tunable knob; ban-list emerges from data.

WHY ONLY question + answer QIDs AS SEEDS (~1500 distinct), NOT ALL 138k
=======================================================================
The reranker computes:
    connected_ratio(Q, D) = #q∈Q with N3(q)∩D≠∅ / |Q|
    purity_ratio(Q, D)    = #d∈D with N3(d)∩Q≠∅ / |D|

By symmetry of the undirected graph: d ∈ N3(q) ⟺ q ∈ N3(d).
So purity_ratio is computable from N3(q) alone:
    purity_ratio = |D ∩ ⋃_q N3(q)| / |D|

We don't need N3 of passage entities. The seed pool collapses from
138,258 to ~1500 distinct, making Layer 3 feasible (~5-10h instead
of ~4000h estimated).

INPUT
=====
- HDT file (for triple lookups + label fallback)
- data/db/node_stats.parquet (source for the DuckDB hub-degree index;
  rebuilt to /tmp/node_stats.duckdb on first run, ~30s)
- data/db/labels.parquet (for hub_label warm cache, optional)
- data/NQ_answer/queries_curated.jsonl (for seed QIDs)

OUTPUT
======
data/n3/hop_sets_t5000.parquet
    Columns: qid (str), neighbor_qid (str), min_distance (uint8)
    One row per (origin, reachable-neighbor) pair. min_distance is the
    shortest distance at which the BFS first reached neighbor.
    Excludes the seed itself (it's at distance 0 by convention, not stored).

data/n3/banned_hubs_t5000.parquet
    Columns: origin_qid, hub_qid, hub_label, hub_degree, first_seen_dist
    One row per (origin, hub-encountered) pair.

API USED
========
hdt.HDTDocument:
    search_triples(s, p, o) -> (iter, count)
        Stream all triples matching the pattern.

pyarrow:
    ParquetWriter(path, schema, compression)
        Streaming parquet writer; calls .write_table(batch) to append
        row groups, .close() to finalize the footer.

ALGORITHM (single seed, max_dist=3)
====================================
    visited = {seed: 0}
    banned  = []
    frontier = {seed}

    for wave in 1..3:
        next_frontier = empty
        for node in frontier:
            if node != seed AND degree(node) > THRESHOLD:
                # Hub: keep in visited (we saw it at its first-seen distance)
                # but DON'T enumerate its neighbors.
                log to banned (deduped)
                continue
            for neighbor in neighbors_q(node):
                if neighbor not in visited:
                    visited[neighbor] = wave
                    add to next_frontier
        frontier = next_frontier

    # Write hop rows (excluding seed itself)
    # Write banned rows

KEY POINT ON DISTANCES
======================
A hub like Q5 is added to visited at the wave WHERE WE FIRST SAW IT
(typically wave 1 if a seed has Q42 → P31 → Q5). Its `first_seen_dist`
in banned_hubs is also that wave. We just don't expand it further.
So Q5 appears in hop_sets at min_distance=1, and in banned_hubs with
first_seen_dist=1.

EXPECTED RUNTIME (with multiprocessing)
========================================
BFS is HDT-I/O bound (mmap reads on a 166 GB file). Single-threaded
saturates one core but barely uses disk; with 12 workers we overlap I/O
across cores. Per-seed cost varies wildly (1-60s+); seeds exceeding
SEED_TIMEOUT_S are aborted with a TIMEOUT log.

Targets:
  - Dry-run (100 seeds, 12 workers): ~5-15 min
  - Full run (~1500 seeds): ~1-3 hours

PARALLELIZATION + TIMEOUT
==========================
Each seed is processed in a worker pool via imap_unordered. Workers
inherit the loaded `degree` (~10 GB) and `label` dicts from the parent
via Linux fork() with copy-on-write — no IPC pickling of large data.
Each worker opens its own HDTDocument (the file mmap is OS-shared).

A signal.SIGALRM-based timeout caps single-seed BFS at SEED_TIMEOUT_S.
On timeout the seed is logged as TIMEOUT, no rows are written for it,
and the worker continues with the next seed.

For quick validation, run dry-run mode (env DRY_RUN=1) which processes
only 100 random seeds. Use this to calibrate threshold + identify slow
seeds before committing to the full run.

HOW TO RUN
==========
From WSL with dl-rag-wsl venv. PYTHONUNBUFFERED=1 prevents Python from
block-buffering stdout when piped to tee (otherwise the log appears
empty for minutes despite the script printing).

    # Dry run on 100 random seeds, 12 workers
    PYTHONUNBUFFERED=1 DRY_RUN=1 python scripts/legacy/build_n3.py 2>&1 | tee /tmp/build_n3_dry.log

    # Full run, 12 workers
    PYTHONUNBUFFERED=1 python scripts/legacy/build_n3.py 2>&1 | tee /tmp/build_n3_full.log

    # Tune worker count (e.g., 4 workers for low memory):
    PYTHONUNBUFFERED=1 N_WORKERS=4 python scripts/legacy/build_n3.py
"""

import json
import os
import random
import signal
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from hdt import HDTDocument


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
NODE_STATS_PATH = REPO_ROOT / "data" / "db" / "node_stats.parquet"
LABELS_PATH = REPO_ROOT / "data" / "db" / "labels.parquet"
QUERIES_PATH = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"

# DuckDB index over node_stats — built once, kept in /tmp (tmpfs).
# /tmp is fast (RAM-backed) and ephemeral (rebuilt if missing). With
# 12 read-only worker connections, the OS page-cache shares across
# them so total RAM stays bounded (~1-2 GB resident).
NODE_STATS_DB_PATH = Path("/tmp/node_stats.duckdb")

OUT_DIR = REPO_ROOT / "data" / "n3"
HOP_SETS_OUT = OUT_DIR / "hop_sets_t5000.parquet"
BANNED_HUBS_OUT = OUT_DIR / "banned_hubs_t5000.parquet"


# ============================================================================
# Tuning knobs
# ============================================================================

# Hub threshold: total_degree above which we don't expand a node.
# 5000 is a balance: Q42 (150) is well below, Q5 (9.7M) well above,
# popular real entities (a famous person ~ 1000-2000) stay below.
THRESHOLD = 5000

# Maximum BFS depth. The user's metrics define "within 3-hop", so 3.
MAX_DIST = 3

# Flush parquet every N seeds processed. Aggressive flushing because
# losing a 10-seed-worth-of-work on Ctrl+C is fine; losing a 100-seed
# worth (the previous default) is annoying for dry-run debugging.
FLUSH_EVERY_N_SEEDS = 10

# Per-seed timeout in seconds. If BFS for a single seed runs longer
# than this, we abort that seed (signal.SIGALRM) and continue with the
# next one, logging it as TIMEOUT. Prevents a single pathological seed
# from blocking the entire pipeline.
#
# CAVEAT: signal.SIGALRM is delivered between Python bytecodes, so if
# we're stuck inside a single C-level call (e.g., a slow search_triples
# enumeration), the timeout fires only when the call returns. In practice
# BFS spends most time in the Python iteration loop, so the alarm
# fires promptly enough.
SEED_TIMEOUT_S = 60

# Number of worker processes for parallel BFS. BFS is HDT-I/O bound
# (mmap reads on a 166 GB file), so parallel workers overlap I/O wait
# across cores. With 24 cores typical on WSL, 12 workers is a balance:
# leaves headroom for OS, page cache, and the parent's parquet writer.
# Override via env: N_WORKERS=4 python scripts/legacy/build_n3.py
N_WORKERS = int(os.environ.get("N_WORKERS", "12"))

# Dry-run mode: process only this many randomly-sampled seeds.
DRY_RUN_N = 100


# ============================================================================
# URI helpers
# ============================================================================

WD_ENTITY = "http://www.wikidata.org/entity/"
WDT_DIRECT = "http://www.wikidata.org/prop/direct/"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
WD_ENTITY_LEN = len(WD_ENTITY)
WDT_DIRECT_LEN = len(WDT_DIRECT)


def is_q_entity_uri(uri: str) -> bool:
    """True iff URI is a pure Q-entity (excludes statement nodes Q42-uuid).

    Same logic as in hdt_export.py: pure ^Q\\d+$ after the prefix.
    """
    if not uri.startswith(WD_ENTITY):
        return False
    last = uri[WD_ENTITY_LEN:]
    if not last.startswith("Q"):
        return False
    return "-" not in last and last[1:].isdigit()


def uri_to_qid(uri: str) -> str:
    """Strip the WD_ENTITY prefix from a URI to get the bare Qxxxx form."""
    return uri[WD_ENTITY_LEN:]


def get_label(qid: str, doc: HDTDocument, cache: dict[str, str | None]) -> str | None:
    """Return the English label for `qid`, using `cache` first, then HDT.

    The cache starts as the {qid: label_en} dict loaded from labels.parquet.
    For QIDs not in cache (typical for hub QIDs that aren't in the dataset
    seed pool — Q5 "human", Q4167836 "Wikimedia category", etc.), we fall
    back to a direct HDT search and memoize the result.

    This is what allows banned_hubs.parquet to have hub_label populated
    for ALL hubs without requiring a second pass through build_labels.py
    or a downstream JOIN.

    HDT LABEL FORMAT
    ----------------
    Triples are stored as:
        <wd:Q5> <rdfs:label> "human"@en
        <wd:Q5> <rdfs:label> "essere umano"@it
    We iterate matches and pick the first object ending in '@en'. The
    object string includes the surrounding double quotes plus the
    language tag, so we strip leading '"' and trailing '"@en' (4 chars).

    Cost: ~1-3 ms per uncached lookup. Total run cost: a few hundred
    unique hubs across all seeds → well under 1 second.
    """
    if qid in cache:
        return cache[qid]
    s_uri = f"{WD_ENTITY}{qid}"
    iter_, _ = doc.search_triples(s_uri, RDFS_LABEL, "")
    for _, _, obj in iter_:
        if obj.endswith("@en"):
            lbl = obj[1:-4]
            cache[qid] = lbl
            return lbl
    cache[qid] = None
    return None


# ============================================================================
# BFS core — single seed, ~30 lines
# ============================================================================

def get_degrees_batch(
    qids: list[str],
    db: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Batch-lookup total_degree for many QIDs in a single DuckDB query.

    The classic alternative is a Python dict {qid: total_degree} loaded
    once, but for 94M entries that's ~10-12 GB per process and explodes
    memory under multiprocessing fork-COW (refcount writes diverge pages).

    We instead query a pre-built DuckDB file (read-only, OS-cached, shared
    across worker processes via the OS page cache) and pull just the
    degrees we need for the current BFS frontier.

    HOW THE QUERY WORKS
    -------------------
    We register the input QIDs as an Arrow table named `frontier_qids`,
    then INNER JOIN against `node_stats` on qid. DuckDB plans this as a
    hash join: build hash on the smaller side (input qids), probe the
    larger node_stats with index on qid. Returns only the matched rows.

    For wave-3 frontiers ~50k qids, a single call is ~50-100ms.

    Returns a dict {qid: total_degree}. QIDs not in node_stats are
    absent from the dict (caller's `.get(qid, 0)` handles default).
    """
    if not qids:
        return {}
    qids_table = pa.table({"qid": qids})
    db.register("frontier_qids", qids_table)
    try:
        rows = db.execute(
            "SELECT n.qid, n.total_degree FROM node_stats n "
            "INNER JOIN frontier_qids f ON n.qid = f.qid"
        ).fetchall()
    finally:
        db.unregister("frontier_qids")
    return dict(rows)


def bfs_3_waves(
    seed_qid: str,
    doc: HDTDocument,
    db: duckdb.DuckDBPyConnection,
    label: dict[str, str | None],
    threshold: int = THRESHOLD,
    max_dist: int = MAX_DIST,
) -> tuple[dict[str, int], list[tuple[str, str | None, int, int]]]:
    """BFS up to max_dist hops from seed, with hub-banning.

    Args:
        seed_qid: starting QID (stripped form, e.g. "Q42")
        doc: opened HDTDocument
        db: read-only DuckDB connection to node_stats (for hub-degree lookup)
        label: dict QID -> English label or None — used as a cache by
               get_label, mutated in-place when hubs are looked up via HDT
        threshold: degree above which a node is treated as terminal hub
        max_dist: maximum BFS depth (3 for our metrics)

    Returns:
        (visited, banned)
        visited: dict QID -> min_distance, EXCLUDING the seed itself
        banned: list of (hub_qid, hub_label, hub_degree, first_seen_dist),
                deduplicated within this seed's BFS

    HOT LOOP DETAIL
    ---------------
    Per wave:
      1. Batch-lookup degrees of all non-seed nodes in the current frontier
         via DuckDB (single query). This avoids 50k individual dict.get()
         which would be 50k Python refcount-writes — bad under fork-COW.
      2. For each node: if degree > threshold, log to banned and skip
         expansion. Otherwise enumerate via 2 search_triples calls (out + in).
      3. Filter results: predicate must be wdt:*, other endpoint must be
         a pure Q-entity (not statement node).
    """
    # --- Seed-hub guard (option A1, simplest hub-handling strategy) ---
    # If the seed itself has degree > threshold, skip BFS entirely.
    # Rationale: 3-hop neighborhood of a hub seed (e.g., Q30=USA, deg=2.3M)
    # is dominated by hub-bridges, offers no discriminative signal, and
    # would either OOM or hit SEED_TIMEOUT_S. Per seed_degree_stats.py,
    # ~17% of queries (174/1000 at t=5000) hit this branch.
    # We log the seed in `banned` (origin_qid == hub_qid, first_seen_dist=0)
    # so downstream can identify these queries and apply dense-only fallback
    # (KG-score = 0 for all candidate passages).
    seed_deg = get_degrees_batch([seed_qid], db).get(seed_qid, 0)
    if seed_deg > threshold:
        seed_lbl = get_label(seed_qid, doc, label)
        return {}, [(seed_qid, seed_lbl, seed_deg, 0)]

    # visited: QID -> wave number when first seen (0 for seed)
    visited: dict[str, int] = {seed_qid: 0}

    # banned: list of (hub_qid, label, degree, first_seen_dist)
    # banned_set: set of hub_qids to dedupe within this seed
    banned: list[tuple[str, str | None, int, int]] = []
    banned_set: set[str] = set()

    # Current BFS frontier (QIDs at current wave's frontier to expand)
    frontier: set[str] = {seed_qid}

    for wave in range(1, max_dist + 1):
        next_frontier: set[str] = set()

        # --- Batch-lookup degrees for all non-seed nodes in this frontier ---
        # The seed is always expanded regardless of degree, so exclude it
        # from the lookup. For wave 1 frontier == {seed_qid}, this returns {}.
        non_seed_frontier = [n for n in frontier if n != seed_qid]
        deg_map = get_degrees_batch(non_seed_frontier, db) if non_seed_frontier else {}

        for node in frontier:
            # --- Hub check: skip expansion if node is too connected ---
            if node != seed_qid:
                deg = deg_map.get(node, 0)
                if deg > threshold:
                    if node not in banned_set:
                        banned_set.add(node)
                        # first_seen_dist = the wave we first saw this node,
                        # which is the wave we entered into visited[node].
                        # get_label uses HDT fallback when the hub isn't in
                        # the labels.parquet cache (typical case for hubs).
                        banned.append(
                            (node, get_label(node, doc, label), deg, visited[node])
                        )
                    continue  # don't enumerate this hub's neighbors

            # --- Enumerate outgoing edges of `node` ---
            node_uri = f"{WD_ENTITY}{node}"
            out_iter, _ = doc.search_triples(node_uri, "", "")
            for _, p, o in out_iter:
                # Filter: only wdt:* predicates pointing to Q-entities
                if (p.startswith(WDT_DIRECT)
                        and is_q_entity_uri(o)):
                    nq = uri_to_qid(o)
                    if nq not in visited:
                        visited[nq] = wave
                        next_frontier.add(nq)

            # --- Enumerate incoming edges of `node` ---
            in_iter, _ = doc.search_triples("", "", node_uri)
            for s, p, _ in in_iter:
                if (p.startswith(WDT_DIRECT)
                        and is_q_entity_uri(s)):
                    nq = uri_to_qid(s)
                    if nq not in visited:
                        visited[nq] = wave
                        next_frontier.add(nq)

        frontier = next_frontier

    # We don't store seed -> seed at distance 0 in hop_sets (redundant).
    # Caller can re-add it if needed.
    visited.pop(seed_qid, None)
    return visited, banned


# ============================================================================
# I/O — load inputs, write outputs
# ============================================================================

def load_seeds() -> list[str]:
    """Collect question + answer QIDs from queries_curated.jsonl.

    Returns a sorted list of distinct QIDs. Sorted for reproducibility
    (same chunk boundaries on re-runs).
    """
    qids: set[str] = set()
    with QUERIES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qids.update(obj.get("question_qids") or [])
            for variant in obj.get("answer_variant_qids") or []:
                qids.update(variant or [])
    return sorted(qids)


def setup_duckdb_index_if_needed() -> None:
    """Ensure /tmp/node_stats.duckdb exists with an index on qid.

    Builds the file once from node_stats.parquet:
      CREATE TABLE node_stats (qid VARCHAR, total_degree UBIGINT)
      INSERT FROM 'node_stats.parquet'
      CREATE INDEX idx_qid ON node_stats(qid)

    Subsequent runs reuse the file. Stored in /tmp (tmpfs) — fast for
    read access, automatically wiped on WSL reboot (no harm, rebuilt
    on next run in ~30s).

    Why DuckDB and not just keep the parquet:
      - parquet has no point-lookup index; without one, each
        `WHERE qid = ?` is a full file scan. With index, lookups are
        microseconds.
      - DuckDB persistent file format embeds the index physically. Read-
        only connections from multiple workers share the OS page cache.
    """
    if NODE_STATS_DB_PATH.exists():
        size_mb = NODE_STATS_DB_PATH.stat().st_size / (1024 ** 2)
        print(f"DuckDB index ready: {NODE_STATS_DB_PATH} ({size_mb:.0f} MB)")
        return

    print(f"Building DuckDB index at {NODE_STATS_DB_PATH} (one-time, ~30-60s)...")
    t0 = time.perf_counter()
    con = duckdb.connect(str(NODE_STATS_DB_PATH))
    try:
        con.execute(
            "CREATE TABLE node_stats AS "
            f"SELECT qid, total_degree FROM '{NODE_STATS_PATH.as_posix()}'"
        )
        con.execute("CREATE INDEX idx_qid ON node_stats(qid)")
    finally:
        con.close()
    elapsed = time.perf_counter() - t0
    size_mb = NODE_STATS_DB_PATH.stat().st_size / (1024 ** 2)
    print(f"  done in {elapsed:.1f}s ({size_mb:.0f} MB)")


def load_label_dict() -> dict[str, str | None]:
    """Load labels.parquet into a {qid: label_en} dict.

    Returns empty dict if labels.parquet doesn't exist (we tolerate
    missing labels — banned_hubs rows just have None in hub_label).
    """
    if not LABELS_PATH.exists():
        print(f"  labels.parquet not found — proceeding without labels")
        return {}
    print("Loading labels.parquet...")
    t0 = time.perf_counter()
    table = pq.read_table(LABELS_PATH, columns=["qid", "label_en"])
    qids = table.column("qid").to_pylist()
    labels = table.column("label_en").to_pylist()
    d = dict(zip(qids, labels))
    print(f"  loaded {len(d):,} labels in {time.perf_counter() - t0:.1f}s")
    return d


# Schemas declared once, reused for all chunk writes
HOPS_SCHEMA = pa.schema([
    ("qid",          pa.string()),
    ("neighbor_qid", pa.string()),
    ("min_distance", pa.uint8()),
])

BANNED_SCHEMA = pa.schema([
    ("origin_qid",      pa.string()),
    ("hub_qid",         pa.string()),
    ("hub_label",       pa.string()),
    ("hub_degree",      pa.uint64()),
    ("first_seen_dist", pa.uint8()),
])


def flush_chunks(
    hops_writer: pq.ParquetWriter,
    banned_writer: pq.ParquetWriter,
    hops_buf: list[tuple[str, str, int]],
    banned_buf: list[tuple[str, str, str | None, int, int]],
) -> None:
    """Append accumulated rows to the parquet writers and clear buffers."""
    if hops_buf:
        # Transpose list-of-tuples to columnar arrays
        hops_table = pa.Table.from_arrays(
            [
                pa.array([r[0] for r in hops_buf]),
                pa.array([r[1] for r in hops_buf]),
                pa.array([r[2] for r in hops_buf], type=pa.uint8()),
            ],
            schema=HOPS_SCHEMA,
        )
        hops_writer.write_table(hops_table)
        hops_buf.clear()

    if banned_buf:
        banned_table = pa.Table.from_arrays(
            [
                pa.array([r[0] for r in banned_buf]),
                pa.array([r[1] for r in banned_buf]),
                pa.array([r[2] for r in banned_buf]),
                pa.array([r[3] for r in banned_buf], type=pa.uint64()),
                pa.array([r[4] for r in banned_buf], type=pa.uint8()),
            ],
            schema=BANNED_SCHEMA,
        )
        banned_writer.write_table(banned_table)
        banned_buf.clear()


# ============================================================================
# Multiprocessing scaffolding
# ============================================================================
#
# WHY MULTIPROCESSING
# -------------------
# BFS is per-seed embarrassingly parallel: each seed's BFS is independent.
# The hot loop is dominated by HDT mmap reads (I/O wait, not CPU). With low
# CPU utilization on a single thread, parallel workers overlap I/O and saturate
# disk bandwidth. Expected speedup: 6-10x on this workload.
#
# MEMORY MODEL
# ------------
# - HDT_PATH is read-only file mmap'd by each worker. The OS deduplicates
#   pages so total RAM doesn't multiply.
# - `degree` dict (~10 GB) is loaded ONCE in the parent before Pool() and
#   inherited by workers via fork() with copy-on-write. Reading the dict
#   from worker code doesn't trigger COW, so memory stays bounded.
# - `label` dict (~10 MB) inherited the same way. Workers MUTATE their copy
#   when calling get_label HDT fallback, which COW-copies the affected pages.
#   Each worker effectively has its own label cache; modest duplication
#   (~MB-scale).
# - Each worker opens its OWN HDTDocument (post-fork) because the C++ binding
#   internal state isn't guaranteed to survive fork cleanly.
#
# RESULTS FLOW
# ------------
# Workers process one seed each (imap_unordered, dynamic load balance) and
# return (seed_qid, visited, banned, elapsed, status). The parent collects
# results, buffers them, flushes to parquet every FLUSH_EVERY_N_SEEDS.
# Pickling visited/banned over IPC: ~200KB-20MB per seed, negligible.

# Worker globals — module-level so that fork() inheritance carries them to
# children. The HDT instance and the DuckDB connection are opened POST-fork
# inside _init_worker (their internal state isn't fork-safe). The label
# cache is loaded in main() pre-fork and inherited via COW; mutations
# (HDT label fallback) cause per-worker COW divergence on the few touched
# pages — totally fine for ~14 MB of dict.
_doc: "HDTDocument | None" = None
_db: "duckdb.DuckDBPyConnection | None" = None
_label: "dict[str, str | None] | None" = None


class SeedTimeout(Exception):
    """Raised by the SIGALRM handler when a single seed's BFS exceeds budget."""
    pass


def _alarm_handler(signum, frame):  # noqa: ARG001 — signature required by signal
    """SIGALRM handler — converts the alarm into a Python exception we can catch.

    Note: Python only delivers signals between bytecodes, so this fires
    when the BFS loop yields control back to Python (typical case).
    """
    raise SeedTimeout()


def _init_worker() -> None:
    """Initializer run once per worker process AFTER fork.

    On Linux (default `fork` start method), the worker has already inherited
    `_label` from the parent via COW. We open per-worker:
      1. HDTDocument (C++ binding internal state isn't fork-safe; underlying
         file mmap is OS-shared so no extra RAM).
      2. DuckDB read-only connection to /tmp/node_stats.duckdb. Multiple
         read-only connections to the same file are explicitly supported;
         OS page cache shares physical memory across them.
      3. SIGALRM handler for per-seed timeout.

    Total init cost: ~30s for HDT open + <1s for DuckDB connect, in parallel
    across workers.
    """
    global _doc, _db
    _doc = HDTDocument(str(HDT_PATH))
    _db = duckdb.connect(str(NODE_STATS_DB_PATH), read_only=True)
    signal.signal(signal.SIGALRM, _alarm_handler)


def _process_seed(seed_qid: str) -> tuple[str, dict[str, int],
                                          list[tuple[str, str | None, int, int]],
                                          float, str]:
    """Worker entry point — process a single seed with timeout.

    Returns:
        (seed_qid, visited, banned, elapsed_s, status)
        status is "ok" or "timeout".

    On timeout, visited/banned are empty (we don't return partial BFS
    state because the alarm interrupted us mid-frontier — the `visited`
    dict at that point has inconsistent distances).
    """
    t0 = time.perf_counter()
    signal.alarm(SEED_TIMEOUT_S)
    try:
        visited, banned = bfs_3_waves(seed_qid, _doc, _db, _label)
        # Detect seed-hub skip: bfs_3_waves returned with origin == hub itself.
        # In that case visited is empty and banned holds exactly the seed row.
        if not visited and len(banned) == 1 and banned[0][0] == seed_qid:
            status = "seed_hub"
        else:
            status = "ok"
    except SeedTimeout:
        visited, banned = {}, []
        status = "timeout"
    finally:
        signal.alarm(0)
    elapsed = time.perf_counter() - t0
    return (seed_qid, visited, banned, elapsed, status)


# ============================================================================
# Main driver
# ============================================================================

def main() -> None:
    # --- Pre-conditions ---
    assert HDT_PATH.exists(), f"HDT not found: {HDT_PATH}"
    assert NODE_STATS_PATH.exists(), (
        f"node_stats.parquet not found at {NODE_STATS_PATH}\n"
        "Run scripts/node_stats.py first (Layer 1.5)."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dry_run = bool(int(os.environ.get("DRY_RUN", "0")))
    print(f"Mode:        {'DRY-RUN (' + str(DRY_RUN_N) + ' random seeds)' if dry_run else 'FULL'}", flush=True)
    print(f"Threshold:   {THRESHOLD}", flush=True)
    print(f"Max dist:    {MAX_DIST}", flush=True)
    print(f"Workers:     {N_WORKERS}", flush=True)
    print(f"Seed timeout: {SEED_TIMEOUT_S}s", flush=True)
    print(f"Flush every: {FLUSH_EVERY_N_SEEDS} seeds", flush=True)
    print(flush=True)

    # --- Load seed list ---
    seeds = load_seeds()
    print(f"Total seed QIDs from dataset: {len(seeds):,}", flush=True)
    if dry_run:
        random.seed(42)  # reproducible sample
        seeds = random.sample(seeds, min(DRY_RUN_N, len(seeds)))
        print(f"  → sampled {len(seeds)} for dry run", flush=True)

    # --- Build DuckDB index (one-time) and load label cache ---
    # Degrees go to DuckDB (file in /tmp, ~5 GB on disk, OS-cached for
    # multi-worker access). Labels are small (~14 MB), kept as Python dict
    # in module global so workers inherit via fork-COW.
    setup_duckdb_index_if_needed()
    global _label
    _label = load_label_dict()

    # --- Open writers (parent only) ---
    print(f"\nWriting to:", flush=True)
    print(f"  {HOP_SETS_OUT}", flush=True)
    print(f"  {BANNED_HUBS_OUT}", flush=True)
    hops_writer = pq.ParquetWriter(HOP_SETS_OUT, HOPS_SCHEMA, compression="snappy")
    banned_writer = pq.ParquetWriter(BANNED_HUBS_OUT, BANNED_SCHEMA, compression="snappy")

    # --- Spawn worker pool ---
    # fork() inherits parent memory (COW); workers open HDT post-fork.
    # No initargs because passing the 10 GB `degree` dict over IPC would
    # pickle it 12 times. fork inheritance is free.
    print(f"\nSpawning {N_WORKERS} workers (each will open HDT, ~30s parallel)...", flush=True)

    hops_buf: list[tuple[str, str, int]] = []
    banned_buf: list[tuple[str, str, str | None, int, int]] = []

    total_hops = 0
    total_banned = 0
    n_ok = 0
    n_seed_hub = 0
    n_timeout = 0
    slow_seeds: list[tuple[str, float]] = []  # for end-of-run summary
    t_start = time.perf_counter()

    print(f"\nStarting BFS over {len(seeds)} seeds...", flush=True)
    print(f"  status seed       reach    ban   t(s)  cum(min)  rate(s/s)  ETA(min)",
          flush=True)
    print(f"  ------ ---------- ------- ----- ----- --------- ---------- --------",
          flush=True)

    try:
        with Pool(N_WORKERS, initializer=_init_worker) as pool:
            # imap_unordered = dynamic load balance; results in completion order
            for i, result in enumerate(pool.imap_unordered(_process_seed, seeds)):
                seed_qid, visited, banned, elapsed, status = result

                if status in ("ok", "seed_hub"):
                    # Both cases write whatever rows BFS produced.
                    # - ok:        normal BFS, visited+banned populated
                    # - seed_hub:  visited={}, banned=[(seed,...,0)] (1 row)
                    for nq, dist in visited.items():
                        hops_buf.append((seed_qid, nq, dist))
                    for hub_qid, hub_label, hub_deg, first_seen in banned:
                        banned_buf.append(
                            (seed_qid, hub_qid, hub_label, hub_deg, first_seen)
                        )
                    total_hops += len(visited)
                    total_banned += len(banned)
                    if status == "ok":
                        n_ok += 1
                    else:
                        n_seed_hub += 1
                else:
                    n_timeout += 1

                # Track slow seeds (>30s) for end-of-run report
                if elapsed > 30.0:
                    slow_seeds.append((seed_qid, elapsed))

                # PER-SEED LOG (verbose) — every seed, with progress + ETA
                cum_elapsed = time.perf_counter() - t_start
                rate = (i + 1) / cum_elapsed if cum_elapsed > 0 else 0.0
                eta_min = (len(seeds) - i - 1) / rate / 60 if rate > 0 else 0.0
                marker = {"ok": "OK     ", "seed_hub": "SKIPHUB", "timeout": "TIMEOUT"}[status]
                print(f"  {marker} {seed_qid:<10} "
                      f"{len(visited):>7,} {len(banned):>5} "
                      f"{elapsed:>5.1f} {cum_elapsed/60:>9.1f} "
                      f"{rate:>10.2f} {eta_min:>8.1f}",
                      flush=True)

                # Flush buffers periodically (resumability + memory bound)
                if (i + 1) % FLUSH_EVERY_N_SEEDS == 0:
                    flush_chunks(hops_writer, banned_writer, hops_buf, banned_buf)
                    print(f"  -- flushed at seed {i+1} --", flush=True)

            # Final flush
            flush_chunks(hops_writer, banned_writer, hops_buf, banned_buf)

    finally:
        # Always close writers — guarantees parquet footer is written.
        hops_writer.close()
        banned_writer.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed/60:.1f} min", flush=True)
    print(f"  seeds processed:   {len(seeds):,}  "
          f"(ok: {n_ok}, seed_hub: {n_seed_hub}, timeout: {n_timeout})",
          flush=True)
    print(f"  total hop rows:    {total_hops:,}", flush=True)
    print(f"  total banned rows: {total_banned:,}", flush=True)
    print(f"  hops_sets size:    {HOP_SETS_OUT.stat().st_size / (1024**2):.1f} MB",
          flush=True)
    print(f"  banned_hubs size:  {BANNED_HUBS_OUT.stat().st_size / 1024:.1f} KB",
          flush=True)

    # Slow seeds report — useful for tuning threshold
    if slow_seeds:
        slow_seeds.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  top {min(20, len(slow_seeds))} slowest seeds (>30s):", flush=True)
        for sq, t in slow_seeds[:20]:
            print(f"    {sq:>10}  {t:>6.1f}s", flush=True)


if __name__ == "__main__":
    main()