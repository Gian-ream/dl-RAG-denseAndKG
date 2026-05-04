"""Layer 4 — KG runtime scorer (unified, disk-persisted).

OBIETTIVO
=========
Runtime API usata dal KG-rerank step (Step 5 del proposal).
Per una coppia (query, passage), dove:

    Q = set di question_qids della query
    D = set di entity QID nel passage

calcola tre quantità a una o più threshold t e una o più distanze massime k:

    connected_ratio(Q, D, t, k) = |{q ∈ Q : ∃ d ∈ D con dist(q,d) ≤ k}| / |Q|
    purity_ratio(Q, D, t, k)    = |{d ∈ D : ∃ q ∈ Q con dist(d,q) ≤ k}| / |D|
    kg_score(Q, D, t, k)        = connected_ratio · purity_ratio

dove `dist(q, d) ≤ k` è calcolato sul grafo filtrato a *bridges* non-hub
(nodi con `total_degree ≤ t`). Gli endpoint q, d sono SEMPRE preservati
(la threshold filtra i bridge intermedi, non gli endpoint — vedi
PROJECT_NOTES §6.2).

ARCHITETTURA — DuckDB persistito
=================================
Una sola classe `KGScorer` con tre modalità di init:

1) **In-memory** (`db_path=None`): connessione `:memory:`, tutto in RAM.
   Carica n1.parquet e edges.parquet ogni volta. Da usare solo per debug
   isolati senza disco. RAM peak ~10-15 GB.

2) **Read-only** (`read_only=True`): apre `db_path` in RO; richiede file
   esistente con tabelle n1+edges già buildate. Modalità tipica per i
   worker di multiprocessing — il main builda una volta, i worker
   condividono il page cache OS senza duplicare RAM.

3) **Read-write con auto-build** (default): apre `db_path` (lo crea se
   manca), e per ciascuna delle tabelle n1/edges:
   - se esiste già → skip (init istantaneo)
   - se manca → builda dal parquet, crea l'indice (per n1), committa.
   Idempotente: chiamate successive sullo stesso db_path partono in ~1s.

Schema delle tabelle:
- `n1` (~93M righe): qid VARCHAR, neighbor VARCHAR, neighbor_degree UBIGINT
  + indice B-tree su qid.
- `edges` (661M righe): subject VARCHAR, object VARCHAR (proiettato dal
  parquet originale, predicate scartato). NESSUN indice — DuckDB usa hash
  join per JOIN su colonne (non lookup ad indice), quindi gli indici su
  edges sarebbero solo overhead di build.

REACHABILITY DEFINITION
=======================
Bridge nodes (intermediate path nodes) sono filtrati: solo nodi con
`total_degree ≤ threshold` sono ammessi come bridge. Gli endpoint q, d
non vengono MAI filtrati.

- dist=1: edge diretto q-d
    Truth: d ∈ N1(q)
    SQL: `SELECT 1 FROM n1 WHERE qid=q AND neighbor=d LIMIT 1`
    Threshold ignorata.

- dist=2: ∃ x ∈ N1(q) ∩ N1(d) con deg(x) ≤ t
    SQL: self-join `n1 a JOIN n1 b ON a.neighbor = b.neighbor`
         con `a.neighbor_degree ≤ t` (un solo check: x = a.neighbor =
         b.neighbor, grado identico).

- dist=3: ∃ x ∈ N1(q), y ∈ N1(d), edge x-y in `edges`, x e y non-hub.
    SQL: SPLIT in due branch UNION ALL (una direzione per branch).
    La formulazione con OR sulla direzione dell'edge forzava il planner
    DuckDB a un cross-product n1×n1×edges = trilioni di righe (verificato
    empiricamente: hang >5h). Split form lascia che il planner costruisca
    hash join chains pulite per ogni direzione (~1-30s ciascuna).

QUERY UNIFICATA min_dist (ottimizzazione per griglia)
======================================================
Il core helper `_reachable_pairs_min_dist` esegue UNA query SQL con 4 CTE
(d1, d2, d3a, d3b) ognuna proietta una colonna letterale `dist`; il
SELECT finale aggrega `MIN(dist) GROUP BY q, d` collassando cammini
multipli sulla stessa coppia. Restituisce mappa `(q, d) → min_dist`.

Vantaggio: per una griglia di ablation `distance × threshold` (3 × 6 = 18
celle), invece di 18 query SQL si fanno solo 6 (una per threshold), e
ogni distance ≤ k si deriva in Python filtrando la mappa. Speed-up ~3x
sull'overhead di parsing/planning DuckDB.

Le API "single configuration" (`connected_ratio`, `purity_ratio`,
`kg_score`, `kg_components`) usano lo stesso helper internamente, scartando
`min_dist` (default cap a 3) — così non c'è duplicazione di SQL.

THREAD SAFETY
=============
Single-threaded sulla connessione DuckDB. Per multi-process: aprire il file
`data/kg.duckdb` in `read_only=True` da ciascun worker. DuckDB supporta
read-only multi-process pulito; il page cache OS viene condiviso (niente
duplicazione di RAM tra worker).

USAGE
=====
    from utils.kg import KGScorer

    # Prima volta (RW, builda data/kg.duckdb in ~5 min):
    scorer = KGScorer()

    # Single configuration (backward compatible):
    components = scorer.kg_components(Q, D, threshold=5000)
    # {"connected_ratio": 0.83, "purity_ratio": 0.42, "kg_score": 0.349}

    # Griglia per ablation (preferita):
    df = scorer.kg_components_grid(Q, D)
    # pd.DataFrame con [distance, threshold, cr, pr, kg_score]

    # Batch su lista di coppie:
    df = scorer.kg_components_grid_batch(pairs)
    # Aggiunge query_id, passage_id come colonne id

    # Worker multiprocessing (file già pronto):
    scorer = KGScorer(read_only=True)

Smoke test (richiede data/n1/n1.parquet + data/db/edges.parquet):
    python -m utils.kg
"""

import json
import time
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa


# ============================================================================
# Path resolution
# ============================================================================

def _find_repo_root() -> Path:
    """Risale dai parent fino a trovare la cartella che contiene pyproject.toml.

    Funziona sia quando il modulo è eseguito come script (`__file__` definito)
    sia quando viene importato in modo non-standard (cwd fallback).
    """
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
EDGES_PATH = REPO_ROOT / "data" / "db" / "edges.parquet"
KG_DUCKDB_PATH = REPO_ROOT / "data" / "kg.duckdb"


# ============================================================================
# KGScorer — unified runtime class
# ============================================================================

class KGScorer:
    """Knowledge-graph reachability scorer per coppie (query, passage).

    Carica n1 + edges in DuckDB (default: persistito su disco a
    `data/kg.duckdb`), poi risponde a query batch via SQL.

    Default threshold tuning: 5000 (vedi seed_degree_stats.py — a questo
    cutoff ~17% delle query cadono nella categoria all-hub e ricevono
    KG-score=0; il resto ottiene full reachability check).
    """

    # Set di threshold default per la griglia di ablation. 0 codifica ∞
    # (no filtro). Stesso set usato da ablation_diagnostic.py per
    # mantenere risultati comparabili.
    DEFAULT_THRESHOLDS: tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 0)

    # Set di distanze max default per la griglia. Cumulativo: distance=k
    # significa cammini di lunghezza ≤ k.
    DEFAULT_DISTANCES: tuple[int, ...] = (1, 2, 3)

    # Sentinel per ∞ nei confronti SQL (max int64).
    _INF_SENTINEL: int = 2**63 - 1

    def __init__(
        self,
        db_path: Path | None = KG_DUCKDB_PATH,
        n1_path: Path = N1_PATH,
        edges_path: Path = EDGES_PATH,
        read_only: bool = False,
        verbose: bool = True,
    ) -> None:
        """Inizializza con persistenza disco (default) o in-memory (fallback).

        TRE MODALITÀ MUTUAMENTE ESCLUSIVE
        ----------------------------------
        1) **In-memory** (`db_path=None`): connessione `:memory:`, carica
           n1+edges dai parquet ogni volta. Da usare solo per debug.

        2) **Read-only** (`read_only=True`): apre `db_path` in RO. Verifica
           che contenga le tabelle n1+edges. Errore se file mancante o
           tabelle assenti. Modalità tipica per i worker MP.

        3) **Read-write con auto-build** (default): apre `db_path`, e per
           ciascuna delle tabelle n1/edges:
           - se esiste già → skip
           - se manca → builda dal parquet, crea l'indice, committa.
           Idempotente: chiamate successive sono ~1s.

        DETECTION DELLE TABELLE
        -----------------------
        Usa `information_schema.tables` (standard SQL) per evitare un
        try/except con log spurii.

        EFFETTI COLLATERALI
        -------------------
        Dopo `__init__`:
            self.db   # duckdb.DuckDBPyConnection (file-backed o :memory:)
            db.tables = {n1, edges}
        RAM peak in modalità file-backed: ~1-3 GB working set (DuckDB usa
        mmap, page cache condiviso tra processi).
        """
        # ============================================================
        # Modalità 1 — in-memory (fallback debug)
        # ============================================================
        if db_path is None:
            self._init_in_memory(n1_path, edges_path, verbose)
            return

        # ============================================================
        # Modalità 2 — read-only (worker multiprocessing)
        # ============================================================
        if read_only:
            assert db_path.exists(), (
                f"db_path {db_path} non esiste — buildare prima con "
                f"KGScorer(read_only=False)"
            )
            self.db = duckdb.connect(str(db_path), read_only=True)
            # Sanity check integrità: deve contenere ENTRAMBE le tabelle.
            tables = {
                r[0] for r in self.db.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_name IN ('n1', 'edges')"
                ).fetchall()
            }
            assert tables == {"n1", "edges"}, (
                f"file {db_path} non contiene n1+edges (trovati: {tables})"
            )
            if verbose:
                print(f"Opened {db_path} read-only (worker mode).", flush=True)
            return

        # ============================================================
        # Modalità 3 — RW con auto-build idempotente (default)
        # ============================================================
        # Crea la cartella padre se serve (tipicamente data/ esiste già).
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(db_path))

        # Detection: quali tabelle sono già presenti nel file?
        existing = {
            r[0] for r in self.db.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name IN ('n1', 'edges')"
            ).fetchall()
        }

        # --- Build n1 se mancante ---
        if "n1" not in existing:
            assert n1_path.exists(), (
                f"n1.parquet not found at {n1_path}\n"
                "Run scripts/pipeline/build_n1.py first (Layer 3)."
            )
            if verbose:
                print(f"Building n1 in {db_path} (one-time, ~30s)...", flush=True)
            t0 = time.perf_counter()
            self.db.execute(
                f"CREATE TABLE n1 AS SELECT * FROM '{n1_path.as_posix()}'"
            )
            # Indice B-tree su qid: cruciale per WHERE qid = ? e per i JOIN
            # sui set Q/D (vedi `_reachable_pairs_min_dist`).
            self.db.execute("CREATE INDEX idx_n1_qid ON n1(qid)")
            if verbose:
                n1_rows = self.db.execute("SELECT COUNT(*) FROM n1").fetchone()[0]
                print(
                    f"  n1: {n1_rows:,} rows + index in "
                    f"{time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
        elif verbose:
            print(f"n1 already in {db_path.name}, skip build.", flush=True)

        # --- Build edges se mancante ---
        # Proietta solo subject + object (no predicate). Niente indici:
        # DuckDB usa hash join per JOIN su colonne, gli indici sarebbero
        # overhead inutile (~5-7 min di build, zero beneficio).
        if "edges" not in existing:
            assert edges_path.exists(), f"edges.parquet not found at {edges_path}"
            if verbose:
                print(
                    f"Building edges in {db_path} (one-time, ~3-5 min)...",
                    flush=True,
                )
            t0 = time.perf_counter()
            self.db.execute(
                f"CREATE TABLE edges AS "
                f"SELECT subject, object FROM '{edges_path.as_posix()}'"
            )
            if verbose:
                edges_rows = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
                print(
                    f"  edges: {edges_rows:,} rows in "
                    f"{time.perf_counter()-t0:.1f}s",
                    flush=True,
                )
        elif verbose:
            print(f"edges already in {db_path.name}, skip build.", flush=True)

        if verbose:
            print(f"Ready ({db_path}).\n", flush=True)

    def _init_in_memory(
        self, n1_path: Path, edges_path: Path, verbose: bool,
    ) -> None:
        """Branch in-memory dell'init — duplicato isolato per chiarezza.

        Carica n1+edges in `:memory:` ogni volta. Niente persistenza.
        Usare solo per debug, perché ogni processo paga ~5 min di build
        e ~10-15 GB di RAM duplicata.
        """
        assert n1_path.exists(), (
            f"n1.parquet not found at {n1_path}\n"
            "Run scripts/pipeline/build_n1.py first (Layer 3)."
        )
        assert edges_path.exists(), f"edges.parquet not found at {edges_path}"

        self.db = duckdb.connect(":memory:")

        if verbose:
            print(f"Loading n1.parquet → in-memory table...", flush=True)
        t0 = time.perf_counter()
        self.db.execute(f"CREATE TABLE n1 AS SELECT * FROM '{n1_path.as_posix()}'")
        self.db.execute("CREATE INDEX idx_n1_qid ON n1(qid)")
        if verbose:
            n1_rows = self.db.execute("SELECT COUNT(*) FROM n1").fetchone()[0]
            print(
                f"  {n1_rows:,} rows in n1, indexed in "
                f"{time.perf_counter()-t0:.1f}s",
                flush=True,
            )

        if verbose:
            print(f"Loading edges.parquet → in-memory table (~1-3 min)...", flush=True)
        t0 = time.perf_counter()
        self.db.execute(
            f"CREATE TABLE edges AS "
            f"SELECT subject, object FROM '{edges_path.as_posix()}'"
        )
        if verbose:
            edges_rows = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            print(
                f"  {edges_rows:,} rows in edges loaded in "
                f"{time.perf_counter()-t0:.1f}s",
                flush=True,
            )
            print(f"Ready (in-memory).\n", flush=True)

    # ------------------------------------------------------------------------
    # Threshold normalization
    # ------------------------------------------------------------------------

    def _resolve_threshold(self, threshold: int | None) -> int:
        """Normalizza la threshold in un int valido per SQL.

        L'utente esprime "nessun filtro" in due modi (None o 0); il SQL deve
        poter fare `WHERE neighbor_degree <= ?` in modo uniforme. Soluzione:
        trasformare ∞ in un sentinel int molto grande (max int64, garantito
        >> qualunque grado realistico — il nodo più connesso di Wikidata ha
        ~10^7 edges).

        None/0 → 2^63-1.   >0 → identità.
        """
        if threshold is None or threshold == 0:
            return self._INF_SENTINEL
        return threshold

    # ------------------------------------------------------------------------
    # Diagnostic: pair-level reachability
    # ------------------------------------------------------------------------

    def is_reachable(
        self, q: str, d: str, threshold: int | None = 5000,
        verbose: bool = False,
    ) -> bool:
        """True iff dist(q, d) ≤ 3 nel grafo filtrato a deg ≤ threshold.

        SCOPO
        -----
        Diagnostica per UNA singola coppia (q, d). Per il workload del
        reranker (Q × D batch su 100 passages) usare `kg_components_grid`
        o `_reachable_pairs_min_dist` — processano insiemi interi in un
        solo SQL, ammortizzando il costo del planner.

        FLUSSO DI ESECUZIONE
        --------------------
        Short-circuit a 4 livelli — appena una distanza dà HIT, ritorna True
        senza eseguire le successive (le distanze più lontane sono sempre
        più costose):
            dist=1  ─→ HIT? return True   (point lookup, ~1-5 ms)
                    ─→ miss
            dist=2  ─→ HIT? return True   (self-join, ~50-200 ms)
                    ─→ miss
            dist=3a ─→ HIT? return True   (3-way join, ~300-2000 ms)
                    ─→ miss
            dist=3b ─→ return r is not None  (direzione opposta)

        Output: bool. True se ∃ percorso di lunghezza ≤ 3 con bridge non-hub.
        """
        # Risolve None/0 → sentinel ∞.
        t = self._resolve_threshold(threshold)

        # ============================================================
        # dist=1: edge diretto (no threshold sugli endpoint)
        # ============================================================
        # n1 contiene la riga (q, d) se esiste un edge q→d O d→q (build_n1
        # fa UNION delle due direzioni). Quindi il check è simmetrico.
        if verbose: print("  dist=1...", end="", flush=True)
        r = self.db.execute(
            "SELECT 1 FROM n1 WHERE qid = ? AND neighbor = ? LIMIT 1",
            [q, d],
        ).fetchone()
        if verbose: print(f" {'HIT' if r else 'miss'}", flush=True)
        if r is not None:
            return True

        # ============================================================
        # dist=2: bridge x condiviso, deg(x) ≤ t
        # ============================================================
        # Self-join su `neighbor`: x è in N1(q) e in N1(d).
        # Un solo controllo di degree (x = a.neighbor = b.neighbor → grado
        # identico, è proprietà del nodo non dell'edge).
        if verbose: print("  dist=2...", end="", flush=True)
        r = self.db.execute(
            """
            SELECT 1
            FROM n1 a
            INNER JOIN n1 b ON a.neighbor = b.neighbor
            WHERE a.qid = ? AND b.qid = ? AND a.neighbor_degree <= ?
            LIMIT 1
            """,
            [q, d, t],
        ).fetchone()
        if verbose: print(f" {'HIT' if r else 'miss'}", flush=True)
        if r is not None:
            return True

        # ============================================================
        # dist=3 dir.1: edge x→y in `edges` originale
        # ============================================================
        # Cammino: q —N1→ x —edge.subj→edge.obj— y ←N1— d
        # Entrambi i bridge x e y devono essere non-hub.
        if verbose: print("  dist=3 dir1...", end="", flush=True)
        r = self.db.execute(
            """
            SELECT 1
            FROM n1 a
            INNER JOIN edges e ON e.subject = a.neighbor
            INNER JOIN n1 b ON e.object = b.neighbor
            WHERE a.qid = ? AND b.qid = ?
              AND a.neighbor_degree <= ? AND b.neighbor_degree <= ?
            LIMIT 1
            """,
            [q, d, t, t],
        ).fetchone()
        if verbose: print(f" {'HIT' if r else 'miss'}", flush=True)
        if r is not None:
            return True

        # ============================================================
        # dist=3 dir.2: edge y→x (direzione opposta)
        # ============================================================
        # Le due direzioni sono complementari su grafo logicamente non
        # orientato (le direzioni edge sono solo come vengono memorizzate
        # i triple SPO). Insieme coprono tutti i cammini dist=3.
        if verbose: print("  dist=3 dir2...", end="", flush=True)
        r = self.db.execute(
            """
            SELECT 1
            FROM n1 a
            INNER JOIN edges e ON e.object = a.neighbor
            INNER JOIN n1 b ON e.subject = b.neighbor
            WHERE a.qid = ? AND b.qid = ?
              AND a.neighbor_degree <= ? AND b.neighbor_degree <= ?
            LIMIT 1
            """,
            [q, d, t, t],
        ).fetchone()
        if verbose: print(f" {'HIT' if r else 'miss'}", flush=True)
        return r is not None

    # ------------------------------------------------------------------------
    # Core batch helper: unified min-distance query
    # ------------------------------------------------------------------------

    def _reachable_pairs_min_dist(
        self, Q: list[str], D: list[str], threshold: int | None,
    ) -> dict[tuple[str, str], int]:
        """UNA query SQL → mappa (q, d) → min_dist per ogni coppia raggiungibile.

        SCOPO
        -----
        Cuore del motore di scoring batch. Calcola in UN solo SQL TUTTE le
        coppie (q, d) ∈ Q × D raggiungibili (dist ≤ 3, filtrato a threshold)
        e annota per ciascuna la **distanza minima** trovata.

        Tutte le API pubbliche (cr/pr/kg_score/kg_components/grid) si
        appoggiano a questo helper — nessuna duplicazione di SQL.

        FORMA DELLA QUERY
        -----------------
        WITH d1  AS (..., 1 AS dist),    -- dist=1 (no threshold)
             d2  AS (..., 2 AS dist),    -- dist=2 (1 threshold check)
             d3a AS (..., 3 AS dist),    -- dist=3 dir.1 (2 threshold check)
             d3b AS (..., 3 AS dist)     -- dist=3 dir.2 (2 threshold check)
        SELECT q, d, MIN(dist) AS min_dist
        FROM (d1 UNION ALL d2 UNION ALL d3a UNION ALL d3b)
        GROUP BY q, d

        L'aggregato `MIN(dist) GROUP BY q, d` collassa cammini multipli
        sulla stessa coppia: se (q, d) è raggiungibile sia a dist=2 che a
        dist=3, ottiene `min_dist=2`.

        BIND PARAMETERS
        ---------------
        5 placeholder `?`:
            posizione 1 → d2:   a.neighbor_degree <= ?
            posizioni 2-3 → d3a: due check su bridge
            posizioni 4-5 → d3b: due check su bridge
        Stesso valore di threshold passato a tutti.

        REGISTRAZIONE Q E D
        -------------------
        Q e D sono materializzati come Arrow Tables (zero-copy dalla list)
        e registrati nella connessione DuckDB con `db.register(...)` come
        view virtuali `Q_set` / `D_set`. Il try/finally garantisce
        l'`unregister` anche su exception (no view leak).

        OUTPUT
        ------
        dict[tuple[str, str], int]
            Chiavi: tutte e sole le coppie (q, d) ∈ Q × D raggiungibili
                    entro 3 hop con threshold dato.
            Valori: min_dist ∈ {1, 2, 3}.
            Coppie NON raggiungibili: assenti dalla mappa.

        TIMING
        ------
        ~50-200 ms per (Q ~3, D ~30) a RAM warm. Dominato dalle CTE d3a/d3b.
        """
        t = self._resolve_threshold(threshold)

        # Materializza Q e D come Arrow tables (zero-copy dalla list di
        # stringhe), registra come view virtuali per il SQL.
        Q_table = pa.table({"qid": Q})
        D_table = pa.table({"qid": D})
        self.db.register("Q_set", Q_table)
        self.db.register("D_set", D_table)

        try:
            rows = self.db.execute(
                """
                WITH
                    -- dist=1: edge diretto (no threshold sugli endpoint)
                    d1 AS (
                        SELECT n.qid AS q, n.neighbor AS d, 1 AS dist
                        FROM n1 n
                        INNER JOIN Q_set ON n.qid = Q_set.qid
                        INNER JOIN D_set ON n.neighbor = D_set.qid
                    ),
                    -- dist=2: bridge condiviso non-hub
                    d2 AS (
                        SELECT a.qid AS q, b.qid AS d, 2 AS dist
                        FROM n1 a
                        INNER JOIN n1 b ON a.neighbor = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                    ),
                    -- dist=3 dir.1: edge x→y
                    d3a AS (
                        SELECT a.qid AS q, b.qid AS d, 3 AS dist
                        FROM n1 a
                        INNER JOIN edges e ON e.subject = a.neighbor
                        INNER JOIN n1 b ON e.object = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                          AND b.neighbor_degree <= ?
                    ),
                    -- dist=3 dir.2: edge y→x
                    d3b AS (
                        SELECT a.qid AS q, b.qid AS d, 3 AS dist
                        FROM n1 a
                        INNER JOIN edges e ON e.object = a.neighbor
                        INNER JOIN n1 b ON e.subject = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                          AND b.neighbor_degree <= ?
                    )
                SELECT q, d, MIN(dist) AS min_dist
                FROM (
                    SELECT * FROM d1
                    UNION ALL SELECT * FROM d2
                    UNION ALL SELECT * FROM d3a
                    UNION ALL SELECT * FROM d3b
                )
                GROUP BY q, d
                """,
                [t, t, t, t, t],
            ).fetchall()
        finally:
            # Cleanup garantito (anche su exception): rimuove le view
            # virtuali per non lasciare riferimenti pendenti agli Arrow
            # buffer.
            self.db.unregister("Q_set")
            self.db.unregister("D_set")

        return {(q, d): mind for q, d, mind in rows}

    # ------------------------------------------------------------------------
    # Single-configuration APIs (backward compatible)
    # ------------------------------------------------------------------------
    # Tutte derivate dallo stesso `_reachable_pairs_min_dist`. Si differenziano
    # solo per cosa estraggono dal risultato; nessuna duplicazione di SQL.

    def connected_ratio(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """Frazione di Q raggiungibile ad almeno un d ∈ D entro 3 hop.

        connected_ratio = |{q ∈ Q : ∃ d ∈ D con dist(q,d) ≤ 3}| / |Q|

        Vicina al recall lato-query.

        Edge case Q vuoto → 0.0 (early return, niente SQL).
        Edge case D vuoto → 0.0 (Q_reached è ∅).
        """
        if not Q:
            return 0.0
        pair_to_md = self._reachable_pairs_min_dist(Q, D, threshold)
        # min_dist ≤ 3 sempre vero (la query non genera dist > 3), quindi
        # basta prendere le chiavi distinte sul lato Q.
        Q_reached = {q for (q, _) in pair_to_md.keys()}
        return len(Q_reached) / len(Q)

    def purity_ratio(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """Frazione di D raggiungibile ad almeno un q ∈ Q entro 3 hop.

        purity_ratio = |{d ∈ D : ∃ q ∈ Q con dist(d,q) ≤ 3}| / |D|

        Vicina alla precisione lato-passage. Per simmetria del grafo non
        orientato, dist(d,q) ≤ 3 ⟺ dist(q,d) ≤ 3 — calcolo unico con
        connected_ratio via `_reachable_pairs_min_dist`.
        """
        if not D:
            return 0.0
        pair_to_md = self._reachable_pairs_min_dist(Q, D, threshold)
        D_reached = {d for (_, d) in pair_to_md.keys()}
        return len(D_reached) / len(D)

    def kg_score(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """KG-score scalare = connected_ratio · purity_ratio.

        Penalizza sia bassa copertura della query sia passage off-topic.
        """
        if not Q or not D:
            return 0.0
        pair_to_md = self._reachable_pairs_min_dist(Q, D, threshold)
        Q_reached = {q for (q, _) in pair_to_md.keys()}
        D_reached = {d for (_, d) in pair_to_md.keys()}
        cr = len(Q_reached) / len(Q)
        pr = len(D_reached) / len(D)
        return cr * pr

    def kg_components(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> dict[str, float]:
        """Calcola in UN solo batch cr, pr, kg_score per single threshold.

        API consigliata quando servono tutti e tre i valori a una sola
        configurazione (es. log per query, debug). Per griglie di
        ablation usare `kg_components_grid`.

        Ritorna dict[str, float] con chiavi connected_ratio, purity_ratio,
        kg_score. Edge case Q/D vuoto → tutti 0.0.
        """
        if not Q or not D:
            return {"connected_ratio": 0.0, "purity_ratio": 0.0, "kg_score": 0.0}
        pair_to_md = self._reachable_pairs_min_dist(Q, D, threshold)
        Q_reached = {q for (q, _) in pair_to_md.keys()}
        D_reached = {d for (_, d) in pair_to_md.keys()}
        cr = len(Q_reached) / len(Q)
        pr = len(D_reached) / len(D)
        return {"connected_ratio": cr, "purity_ratio": pr, "kg_score": cr * pr}

    # ------------------------------------------------------------------------
    # Grid API: single pair → DataFrame
    # ------------------------------------------------------------------------

    def kg_components_grid(
        self,
        Q: list[str],
        D: list[str],
        distances: Iterable[int] = DEFAULT_DISTANCES,
        thresholds: Iterable[int | None] = DEFAULT_THRESHOLDS,
    ) -> pd.DataFrame:
        """Griglia (distance × threshold) per UNA coppia (Q, D) → DataFrame.

        SCOPO
        -----
        API principale per ablation studies. Esegue UNA query SQL per ogni
        threshold (annota min_dist), poi in Python deriva ciascuna distance
        filtrando `min_dist ≤ k`. Niente lavoro SQL ripetuto sulla
        dimensione distance.

        Per griglia 3×6: 6 query SQL totali invece di 18 (≈3x speed-up
        sull'overhead di parsing/planning).

        OUTPUT
        ------
        pd.DataFrame con colonne:
            distance         : int   (∈ distances)
            threshold        : int|None (chiave originale, 0/None per ∞)
            connected_ratio  : float ∈ [0, 1]
            purity_ratio     : float ∈ [0, 1]
            kg_score         : float ∈ [0, 1]   (= cr · pr)

        Numero di righe: len(distances) × len(thresholds). Default: 18.

        EDGE CASE
        ---------
        Q o D vuoto: ritorna DataFrame con score=0 per tutte le righe della
        griglia, niente SQL eseguita.

        NOTA INTERPRETATIVA — threshold inerte a distance=1
        ----------------------------------------------------
        A distance=1 non ci sono nodi-bridge (l'edge è diretto). Le righe
        distance=1 saranno IDENTICHE al variare del threshold a parità di
        Q, D. Non è un bug. A distance=2 un bridge è soggetto a threshold,
        a distance=3 due bridge → effetto threshold più pronunciato.

        TIMING
        ------
        ~300-1200 ms su Q×D piccoli (6 query SQL × ~50-200 ms ciascuna).
        """
        distances = list(distances)
        thresholds = list(thresholds)

        # Validazione: solo {1, 2, 3} sono supportate (la query ha 4 CTE max).
        for k in distances:
            assert k in (1, 2, 3), (
                f"distance {k} non valida — supportate solo (1, 2, 3)"
            )

        # Edge case Q o D vuoto: ritorna DataFrame di zeri, no SQL.
        if not Q or not D:
            zero_rows = [
                {
                    "distance": k,
                    "threshold": t,
                    "connected_ratio": 0.0,
                    "purity_ratio": 0.0,
                    "kg_score": 0.0,
                }
                for t in thresholds for k in distances
            ]
            return pd.DataFrame(zero_rows)

        nQ, nD = len(Q), len(D)
        rows: list[dict] = []

        # Loop esterno sulle threshold (1 query SQL ciascuna). Per ogni
        # threshold otteniamo (q, d) → min_dist e deriviamo TUTTE le distance
        # in Python filtrando la mappa — gratis.
        for t in thresholds:
            pair_to_md = self._reachable_pairs_min_dist(Q, D, t)

            for k in distances:
                Q_reached = {q for (q, _), md in pair_to_md.items() if md <= k}
                D_reached = {d for (_, d), md in pair_to_md.items() if md <= k}
                cr = len(Q_reached) / nQ
                pr = len(D_reached) / nD
                rows.append({
                    "distance": k,
                    "threshold": t,
                    "connected_ratio": cr,
                    "purity_ratio": pr,
                    "kg_score": cr * pr,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------------
    # Grid API: batch over multiple (query, passage) pairs
    # ------------------------------------------------------------------------

    def kg_components_grid_batch(
        self,
        pairs: list[tuple[str, list[str], str, list[str]]],
        distances: Iterable[int] = DEFAULT_DISTANCES,
        thresholds: Iterable[int | None] = DEFAULT_THRESHOLDS,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Wrapper batch su lista di coppie → DataFrame "lungo".

        SCOPO
        -----
        Comodo quando hai centinaia/migliaia di coppie (query, passage) da
        scorare. Loopa su `kg_components_grid` per ogni coppia e concatena
        i risultati, aggiungendo colonne identifier.

        INPUT
        -----
        pairs : list[tuple[query_id, Q, passage_id, D]]
            Ogni elemento:
                query_id    : str — id query (es. NQ id)
                Q           : list[str] — question_qids
                passage_id  : str — id passage (es. wiki id)
                D           : list[str] — entity QIDs del passage

        OUTPUT
        ------
        pd.DataFrame con colonne:
            query_id         : str
            passage_id       : str
            distance         : int
            threshold        : int|None
            connected_ratio  : float
            purity_ratio     : float
            kg_score         : float

        Numero righe: len(pairs) × len(distances) × len(thresholds).

        ESEMPIO USO
        -----------
            pairs = [
                ("nq_42", ["Q937"], "wiki_123", ["Q183", "Q142"]),
                ("nq_42", ["Q937"], "wiki_456", ["Q31"]),
            ]
            df = scorer.kg_components_grid_batch(pairs)
            df.pivot_table(
                index=["query_id", "passage_id"],
                columns=["distance", "threshold"],
                values="kg_score",
            )

        TIMING
        ------
        Lineare in len(pairs). Per 1000 coppie × 18 celle: ~5-20 min
        single-thread. Per workload più grandi → multiprocessing con
        `KGScorer(read_only=True)` su `data/kg.duckdb`.
        """
        pairs = list(pairs)
        sub_dfs: list[pd.DataFrame] = []

        for idx, (query_id, Q, passage_id, D) in enumerate(pairs):
            sub = self.kg_components_grid(Q, D, distances, thresholds)
            sub.insert(0, "passage_id", passage_id)
            sub.insert(0, "query_id", query_id)
            sub_dfs.append(sub)

            if verbose and (idx + 1) % 50 == 0:
                print(f"  processed {idx+1}/{len(pairs)} pairs", flush=True)

        if not sub_dfs:
            return pd.DataFrame(columns=[
                "query_id", "passage_id", "distance", "threshold",
                "connected_ratio", "purity_ratio", "kg_score",
            ])
        return pd.concat(sub_dfs, ignore_index=True)


# ============================================================================
# Smoke test
# ============================================================================

def _smoke_test() -> None:
    """Sanity check end-to-end di KGScorer.

    COSA FA
    -------
    1. Init in modalità default (RW, persiste in data/kg.duckdb). Se il file
       esiste già le tabelle non vengono ricostruite — init ~1s.
       Se non esiste, builda da n1.parquet + edges.parquet — ~5 min.
    2. Carica la prima query e il primo passage dai file curated.
    3. Diagnostica `is_reachable` sulla prima coppia (verbose).
    4. Diagnostica `_reachable_pairs_min_dist` (mostra distribuzione min_dist).
    5. `kg_components_grid` con default — stampa DataFrame 3×6.
    6. `kg_components_grid_batch` su 2 coppie sintetiche — verifica schema
       lungo con query_id/passage_id.

    OUTPUT ATTESO (RAM warm, file kg.duckdb esistente)
    ---------------------------------------------------
    Init total:                            ~1 s
    is_reachable:                          1-3 s
    _reachable_pairs_min_dist:             50-200 ms
    kg_components_grid (3×6 = 18 righe):   300-1200 ms
    kg_components_grid_batch (2 coppie):   600-2400 ms
    """
    import pyarrow.parquet as pq

    queries_path = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
    passages_path = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

    print("=" * 70)
    print("KGScorer smoke test")
    print("=" * 70)

    # --- Fase 1: init ---
    print()
    t0 = time.perf_counter()
    scorer = KGScorer()
    print(f"Init total: {time.perf_counter() - t0:.1f}s")

    # --- Fase 2: carica query 0 e passage 0 ---
    with queries_path.open("r", encoding="utf-8") as f:
        q_obj = json.loads(next(f))
    Q = q_obj.get("question_qids") or []
    print(f"\nQuery: {q_obj.get('question', '<no text>')[:80]}")
    print(f"  question_qids: {Q}")
    if not Q:
        print("  ! query has no question_qids — smoke test will short-circuit")
        return

    pass_table = pq.read_table(passages_path).slice(0, 1)
    D = pass_table.column("qids")[0].as_py()
    print(f"  first passage entities ({len(D)}): "
          f"{D[:8]}{'...' if len(D) > 8 else ''}")
    if not D:
        print("  ! passage has no entities — smoke test will short-circuit")
        return

    # --- Fase 3: is_reachable diagnostico ---
    q0, d0 = Q[0], D[0]
    print(f"\nis_reachable({q0}, {d0}, t=5000):")
    t0 = time.perf_counter()
    is_r = scorer.is_reachable(q0, d0, threshold=5000, verbose=True)
    print(f"  → {is_r}  (total {(time.perf_counter()-t0)*1000:.1f} ms)")

    # --- Fase 4: distribuzione min_dist a t=5000 ---
    print(f"\n_reachable_pairs_min_dist(Q, D, t=5000):")
    t0 = time.perf_counter()
    mind = scorer._reachable_pairs_min_dist(Q, D, threshold=5000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(mind)} reachable pairs in {elapsed_ms:.1f} ms")
    if mind:
        dist_counts = {1: 0, 2: 0, 3: 0}
        for v in mind.values():
            dist_counts[v] += 1
        print(f"  distribution: dist=1: {dist_counts[1]}, "
              f"dist=2: {dist_counts[2]}, dist=3: {dist_counts[3]}")

    # --- Fase 5: griglia completa 3×6 ---
    t0 = time.perf_counter()
    df = scorer.kg_components_grid(Q, D)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\nkg_components_grid(Q, D)  ({elapsed_ms:.1f} ms):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False))

    # --- Fase 6: batch su 2 coppie sintetiche ---
    pairs = [
        ("nq_smoke_0", Q, "wiki_smoke_a", D),
        ("nq_smoke_0", Q, "wiki_smoke_b", D[:5] if len(D) >= 5 else D),
    ]
    t0 = time.perf_counter()
    df_batch = scorer.kg_components_grid_batch(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\nkg_components_grid_batch (2 pairs)  ({elapsed_ms:.1f} ms):")
    print(f"  shape: {df_batch.shape}, columns: {list(df_batch.columns)}")
    print(df_batch.head(6).to_string(index=False))

    print("\nSmoke test OK.")


if __name__ == "__main__":
    _smoke_test()