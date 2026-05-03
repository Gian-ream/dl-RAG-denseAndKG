"""Layer 4 — KG runtime scorer with multi-threshold reachability.

OBJECTIVE
=========
Provide the runtime API used by the KG-rerank step (Step 5 of the proposal).
For a (query, passage) pair, where:

    Q = set of question_qids of the query
    D = set of entity QIDs in the passage

compute three quantities at one or more degree thresholds t:

    connected_ratio(Q, D, t) = |{q ∈ Q : ∃ d ∈ D with dist(q,d) ≤ 3}| / |Q|
    purity_ratio(Q, D, t)    = |{d ∈ D : ∃ q ∈ Q with dist(d,q) ≤ 3}| / |D|
    kg_score(Q, D, t)        = connected_ratio · purity_ratio

where `dist(q, d) ≤ 3` is computed in the graph filtered to non-hub *bridges*
(nodes with `total_degree ≤ t`). Endpoints q, d are always preserved
regardless of their own degree (the threshold prunes path-bridges, not
endpoints — see PROJECT_NOTES §6.2).

ARCHITECTURE (vedi §4.10)
=========================
DuckDB-uniform, single connection, in-memory:
- `n1.parquet` (~93M rows of (qid, neighbor, neighbor_degree)) loaded into
  table `n1` with index on `qid` (used for fast point-lookup `WHERE qid = ?`).
- `edges.parquet` (661M Q-Q wdt:* triples) loaded into table `edges`
  projecting only `subject, object`. NO indices on edges — DuckDB uses hash
  joins for JOIN conditions like `e.subject = a.neighbor`, ignoring B-tree
  indices on the joined columns. Building them is wasted init time (~5-7 min).
- All distance checks (1, 2, 3) expressed as SQL queries — no Python set ops
  in the hot path. Single-pair `is_reachable` runs up to 4 short-circuiting
  queries; batch `_reachable_pairs(Q, D, t)` runs ONE big query with WITH/CTE
  clauses (Common Table Expression, introduces a temporary table through WITH which lives
  only in the singe declaration statement),
  returning all reached endpoints in one shot.
- Batch scoring of (query, passage): one SQL returns BOTH `Q_reached` (for
  connected_ratio) AND `D_reached` (for purity_ratio) — the two metrics share
  computation, the SQL is run only once per (Q, D, threshold) tuple.

REACHABILITY DEFINITION
=======================
Bridge nodes (intermediate path nodes) are filtered: only nodes with
`total_degree ≤ threshold` are allowed as bridges. Endpoints q, d are
NEVER filtered (preserves the "evaluate every query" property).

- dist=1: q-d direct edge in the original graph
    Truth condition: d ∈ N1(q)  (equivalently, q ∈ N1(d) by undirected symmetry)
    SQL form: `SELECT 1 FROM n1 WHERE qid=q AND neighbor=d LIMIT 1`
    Threshold ignored — endpoints always preserved.

- dist=2: ∃ x with edge q-x, edge x-d, where x is non-hub
    Truth condition: ∃ x ∈ N1(q) ∩ N1(d) with deg(x) ≤ t
    SQL form: `JOIN n1 a, n1 b ON a.neighbor = b.neighbor`
              `WHERE a.qid=q, b.qid=d, a.neighbor_degree ≤ t`
    NB: filter only on `a.neighbor_degree` — x is the SAME node as
    `a.neighbor` and `b.neighbor` (since they're equal in the join), so one
    degree check suffices.

- dist=3: ∃ x ∈ N1(q), y ∈ N1(d), edge x-y in graph, both x and y non-hub
    Truth condition: ∃ x ∈ N1_filtered(q), y ∈ N1_filtered(d), edge(x,y)
    SQL form: split in TWO queries (UNION ALL), one per edge direction.
    Reason: the "OR-ed direction" formulation (single query with
    `(e.subject = a.neighbor AND e.object = b.neighbor)
      OR (e.object = a.neighbor AND e.subject = b.neighbor)`)
    forces DuckDB to consider a cross-product of `n1 × n1 × edges` =
    trillions of rows, which never completes (verified empirically:
    >5 hours hang). Split form lets the planner build clean hash joins
    on each direction independently (~1-30s per query).

WHY DUCKDB AND NOT PYTHON DICT
==============================
User explicitly chose uniform DuckDB (vedi conversazione 2026-04-29):
- One architecture for all three distance checks (no mixing dict + SQL)
- Python dict + LRU ( Least Recently Used, una strategia di caching: quando la
  cache è piena e devi inserire un nuovo elemento, si butta via
  quello che non viene letto da più tempo) would be ~10-100x faster on single-pair check, but
  for batch scoring (the actual reranker workload) DuckDB query optimizer
  competes well via vectorized execution
- ~10-15 GB RAM peak (acceptable on Windows venv with ≥16 GB host)

MEMORY NOTES
============
On init (Windows venv):
- n1 table:    ~1-3 GB depending on N1 sizes
- edges table: ~5-10 GB (661M rows × 2 string columns, dictionary-encoded)
- 3 indices:   ~3-5 GB
- Total peak:  ~10-18 GB

If RAM is tight, change the `CREATE TABLE` statements for edges to
`CREATE VIEW edges AS SELECT subject, object FROM 'edges.parquet'`.
DuckDB will read from disk per query (slower but ~0 RAM). Indices
won't be available on a VIEW, so dist=3 queries become full scans
(~1-3s each).

THREAD SAFETY
=============
Single-threaded design. The `db` connection MUST NOT be shared across
Python threads — DuckDB serializes shared-connection calls and cursors
can interleave. For multi-process scaling (later if needed), persist
n1+edges into a `data/kg.duckdb` file and have each worker open its
own read-only connection.

USAGE
=====
    from scripts.kg import KGScorer

    scorer = KGScorer()  # loads n1 + edges, ~1-2 min
    components = scorer.kg_components(Q, D, threshold=5000)
    # {"connected_ratio": 0.83, "purity_ratio": 0.42, "kg_score": 0.349}

    # Multi-threshold for ablation
    multi = scorer.kg_score_multi(Q, D)
    # {500: 0.12, 1000: 0.18, 2000: 0.27, 5000: 0.349, 10000: 0.41, 0: 0.78}

Standalone smoke test:
    .venv\\Scripts\\python.exe scripts\\kg.py
"""

import json
import time
from pathlib import Path
from typing import Iterable

import duckdb
import pyarrow as pa


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
N1_PATH = REPO_ROOT / "data" / "n1" / "n1.parquet"
EDGES_PATH = REPO_ROOT / "data" / "db" / "edges.parquet"


# ============================================================================
# KGScorer — main runtime class
# ============================================================================

class KGScorer:
    """Knowledge-graph reachability scorer for (query, passage) pairs.

    Loads n1 + edges into in-memory DuckDB tables with indices, then
    answers batch reachability queries via SQL.

    Default threshold tuning: 5000 (vedi seed_degree_stats.py — at this
    cutoff ~17% of queries fall into the all-hub category and get
    KG-score=0; the rest get full reachability check).
    """

    # Default threshold sweep for `kg_score_multi`. 0 encodes ∞ (no filter).
    # Same set used by ablation_diagnostic.py — keeps results comparable.
    DEFAULT_THRESHOLDS: tuple[int, ...] = (500, 1000, 2000, 5000, 10000, 0)

    # Sentinel for ∞ in SQL comparisons (max int64).
    _INF_SENTINEL: int = 2**63 - 1

    def __init__(
        self,
        n1_path: Path = N1_PATH,
        edges_path: Path = EDGES_PATH,
        verbose: bool = True,
    ) -> None:
        """Carica n1 + edges in DuckDB in-memory.

        FASI DI INIZIALIZZAZIONE
        ------------------------
        1) **Sanity check** sui due path: se n1.parquet manca, l'errore
           rimanda all'utente lo step da rieseguire (build_n1.py).
        2) **Connessione DuckDB in-memory**: `:memory:` → tutto vive in RAM
           per la durata del processo, nessun file `.duckdb` su disco.
           Una sola connessione condivisa per istanza (single-thread, vedi
           `THREAD SAFETY` nel module docstring).
        3) **Tabella `n1`** caricata da parquet con `CREATE TABLE AS SELECT *`
           (materializza in RAM, ~93M righe × 3 colonne ≈ 1-3 GB).
           Schema risultante:
               qid              VARCHAR    -- entità target
               neighbor         VARCHAR    -- vicino diretto
               neighbor_degree  UBIGINT    -- grado totale del vicino
           Subito dopo si crea l'**indice B-tree su `qid`**: ogni query in
           `is_reachable` e `_reachable_pairs` filtra per `n.qid = ?` o
           join sul subset target — l'indice trasforma queste lookup in
           O(log n) invece di full-scan.
        4) **Tabella `edges`** caricata proiettando SOLO `subject, object`
           (no `predicate`): ~661M × 2 stringhe dictionary-encoded ≈ 5-10 GB.
           **NESSUN indice** su edges — vedi nota sotto.

        PERCHÉ NESSUN INDICE SU `edges`
        --------------------------------
        DuckDB usa **hash join** (non lookup ad indice) per JOIN tipo
        `e.subject = a.neighbor` dove RHS è un riferimento a colonna,
        non un literal. Il planner costruisce una hash table sul lato
        più piccolo (tipicamente la N1 filtrata, ~100-1000 righe) e fa
        un probe vettorizzato sull'altro. L'indice B-tree non viene mai
        consultato — nei test empirici creare gli indici aggiungeva
        ~5-7 min di init senza alcun beneficio sulle query di reachability.

        EFFETTI COLLATERALI
        -------------------
        Dopo `__init__`:
            self.db        # duckdb.DuckDBPyConnection (in-memory)
            self.db.tables = {n1, edges}
            RAM peak ≈ 7-15 GB (Windows venv).
        L'istanza è pronta per chiamate batch (`kg_components`,
        `_reachable_pairs`, ...) e single-pair (`is_reachable`).
        """
        # --- Fase 1: sanity check sui dataset richiesti ---
        # Messaggi diversi per i due file: n1 può essere ricostruito,
        # edges è il dataset Layer 1 (rebuild = ore di HDT scan).
        assert n1_path.exists(), (
            f"n1.parquet not found at {n1_path}\n"
            "Run scripts/build_n1.py first (Layer 3)."
        )
        assert edges_path.exists(), f"edges.parquet not found at {edges_path}"

        # --- Fase 2: apertura connessione in-memory ---
        # Tutta la sessione vive su questa singola connessione.
        # `:memory:` significa: nessun file .duckdb persistente, alla
        # chiusura dell'oggetto la RAM viene rilasciata.
        self.db = duckdb.connect(":memory:")

        # --- Fase 3: caricamento n1 + indice su qid ---
        # `CREATE TABLE AS SELECT *` materializza il parquet in RAM
        # (DuckDB lo leggerebbe altrimenti on-demand con un VIEW, vedi
        # MEMORY NOTES nel module docstring).
        if verbose:
            print(f"Loading n1.parquet → in-memory table...", flush=True)
        t0 = time.perf_counter()
        self.db.execute(f"CREATE TABLE n1 AS SELECT * FROM '{n1_path.as_posix()}'")
        # Indice B-tree su qid: usato per `WHERE qid = ?` (point lookup)
        # e per i JOIN `INNER JOIN Q_set ON n.qid = Q_set.qid`.
        # Costruzione: ~5-15s su 93M righe.
        self.db.execute("CREATE INDEX idx_n1_qid ON n1(qid)")
        if verbose:
            n1_rows = self.db.execute("SELECT COUNT(*) FROM n1").fetchone()[0]
            print(f"  {n1_rows:,} rows in n1, indexed in {time.perf_counter()-t0:.1f}s",
                  flush=True)

        # --- Fase 4: caricamento edges (senza indici) ---
        if verbose:
            print(f"Loading edges.parquet → in-memory table (~1-3 min)...", flush=True)
        t0 = time.perf_counter()
        # Project only `subject` and `object` — we don't need `predicate` for
        # reachability (any wdt:* edge counts the same). Risparmia ~30% RAM
        # rispetto a `SELECT *`.
        self.db.execute(
            f"CREATE TABLE edges AS "
            f"SELECT subject, object FROM '{edges_path.as_posix()}'"
        )
        # NOTE: NO indices on edges. DuckDB uses hash joins (not index lookups)
        # for JOIN conditions like `e.subject = a.neighbor` where the right side
        # is a column reference, not a literal. Indices on subject/object would
        # be wasted init time. Reachability is dominated by hash joins on small
        # filtered N1 sets (~100-1000 rows) probed against edges (vectorized).
        if verbose:
            edges_rows = self.db.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            print(f"  {edges_rows:,} rows in edges loaded in "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
            print(f"Ready.\n", flush=True)

    # ------------------------------------------------------------------------
    # Threshold normalization
    # ------------------------------------------------------------------------

    def _resolve_threshold(self, threshold: int | None) -> int:
        """Normalizza il valore di threshold passato dall'utente in un int valido per SQL.

        SCOPO
        -----
        L'utente esprime "nessun filtro" in due modi (None o 0), ma il SQL
        deve poter fare `WHERE neighbor_degree <= ?` in modo uniforme,
        senza generare due varianti di query (una con/senza WHERE).
        Soluzione: trasformare "∞" in un sentinel int molto grande, che
        soddisfa sempre il `<=` rendendo il filtro inerte.

        INPUT
        -----
        threshold : int | None
            None  → ∞ (no filter)
            0     → ∞ (no filter — convenzione dello script ablation)
            >0    → soglia letterale di grado massimo per i bridge

        OUTPUT
        ------
        int : valore da passare al SQL come bind parameter
            None/0 → 2^63 - 1 (= 9223372036854775807, max int64)
            >0     → identità (il valore stesso)

        PERCHÉ 2^63 - 1
        ---------------
        È il massimo intero rappresentabile in int64 (BIGINT in DuckDB).
        Garantito >> qualunque grado realistico (il nodo più connesso di
        Wikidata ha ~10^7 edges). Il filtro `<=` resta sempre vero senza
        ramificare la query.
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
        """True iff dist(q, d) ≤ 3 in the graph filtered to deg ≤ threshold.

        SCOPO
        -----
        Diagnostica / debug per una singola coppia (q, d). Per il workload
        del reranker (Q × D batch su 100 passages) usare `kg_components`
        o `_reachable_pairs` — processano insiemi interi in UN solo SQL,
        ammortizzando il costo del planner.

        INPUT
        -----
        q : str        QID del lato "query" (es. "Q937" = Albert Einstein)
        d : str        QID del lato "passage" (es. "Q183" = Germany)
        threshold : int | None
                       Soglia di grado per i nodi-bridge. None o 0 → ∞
                       (nessun filtro). Tipici: 500, 1000, 2000, 5000, 10000.
        verbose : bool Stampa "  dist=k... HIT/miss" per ogni fase.
                       Utile quando una coppia stalla — fa vedere su quale
                       distance si blocca il planner.

        OUTPUT
        ------
        bool : True se ∃ percorso di lunghezza ≤ 3 tra q e d con tutti
               i bridge filtrati a `total_degree <= threshold`.

        FLUSSO DI ESECUZIONE
        --------------------
        Short-circuit a 4 livelli — appena una distanza dà HIT, ritorna
        True senza eseguire le successive (le distanze più lontane sono
        sempre più costose):
            dist=1  ─→ HIT? return True
                    ─→ miss
            dist=2  ─→ HIT? return True
                    ─→ miss
            dist=3a ─→ HIT? return True   (edge direction subject→object)
                    ─→ miss
            dist=3b ─→ return r is not None  (edge direction object→subject)

        QUERY DESIGN
        ------------
        Tutte le JOIN sono esplicite (`INNER JOIN ... ON ...`), evitiamo
        il comma-join che rende ambigui i predicati di filtro.
        dist=3 è SPLIT in due branch UNION-ALL-equivalenti (qui in due
        chiamate Python separate, in `_reachable_pairs` come UNION ALL):
        la formulazione con OR sulla direzione dell'edge
            (e.subject = a.neighbor AND e.object = b.neighbor)
            OR (e.object = a.neighbor AND e.subject = b.neighbor)
        forzava il planner a un cross-product n1×n1×edges = trilioni di
        righe (verificato empiricamente con hang >5h).

        TIMING (su `M1 small subset`, 1 coppia, RAM warm)
        --------------------------------------------------
        dist=1   ~1-5 ms   (point lookup su indice n1.qid)
        dist=2   ~50-200 ms (hash join su n1 self-join, filtrato sui qid)
        dist=3a  ~300-2000 ms (n1 ⋈ edges ⋈ n1)
        dist=3b  ~300-2000 ms (idem, direzione opposta)
        Totale worst case (miss su tutte): ~1-5 s.
        """
        # Risolve None/0 → sentinel ∞ così il SQL `<=` rimane uniforme
        # senza dover branch-are la query per "nessun threshold".
        t = self._resolve_threshold(threshold)

        # ============================================================
        # dist=1: edge diretto q-d (nessun filtro sugli endpoint)
        # ============================================================
        # Query: cerca una riga in n1 con (qid=q, neighbor=d).
        # Costruita N1 via UNION delle due direzioni (build_n1.py), quindi
        # se esiste un edge q→d O d→q, n1 contiene la riga (q, d) e/o (d, q).
        # Test simmetrico — basta cercare (q, d).
        # SHAPE OUTPUT: 1 riga `(1,)` se esiste l'edge, None altrimenti.
        # `LIMIT 1` permette short-circuit non appena trovata la prima.
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
        # Self-join di n1 su `neighbor`: una riga (a.qid=q, a.neighbor=x)
        # AND (b.qid=d, b.neighbor=x) → x è vicino di entrambi.
        # PERCHÉ FILTRARE SOLO `a.neighbor_degree`: nel join `a.neighbor =
        # b.neighbor`, le due colonne contengono lo STESSO nodo x; il loro
        # grado è identico (è proprietà dell'entità, non dell'edge), quindi
        # un solo controllo basta.
        # SHAPE OUTPUT: 1 riga `(1,)` se ∃ bridge non-hub, None altrimenti.
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
        # dist=3 direzione 1: edge a.neighbor → b.neighbor (subj→obj)
        # ============================================================
        # Cerca: q — x — y — d con edge x→y in `edges` originale.
        # Schema concettuale del cammino:
        #     q ──(N1)──> x ──(edges, dir.1)──> y <──(N1)── d
        # JOIN flow:
        #   n1 a       → righe (a.qid=q, a.neighbor=x), x ∈ N1(q)
        #   ⋈ edges e  → e.subject=x, e.object=y (edge x→y)
        #   ⋈ n1 b     → b.qid=d, b.neighbor=y (y ∈ N1(d))
        # FILTRI:
        #   a.qid = q, b.qid = d            → ancorano agli endpoint
        #   a.neighbor_degree <= t          → x è bridge non-hub
        #   b.neighbor_degree <= t          → y è bridge non-hub
        # SHAPE OUTPUT: 1 riga `(1,)` se ∃ cammino dir.1, None altrimenti.
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
        # dist=3 direzione 2: edge b.neighbor → a.neighbor (obj→subj)
        # ============================================================
        # Identica a dir.1 ma con l'edge in direzione opposta:
        #     q ──(N1)──> x <──(edges, dir.2)── y <──(N1)── d
        # Le due query sono complementari: insieme coprono entrambe le
        # direzioni di edge x↔y. Su grafo non orientato (Wikidata è
        # logicamente non-orientato, le direzioni edge sono solo come
        # vengono memorizzate i triple SPO) la coppia dir1+dir2 è
        # esaustiva — non perde alcun cammino dist=3.
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
        # Ultima fase: il valore di `r` è il risultato finale.
        # `r is not None` → True se HIT su dir2, False se miss su tutte e 4.
        return r is not None

    # ------------------------------------------------------------------------
    # Core batch helper: compute reached endpoints
    # ------------------------------------------------------------------------

    def _reachable_pairs(
        self, Q: list[str], D: list[str], threshold: int | None,
    ) -> tuple[set[str], set[str]]:
        """Return (Q_reached, D_reached) sets.

        SCOPO
        -----
        Cuore del motore di scoring batch. Calcola in UN solo SQL TUTTE
        le coppie (q, d) raggiungibili (dist ≤ 3, filtrato a threshold)
        per Q × D, poi ne estrae:
            Q_reached = {q ∈ Q : ∃ d ∈ D with dist(q,d) ≤ 3 (filtered)}
            D_reached = {d ∈ D : ∃ q ∈ Q with dist(d,q) ≤ 3 (filtered)}

        Entrambi gli insiemi vengono dal MEDESIMO query result —
        `connected_ratio` e `purity_ratio` condividono il lavoro:
        una sola hash-build, un solo scan di edges.

        INPUT
        -----
        Q : list[str]   QID degli "endpoint query" (tipicamente 1-5 elementi
                        per query NQ; le `question_qids` post-curation).
        D : list[str]   QID degli "endpoint passage" (~30-100 elementi per
                        passaggio Wikipedia post-ReFiNed).
        threshold : int | None
                        Soglia di grado; None/0 → ∞.

        OUTPUT
        ------
        (Q_reached, D_reached) : tuple[set[str], set[str]]
            Q_reached ⊆ Q
            D_reached ⊆ D
            (q, d) ∈ rows  ⟹  q ∈ Q_reached  AND  d ∈ D_reached
            (ma non viceversa: un q può comparire con più d e contribuire
            una sola volta a Q_reached perché è un insieme.)

        IMPLEMENTAZIONE
        ---------------
        1. **Materializza Q e D come Arrow tables** e li registra in DuckDB
           con `db.register("Q_set", ...)` — diventano "view" virtuali
           consultabili dal SQL. Non si copia su disco, solo riferimenti.
        2. Esegue UNA query con CTE (`WITH ... SELECT ...`) composta da
           4 sotto-query: d1, d2, d3a, d3b — una per ciascuna distanza
           (dist=3 splittata per direzione di edge).
        3. `UNION ALL` concatena (no de-dup intermedio, più veloce);
           `SELECT DISTINCT q, d FROM (...)` deduplica solo alla fine.
        4. **Cleanup garantito** (try/finally) per disregistrare le view —
           altrimenti chiamate successive con lo stesso nome di tabella
           fallirebbero o useranno dati vecchi.
        5. **Proiezione finale**: dalle righe `(q, d)` si estraggono i due
           set con set-comprehension Python — operazione O(|rows|) su
           ~migliaia di righe al massimo.

        FORMA DELLA QUERY
        -----------------
        WITH
            d1   AS (...),    -- dist=1 hits
            d2   AS (...),    -- dist=2 hits
            d3a  AS (...),    -- dist=3 dir.1 hits
            d3b  AS (...)     -- dist=3 dir.2 hits
        SELECT DISTINCT q, d FROM (
            d1 UNION ALL d2 UNION ALL d3a UNION ALL d3b
        )

        Le 4 CTE sono indipendenti: il planner DuckDB le esegue in
        parallelo (vectorized executor). Per ognuna costruisce hash join
        chains pulite — niente OR, niente cross-product.

        DETTAGLIO PER OGNI CTE
        ----------------------
        d1 (dist=1): n1 ⋈ Q_set ⋈ D_set
            - Punto d'ingresso: indice idx_n1_qid filtra n.qid ∈ Q
            - Output: tutte le coppie (q, d) con edge diretto
            - NESSUN filtro su threshold (gli endpoint non si filtrano mai)

        d2 (dist=2): n1 a ⋈ n1 b ⋈ Q_set ⋈ D_set
            - Self-join su `a.neighbor = b.neighbor` = bridge condiviso x
            - Filtra `a.neighbor_degree <= ?` (un solo check, x = a.neighbor
              = b.neighbor → grado identico)
            - Output: coppie (q, d) con bridge non-hub

        d3a (dist=3, direzione subj→obj):
            n1 a ⋈ edges e (e.subject=a.neighbor) ⋈ n1 b (e.object=b.neighbor)
            ⋈ Q_set ⋈ D_set
            - Cammino: q —N1→ x —edge.subj→edge.obj— y ←N1— d
            - Due check di degree (su x = a.neighbor e y = b.neighbor),
              ENTRAMBI i bridge devono essere non-hub

        d3b (dist=3, direzione obj→subj):
            n1 a ⋈ edges e (e.object=a.neighbor) ⋈ n1 b (e.subject=b.neighbor)
            - Cammino: q —N1→ x ←edge.obj←edge.subj— y ←N1— d
            - Identico a d3a ma con l'edge in direzione opposta

        BINDING DEI PARAMETRI
        ---------------------
        La lista `[t, t, t, t, t]` corrisponde ai 5 placeholder `?`:
            posizione 1 → d2:   a.neighbor_degree <= ?
            posizione 2 → d3a:  a.neighbor_degree <= ?
            posizione 3 → d3a:  b.neighbor_degree <= ?
            posizione 4 → d3b:  a.neighbor_degree <= ?
            posizione 5 → d3b:  b.neighbor_degree <= ?
        Stesso valore di threshold passato a tutti.

        TIMING (su Q ~3 elementi, D ~30 elementi, RAM warm)
        ---------------------------------------------------
        Totale ~50-200 ms tipicamente, dominato dalle CTE d3a + d3b.
        """
        t = self._resolve_threshold(threshold)

        # --- Registra Q e D come tabelle virtuali nella connessione DuckDB ---
        # `pa.table({"qid": Q})` crea un Arrow Table zero-copy dalla list di
        # stringhe. `db.register(name, arrow_table)` lo rende interrogabile
        # come una vera tabella SQL via il nome `Q_set` / `D_set`.
        # Dimensione tipica: Q ~1-5 righe, D ~30-100 righe — micro overhead.
        Q_table = pa.table({"qid": Q})
        D_table = pa.table({"qid": D})
        self.db.register("Q_set", Q_table)
        self.db.register("D_set", D_table)

        try:
            # All JOINs explicit. dist=3 split into two UNION ALL branches
            # (one per edge direction) — the OR formulation forces a
            # cross-product blow-up in the DuckDB query planner.
            rows = self.db.execute(
                """
                WITH
                    -- dist=1: q-d direct edge (no threshold on endpoints)
                    d1 AS (
                        SELECT n.qid AS q, n.neighbor AS d
                        FROM n1 n
                        INNER JOIN Q_set ON n.qid = Q_set.qid
                        INNER JOIN D_set ON n.neighbor = D_set.qid
                    ),
                    -- dist=2: shared non-hub neighbor
                    d2 AS (
                        SELECT a.qid AS q, b.qid AS d
                        FROM n1 a
                        INNER JOIN n1 b ON a.neighbor = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                    ),
                    -- dist=3 direction 1: x → y edge
                    d3a AS (
                        SELECT a.qid AS q, b.qid AS d
                        FROM n1 a
                        INNER JOIN edges e ON e.subject = a.neighbor
                        INNER JOIN n1 b ON e.object = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                          AND b.neighbor_degree <= ?
                    ),
                    -- dist=3 direction 2: y → x edge
                    d3b AS (
                        SELECT a.qid AS q, b.qid AS d
                        FROM n1 a
                        INNER JOIN edges e ON e.object = a.neighbor
                        INNER JOIN n1 b ON e.subject = b.neighbor
                        INNER JOIN Q_set ON a.qid = Q_set.qid
                        INNER JOIN D_set ON b.qid = D_set.qid
                        WHERE a.neighbor_degree <= ?
                          AND b.neighbor_degree <= ?
                    )
                SELECT DISTINCT q, d FROM (
                    SELECT * FROM d1
                    UNION ALL SELECT * FROM d2
                    UNION ALL SELECT * FROM d3a
                    UNION ALL SELECT * FROM d3b
                )
                """,
                [t, t, t, t, t],
            ).fetchall()
        finally:
            # Cleanup: rimuove le view virtuali. Importante perché:
            # 1. Una chiamata successiva con `register("Q_set", ...)` di
            #    DuckDB ricrea silenziosamente, ma è bene non lasciare
            #    riferimenti pendenti agli Arrow buffer (CPython GC).
            # 2. Se la query ha throw-ato, il finally garantisce comunque
            #    la rimozione (no leak di view pendenti su errori).
            self.db.unregister("Q_set")
            self.db.unregister("D_set")

        # --- Proiezione finale: estrai gli endpoint reached ---
        # rows è una list[tuple[str, str]] di (q, d) DISTINCT.
        # Set-comprehension: O(|rows|), tipicamente ≤ migliaia.
        # Esempio:
        #   rows = [("Q1", "Q10"), ("Q1", "Q20"), ("Q2", "Q10")]
        #   Q_reached = {"Q1", "Q2"}     (q1 raggiunge sia d10 che d20)
        #   D_reached = {"Q10", "Q20"}   (d10 raggiunto da q1 e q2)
        Q_reached = {q for q, _ in rows}
        D_reached = {d for _, d in rows}
        return Q_reached, D_reached

    # ------------------------------------------------------------------------
    # Public single-threshold APIs
    # ------------------------------------------------------------------------

    def connected_ratio(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """Frazione delle entità di Q raggiungibili ad ALMENO una di D entro 3 hop.

        FORMULA
        -------
        connected_ratio(Q, D, t) = |{q ∈ Q : ∃ d ∈ D with dist(q, d) ≤ 3}| / |Q|

        SEMANTICA
        ---------
        "Quanto della query trova *qualche* aggancio nel passage?".
        Vicina al recall lato-query: alta se la query è ben coperta
        dal passage, bassa se i suoi entity-mention restano isolati.

        INPUT / OUTPUT
        --------------
        Q : list[str]   QID question_qids della query (post-curation)
        D : list[str]   QID entità del passage (post-ReFiNed + curation)
        threshold : int|None  soglia bridge non-hub (None/0 = ∞)

        Returns:
            float ∈ [0.0, 1.0]
                0.0 se Q vuoto o nessun q raggiunge alcun d
                1.0 se TUTTI i q ∈ Q raggiungono almeno un d ∈ D

        EDGE CASE
        ---------
        Se Q è vuota → 0.0 (early return, niente SQL).
        Se D è vuota → 0.0 (Q_reached è ∅, len(Q_reached)/len(Q) = 0).

        ESEMPIO
        -------
        Q = ["Q937"]                  (Albert Einstein)
        D = ["Q183", "Q142", "Q31"]   (Germany, France, Belgium)
        connected_ratio(Q, D, 5000) → 1.0  se Einstein ha cammino
                                          ≤3 verso almeno uno dei tre.
        """
        if not Q:
            return 0.0
        # Riusa il batch helper: serve solo Q_reached, scarta D_reached.
        # Costo identico a `purity_ratio` o `kg_score` — la query SQL è
        # la stessa, cambia solo cosa estraiamo dal risultato.
        Q_reached, _ = self._reachable_pairs(Q, D, threshold)
        return len(Q_reached) / len(Q)

    def purity_ratio(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """Frazione delle entità di D raggiungibili ad ALMENO una di Q entro 3 hop.

        FORMULA
        -------
        purity_ratio(Q, D, t) = |{d ∈ D : ∃ q ∈ Q with dist(d, q) ≤ 3}| / |D|

        SEMANTICA
        ---------
        "Quanto del passage è 'pertinente' alla query?".
        Vicina alla precisione lato-passage: alta se le entità del
        passage sono tutte agganciate alla query (passage focalizzato),
        bassa se il passage è un mix di topic dove la query è solo
        una piccola parte.

        Per la simmetria del grafo non-orientato:
            d ∈ N3(q)  ⟺  q ∈ N3(d)
        quindi `dist(d, q) ≤ 3` equivale a `dist(q, d) ≤ 3` — il calcolo
        è batch UNICO con `connected_ratio`, vedi `_reachable_pairs`.

        INPUT / OUTPUT
        --------------
        Q, D, threshold : come `connected_ratio`.

        Returns:
            float ∈ [0.0, 1.0]
                0.0 se D vuoto o nessun d ha cammino verso Q
                1.0 se TUTTI i d ∈ D raggiungono almeno un q ∈ Q

        ESEMPIO
        -------
        Q = ["Q937"]
        D = ["Q183", "Q142", "Q31"]
        purity_ratio = 0.667  → 2 dei 3 entity di D hanno cammino ≤3
                                verso Einstein, 1 no (passage parzial-
                                mente focalizzato sulla query).
        """
        if not D:
            return 0.0
        # Stessa query batch di connected_ratio. Idealmente i due
        # ratio andrebbero chiamati via `kg_components` per evitare
        # di rifare il SQL — qui mantengono semantica indipendente.
        _, D_reached = self._reachable_pairs(Q, D, threshold)
        return len(D_reached) / len(D)

    def kg_score(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> float:
        """KG-score scalare = connected_ratio · purity_ratio alla threshold data.

        FORMULA
        -------
        kg_score(Q, D, t) = connected_ratio(Q, D, t) · purity_ratio(Q, D, t)

        SEMANTICA (analoga F1-style su recall × precision)
        --------------------------------------------------
        Punteggio singolo che combina:
        - copertura della query (connected_ratio): "ho agganciato la query?"
        - purezza del passage (purity_ratio): "il passage è on-topic?"
        Ottimo solo se ENTRAMBI sono alti — un passage che cita la
        query ma è altrimenti irrilevante penalizza purity_ratio.

        INPUT / OUTPUT
        --------------
        Q, D, threshold : standard.

        Returns:
            float ∈ [0.0, 1.0]
                0.0 se Q vuoto, D vuoto, o nessuna coppia raggiungibile
                1.0 se TUTTI i q ∈ Q E TUTTI i d ∈ D sono mutuamente connessi

        ESEMPIO
        -------
        cr = 1.0, pr = 0.667 → kg_score = 0.667
        cr = 0.5, pr = 0.5  → kg_score = 0.25
        """
        if not Q or not D:
            return 0.0
        # Una sola call a _reachable_pairs (un solo SQL): poi due len().
        Q_reached, D_reached = self._reachable_pairs(Q, D, threshold)
        cr = len(Q_reached) / len(Q)
        pr = len(D_reached) / len(D)
        return cr * pr

    def kg_components(
        self, Q: list[str], D: list[str], threshold: int | None = 5000,
    ) -> dict[str, float]:
        """Calcola e restituisce in UN solo batch SQL connected_ratio,
        purity_ratio E kg_score — efficiente per il reranker.

        SCOPO
        -----
        API consigliata quando servono tutte e tre le metriche (es.
        debug, analisi, log per query): evita di chiamare 3 volte
        `_reachable_pairs` riusando l'unico result set.

        INPUT
        -----
        Q : list[str]
        D : list[str]
        threshold : int | None    default 5000

        OUTPUT
        ------
        dict[str, float] con chiavi:
            "connected_ratio"  ∈ [0.0, 1.0]
            "purity_ratio"     ∈ [0.0, 1.0]
            "kg_score"         ∈ [0.0, 1.0]   (= connected · purity)

        Edge case Q vuoto / D vuoto → tutti e tre = 0.0.

        ESEMPIO DI RITORNO
        ------------------
            {"connected_ratio": 1.0,
             "purity_ratio":    0.667,
             "kg_score":        0.667}

        TIMING
        ------
        ~50-200 ms (un solo `_reachable_pairs`), confronto:
        - 3 chiamate separate (cr, pr, score): ~150-600 ms
        - questa API: ~50-200 ms (3x speed-up).
        """
        if not Q or not D:
            return {"connected_ratio": 0.0, "purity_ratio": 0.0, "kg_score": 0.0}
        Q_reached, D_reached = self._reachable_pairs(Q, D, threshold)
        cr = len(Q_reached) / len(Q)
        pr = len(D_reached) / len(D)
        # Stesso oggetto restituito è subito serializzabile a JSON
        # (utile per log per-query in cui salviamo lo score breakdown).
        return {"connected_ratio": cr, "purity_ratio": pr, "kg_score": cr * pr}

    # ------------------------------------------------------------------------
    # Multi-threshold API (ablation)
    # ------------------------------------------------------------------------

    def kg_score_multi(
        self,
        Q: list[str],
        D: list[str],
        thresholds: Iterable[int | None] = DEFAULT_THRESHOLDS,
    ) -> dict[int | None, float]:
        """Esegue kg_score a multiple soglie — utile per ablation studies.

        SCOPO
        -----
        Per ogni threshold in input, calcola lo `kg_score` corrispondente.
        Permette di osservare come varia il punteggio al variare del filtro
        sui nodi-bridge: più la threshold è bassa, più il grafo diventa
        sparso (filtra hub) e più il punteggio cala — comportamento
        monotonico atteso.

        INPUT
        -----
        Q : list[str]
        D : list[str]
        thresholds : Iterable[int | None]
            Default: DEFAULT_THRESHOLDS = (500, 1000, 2000, 5000, 10000, 0)
            Convenzione: 0 (o None) = ∞ (nessun filtro su bridge).

        OUTPUT
        ------
        dict[int | None, float] : mappa threshold → kg_score
            Le chiavi sono gli stessi valori passati in `thresholds`
            (preserva la chiave originale, anche None se passato).

        ESEMPIO DI RITORNO
        ------------------
            {500:    0.154,
             1000:   0.308,
             2000:   0.385,
             5000:   0.385,
             10000:  0.500,
             0:      0.615}     # 0 = ∞ → nessun filtro

        IMPLEMENTAZIONE
        ---------------
        Loop Python su `kg_score`: ogni threshold lancia una `_reachable_pairs`
        indipendente (un SQL ciascuna). Costo ≈ N_thresholds × singolo
        kg_score ≈ 6 × ~50ms = ~300ms su Q×D piccoli.

        ALTERNATIVA SCARTATA (fused SQL)
        --------------------------------
        Si potrebbe calcolare in UN solo SQL il `min_threshold_required`
        per ciascuna coppia (q, d) raggiungibile, poi raggruppare per
        threshold per contare i reached. Più efficiente in teoria, ma:
        - Overhead aggiuntivo del CASE/MIN aggregation per coppia
        - Complessità del SQL aumenta significativamente (debug più duro)
        - Beneficio marginale per |thresholds| = 6 (stessa query, ~6x meno scan)
        Decisione (vedi conversazione 2026-04-29): NON fare il fuse,
        loop Python è sufficiente.

        USO TIPICO
        ----------
            # Ablation per una (Q, D) singola
            multi = scorer.kg_score_multi(Q, D)
            print({t: f"{s:.3f}" for t, s in multi.items()})

            # Ablation across full eval set
            results = []
            for Q, D in pairs:
                results.append(scorer.kg_score_multi(Q, D))
            df = pd.DataFrame(results)  # colonne = thresholds
        """
        # Dict-comprehension: chiama kg_score N volte e raccoglie i risultati.
        # Le chiavi del dict di output sono i threshold ORIGINALI passati
        # (non risolti via _resolve_threshold) — preserva l'identità per
        # report/log: l'utente che ha passato 0 vede `0` nella chiave,
        # non `9223372036854775807`.
        return {t: self.kg_score(Q, D, t) for t in thresholds}


# ============================================================================
# Smoke test — load + basic API checks on real data
# ============================================================================

def _smoke_test() -> None:
    """Sanity check minimo end-to-end della classe KGScorer.

    SCOPO
    -----
    Verifica sintattica/runtime: dopo modifiche a kg.py o a n1.parquet,
    eseguire `python scripts/kg.py` deve completare in pochi secondi
    e stampare numeri sensati. NON è una validazione semantica della
    metrica — per quella vedere `ablation_diagnostic.py` o lo scoring
    completo nel notebook reranker.

    COSA FA
    -------
    1. Prende la PRIMA query da queries_curated.jsonl (non scelta in modo
       intelligente — solo un sample sintattico).
    2. Prende il PRIMO passage da passage_entities_curated.parquet (non
       è il top-100 di quella query: probabilmente non c'è alcuna
       relazione semantica forte).
    3. Inizializza KGScorer (carica n1 + edges, ~15s).
    4. Esegue `is_reachable` sulla prima coppia (q0, d0) con verbose ON,
       per vedere su quale dist fa hit/miss.
    5. Esegue `kg_components` su Q × D con threshold default (5000).
    6. Esegue `kg_score_multi` su tutte le DEFAULT_THRESHOLDS per
       osservare la monotonia (più la threshold cresce, più il punteggio
       NON diminuisce — proprietà del filtro).

    OUTPUT ATTESO (RAM warm)
    ------------------------
    Init total:                            ~15 s
    is_reachable:                          1-3 s
    kg_components(t=5000):                 50-200 ms
    kg_score_multi(6 soglie):              ~300-500 ms totali
    """
    import pyarrow.parquet as pq

    queries_path = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
    passages_path = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

    print("=" * 70)
    print("KGScorer smoke test")
    print("=" * 70)

    # --- Fase 1: carica prima riga da queries_curated.jsonl ---
    # `next(f)` legge solo il primo line del JSONL (formato Layer 2):
    # ogni riga è un dict con question, question_qids, answer_variant_qids, ...
    # Estraiamo `question_qids` (può essere [] su query senza entity-link
    # ReFiNed: short-circuit con messaggio).
    with queries_path.open("r", encoding="utf-8") as f:
        q_obj = json.loads(next(f))
    Q = q_obj.get("question_qids") or []
    print(f"\nQuery: {q_obj.get('question', '<no text>')[:80]}")
    print(f"  question_qids: {Q}")
    if not Q:
        print("  ! query has no question_qids — smoke test will short-circuit")
        return

    # --- Fase 2: carica primo passage da passage_entities_curated.parquet ---
    # `slice(0, 1)` prende solo la prima riga (passage). La colonna `qids`
    # è list<string> in Arrow → `.as_py()` la converte in list[str] Python.
    # NB: questo passage NON è correlato semanticamente alla query — è solo
    # un sample per testare che le API funzionino su input reale.
    pass_table = pq.read_table(passages_path).slice(0, 1)
    D = pass_table.column("qids")[0].as_py()
    print(f"  first passage entities ({len(D)}): {D[:8]}{'...' if len(D) > 8 else ''}")
    if not D:
        print("  ! passage has no entities — smoke test will short-circuit")
        return

    # --- Fase 3: init dello scorer (parte più lenta del test) ---
    # Misuriamo l'init separatamente — ~15s tipicamente, dominato dal
    # caricamento di edges.parquet (~661M righe → 5-10 GB RAM).
    print()
    t0 = time.perf_counter()
    scorer = KGScorer()
    print(f"Init total: {time.perf_counter() - t0:.1f}s")

    # --- Fase 4: is_reachable diagnostico sulla PRIMA coppia (q0, d0) ---
    # Verbose ON per vedere su quale phase di distance fa HIT o miss
    # (utile per debug se una coppia stalla o ritorna risultato inatteso).
    q0, d0 = Q[0], D[0]
    print(f"\nis_reachable({q0}, {d0}, t=5000):")
    t0 = time.perf_counter()
    is_r = scorer.is_reachable(q0, d0, threshold=5000, verbose=True)
    print(f"  → {is_r}  (total {(time.perf_counter()-t0)*1000:.1f} ms)")

    # --- Fase 5: kg_components batch su Q × D, threshold=5000 ---
    # Calcola in UN solo batch SQL i 3 valori (cr, pr, score). Mostra
    # anche il timing in ms — riferimento per stimare il costo del
    # reranker su tutti i (query, top-100 passages).
    t0 = time.perf_counter()
    components = scorer.kg_components(Q, D, threshold=5000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\nkg_components(Q, D, t=5000)  ({elapsed_ms:.1f} ms):")
    for k, v in components.items():
        print(f"  {k:>16}: {v:.4f}")

    # --- Fase 6: ablation multi-threshold ---
    # Stampa kg_score per ciascun threshold in DEFAULT_THRESHOLDS.
    # Property check informale: la sequenza di score deve essere
    # MONOTONA NON DECRESCENTE rispetto al threshold (a soglie più
    # alte permettiamo più bridge → più cammini → score >= score_inferiore).
    # Se vedi una violazione, c'è un bug nella query SQL.
    t0 = time.perf_counter()
    multi = scorer.kg_score_multi(Q, D)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\nkg_score_multi(Q, D)  ({elapsed_ms:.1f} ms total):")
    for t_val, score in multi.items():
        # Etichetta human-readable: 0/None → "inf", altri → str(int).
        t_label = "inf" if t_val in (None, 0) else str(t_val)
        print(f"  t={t_label:>6}: {score:.4f}")

    print("\nSmoke test OK.")


if __name__ == "__main__":
    _smoke_test()