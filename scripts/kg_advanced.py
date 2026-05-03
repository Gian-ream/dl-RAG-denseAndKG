"""Layer 4 (advanced) — KG runtime scorer with min-distance annotation and disk persistence.

OBIETTIVO
=========
Variante di `scripts/kg.py` ottimizzata per griglie di ablation (distance × threshold).
Rispetto alla classe base `KGScorer`:

1. **Una sola query SQL per threshold** (non 3): la query annota per ogni
   coppia (q, d) raggiungibile la **distanza minima** a cui è stata trovata,
   poi in Python si deriva il risultato per ciascun `max_distance ∈ {1, 2, 3}`
   filtrando `min_dist ≤ k`. Niente lavoro SQL ripetuto tra distance.
2. **Persistenza su disco** (`data/kg.duckdb`): la prima init builda le
   tabelle (~5 min, una volta sola), le successive aprono il file già
   pronto in pochi secondi. Multiprocessing-ready: i worker aprono il
   file in `read_only=True`, condividono il page cache OS, niente RAM
   duplicata tra processi.
3. **API griglia che ritorna `pd.DataFrame`**: pronto per analisi/plot.

PARAMETRI ESPLICITI
===================
A differenza della classe base (`max_distance` implicito = 3), qui
`distances` e `thresholds` sono SEMPRE espliciti nelle API pubbliche.
Niente default dietro le quinte.

ARCHITETTURA SQL — query unificata
==================================
Le 4 CTE (d1, d2, d3a, d3b) restano identiche a quelle di `kg.py`, ma
ognuna proietta in più una colonna letterale `dist` (1, 2, 3, 3).
L'aggregato finale `MIN(dist) GROUP BY q, d` collassa cammini multipli
sulla stessa coppia restituendo la distanza minima.

    WITH d1 AS (..., 1 AS dist),
         d2 AS (..., 2 AS dist),
         d3a AS (..., 3 AS dist),
         d3b AS (..., 3 AS dist)
    SELECT q, d, MIN(dist) AS min_dist
    FROM (d1 UNION ALL d2 UNION ALL d3a UNION ALL d3b)
    GROUP BY q, d

Costo per coppia (Q, D) e griglia 3×6:
- Approccio loop classico: 18 query SQL (3 dist × 6 threshold)
- Approccio min_dist (questo file): 6 query SQL (1 per threshold)
- Riduzione ~3x sull'overhead di parsing/planning DuckDB.

NOTA IMPORTANTE: la threshold modifica il filtro `WHERE neighbor_degree <= ?`,
quindi non si può fattorizzare ulteriormente in una sola query SQL senza
materializzare TUTTI i cammini (cardinalità intermedia esplosiva).
6 query / threshold sono il sweet spot.

PERSISTENZA SU DISCO
====================
Default: `data/kg.duckdb` viene creato/riusato automaticamente.
- Prima init in modalità RW: builda tabelle + indice, committa al file
  (~5 min, ~10-15 GB su disco).
- Init successive RW: file già contiene n1+edges → skip rebuild (~1s).
- Init RO (worker MP): apre il file in read_only, errore se non esiste.

Per forzare il vecchio comportamento in-memory: passare `db_path=None`.

USAGE
=====
    from scripts.kg_advanced import KGScorerAdvanced

    # Prima volta (RW, builda il file):
    scorer = KGScorerAdvanced()
    df = scorer.kg_components_grid(Q, D)
    # → pd.DataFrame con colonne [distance, threshold, connected_ratio,
    #                              purity_ratio, kg_score]

    # Worker MP (file già pronto, RO):
    scorer = KGScorerAdvanced(read_only=True)

    # Batch su lista di (query_id, Q, passage_id, D):
    df = scorer.kg_components_grid_batch(pairs)
    # → DataFrame con anche colonne query_id, passage_id

Smoke test:
    .venv\\Scripts\\python.exe scripts\\kg_advanced.py
"""

import json
import time
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
import pyarrow as pa

from scripts.kg import KGScorer, REPO_ROOT, N1_PATH, EDGES_PATH


# Default disk path for the persistent DuckDB file. Lives under data/ so
# it's automatically gitignored alongside the rest of the dataset.
KG_DUCKDB_PATH = REPO_ROOT / "data" / "kg.duckdb"


# ============================================================================
# KGScorerAdvanced — extends KGScorer with min-dist query and disk persistence
# ============================================================================

class KGScorerAdvanced(KGScorer):
    """KG scorer con persistenza su disco e griglia (distance × threshold).

    Eredita da `KGScorer` per riusare:
        - `_resolve_threshold` (sentinel per ∞)
        - `is_reachable` (diagnostico, batch=False)
        - `connected_ratio` / `purity_ratio` / `kg_score` / `kg_components`
          (API single-threshold, single-distance — restano disponibili per
          chi vuole solo quelle senza pagare il dataframe overhead)

    Sovrascrive:
        - `__init__`: aggiunge persistenza disco + modalità read_only

    Aggiunge:
        - `_reachable_pairs_min_dist`: query unificata che restituisce
          (q, d) → min_dist invece di solo (Q_reached, D_reached)
        - `kg_components_grid`: griglia per UNA coppia (Q, D), DataFrame output
        - `kg_components_grid_batch`: wrapper su lista di coppie
    """

    # Default distance set per la griglia. Cumulativo: dist=k include i
    # cammini di lunghezza ≤ k. Tipicamente vogliamo tutti e tre per
    # vedere come il punteggio cresce all'aumentare del raggio.
    DEFAULT_DISTANCES: tuple[int, ...] = (1, 2, 3)

    def __init__(
        self,
        db_path: Path | None = KG_DUCKDB_PATH,
        n1_path: Path = N1_PATH,
        edges_path: Path = EDGES_PATH,
        read_only: bool = False,
        verbose: bool = True,
    ) -> None:
        """Inizializza con persistenza su disco (default) o in-memory (fallback).

        FASI DI INIZIALIZZAZIONE
        ------------------------
        Tre modalità mutuamente esclusive a seconda dei parametri:

        1) **In-memory** (`db_path=None`): delega al `__init__` della classe
           base. Comportamento identico a `KGScorer`. Da usare solo per
           test/debug isolati — niente persistenza, niente MP.

        2) **Read-only** (`read_only=True`): apre `db_path` in RO, verifica
           che contenga le tabelle `n1` e `edges`. Errore se file mancante
           o tabelle assenti. Modalità tipica per i worker MP: il main ha
           già buildato il file, i worker lo aprono e basta.

        3) **Read-write con auto-build** (default): apre `db_path` (lo crea
           se non esiste), e per ciascuna delle tabelle `n1`/`edges`:
           - se esiste già nel file → skip (init istantaneo)
           - se manca → builda dal parquet, crea l'indice (per n1), committa
           Idempotente: chiamate successive sullo stesso `db_path` sono
           istantanee, non rileggono i parquet.

        DETECTION DELLE TABELLE
        -----------------------
        Usiamo `information_schema.tables` (standard SQL, supportato da
        DuckDB) per controllare cosa esiste. Più robusto di un `try/except`
        su `SELECT * FROM n1 LIMIT 0` perché non genera log spurii.

        EFFETTI COLLATERALI
        -------------------
        Dopo `__init__`:
            self.db        # duckdb.DuckDBPyConnection (file-backed o :memory:)
            self.db.tables = {n1, edges} (verificato/buildato)

        RAM peak in modalità file-backed: molto inferiore a in-memory perché
        DuckDB usa mmap (~1-3 GB working set vs ~10-15 GB in-memory).
        Le query potranno essere leggermente più lente (~10-30%) per il
        primo accesso a una pagina non in page cache, poi a regime sono
        comparabili.

        PERCHÉ NON SUPER().__INIT__() PER IL CASO 3
        -------------------------------------------
        Il super costruisce sempre tutto in-memory con `:memory:`, senza
        considerare detection di tabelle esistenti. Riusarlo richiederebbe
        di passare un path al super e modificarlo a sua volta — preferiamo
        duplicare la logica di build qui, che è breve e self-contained.
        """
        # --- Modalità in-memory: delega al super ---
        # Caso d'uso: debug locale, smoke test isolati, ambienti senza
        # spazio disco. Comportamento 100% backward compatible col vecchio.
        if db_path is None:
            super().__init__(n1_path=n1_path, edges_path=edges_path, verbose=verbose)
            return

        # --- Modalità read_only (worker MP): apri ed esci ---
        # Il file deve esistere ed essere già stato buildato da qualcuno
        # (tipicamente il processo main prima di forkare i worker).
        if read_only:
            assert db_path.exists(), (
                f"db_path {db_path} non esiste — buildare prima con "
                f"KGScorerAdvanced(read_only=False)"
            )
            self.db = duckdb.connect(str(db_path), read_only=True)
            # Verifica integrità: deve contenere ENTRAMBE le tabelle.
            tables = {
                r[0] for r in self.db.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_name IN ('n1', 'edges')"
                ).fetchall()
            }
            assert tables == {"n1", "edges"}, (
                f"file {db_path} non contiene n1+edges (trovati: {tables}). "
                f"Rebuild necessario in RW mode."
            )
            if verbose:
                print(f"Opened {db_path} read-only (worker mode).", flush=True)
            return

        # --- Modalità default: RW con auto-build delle tabelle mancanti ---
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
        # Stessa logica del super KGScorer (CTAS + indice su qid), ma il
        # CREATE TABLE qui persiste sul file invece che in RAM.
        if "n1" not in existing:
            assert n1_path.exists(), (
                f"n1.parquet not found at {n1_path}\n"
                "Run scripts/build_n1.py first (Layer 3)."
            )
            if verbose:
                print(f"Building n1 in {db_path} (one-time, ~30s)...", flush=True)
            t0 = time.perf_counter()
            self.db.execute(
                f"CREATE TABLE n1 AS SELECT * FROM '{n1_path.as_posix()}'"
            )
            # Indice B-tree su qid: cruciale per `WHERE qid = ?` e i JOIN
            # di Q_set/D_set. Stesso motivo di kg.py.
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
        # Stessa proiezione di kg.py: solo subject + object, no predicate.
        # Niente indici (vedi nota nel docstring di KGScorer).
        if "edges" not in existing:
            assert edges_path.exists(), f"edges.parquet not found at {edges_path}"
            if verbose:
                print(f"Building edges in {db_path} (one-time, ~3-5 min)...", flush=True)
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

    # ------------------------------------------------------------------------
    # Core: unified min-distance query
    # ------------------------------------------------------------------------

    def _reachable_pairs_min_dist(
        self, Q: list[str], D: list[str], threshold: int | None,
    ) -> dict[tuple[str, str], int]:
        """UNA query SQL → mappa (q, d) → min_dist per ogni coppia raggiungibile.

        DIFFERENZA RISPETTO A `_reachable_pairs`
        -----------------------------------------
        La versione base ritorna `(Q_reached, D_reached)`: due insiemi
        agnostici alla distanza a cui ogni coppia è stata raggiunta.
        Qui ritorniamo invece una mappa esplicita `(q, d) → min_dist`
        che permette di derivare in Python il risultato per QUALUNQUE
        `max_distance ∈ {1, 2, 3}` filtrando `min_dist ≤ k`, senza
        rifare la query SQL.

        FORMA DELLA QUERY
        -----------------
        WITH d1  AS (..., 1 AS dist),    -- dist=1 (no threshold)
             d2  AS (..., 2 AS dist),    -- dist=2 (1 threshold check)
             d3a AS (..., 3 AS dist),    -- dist=3 dir.1 (2 threshold checks)
             d3b AS (..., 3 AS dist)     -- dist=3 dir.2 (2 threshold checks)
        SELECT q, d, MIN(dist) AS min_dist
        FROM (d1 UNION ALL d2 UNION ALL d3a UNION ALL d3b)
        GROUP BY q, d

        L'aggregato `MIN(dist) GROUP BY q, d` collassa cammini multipli
        sulla stessa coppia: se (q, d) è raggiungibile sia a dist=2 che
        a dist=3, la coppia compare con `min_dist=2`.

        BIND PARAMETERS
        ---------------
        5 placeholder `?` (identici a `_reachable_pairs`):
            posizione 1 → d2:   a.neighbor_degree <= ?
            posizioni 2-3 → d3a: due check su bridge
            posizioni 4-5 → d3b: due check su bridge
        Stesso valore di threshold passato a tutti.

        OUTPUT
        ------
        dict[tuple[str, str], int]
            Chiavi: tutte e sole le coppie (q, d) ∈ Q × D raggiungibili
                    entro 3 hop con threshold dato.
            Valori: min_dist ∈ {1, 2, 3}.

            Coppie NON raggiungibili: assenti dalla mappa (NON presenti
            con valore None). Filtrare con `mappa.get((q, d), None)`.

        TIMING
        ------
        ~50-200 ms per coppia (Q ~3, D ~30). Stesso ordine di grandezza
        di `_reachable_pairs` (la GROUP BY aggiunge overhead trascurabile
        rispetto al lavoro dei JOIN).
        """
        t = self._resolve_threshold(threshold)

        # Materializza Q e D come Arrow tables, registra come view virtuali.
        # Identico al pattern di _reachable_pairs.
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
                    -- dist=3 dir.1: edge x→y in `edges` originale
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
                    -- dist=3 dir.2: edge y→x (direzione opposta)
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
            # Cleanup garantito (anche su exception).
            self.db.unregister("Q_set")
            self.db.unregister("D_set")

        # Pack in dict: chiavi (q, d), valore int.
        return {(q, d): mind for q, d, mind in rows}

    # ------------------------------------------------------------------------
    # Grid API: single pair → DataFrame
    # ------------------------------------------------------------------------

    def kg_components_grid(
        self,
        Q: list[str],
        D: list[str],
        distances: Iterable[int] = DEFAULT_DISTANCES,
        thresholds: Iterable[int | None] = KGScorer.DEFAULT_THRESHOLDS,
    ) -> pd.DataFrame:
        """Griglia (distance × threshold) per UNA coppia (Q, D) → DataFrame.

        SCOPO
        -----
        API principale per ablation studies. Esegue UNA query SQL per
        ogni threshold (annota min_dist), poi in Python filtra per
        ciascuna `distance` desiderata. Niente lavoro SQL ripetuto.

        INPUT
        -----
        Q : list[str]   QID question_qids della query
        D : list[str]   QID entità del passage
        distances : Iterable[int]
            Sottoinsieme di {1, 2, 3}. Default: (1, 2, 3) — tutta la
            griglia. Cumulativo: `distance=k` significa cammini ≤ k hop.
        thresholds : Iterable[int | None]
            Default: KGScorer.DEFAULT_THRESHOLDS = (500, 1000, 2000, 5000,
            10000, 0). 0/None = ∞ (no filtro bridge).

        OUTPUT
        ------
        pd.DataFrame con colonne:
            distance         : int   (∈ distances)
            threshold        : int|None (chiave originale, 0/None per ∞)
            connected_ratio  : float ∈ [0, 1]
            purity_ratio     : float ∈ [0, 1]
            kg_score         : float ∈ [0, 1]   (= cr · pr)

        Numero di righe: `len(distances) × len(thresholds)`. Default: 18.

        EDGE CASE
        ---------
        Se Q o D è vuoto: ritorna DataFrame con tutte le righe della griglia
        e cr=pr=kg_score=0.0. Niente query SQL eseguita (early exit).

        NOTA INTERPRETATIVA — threshold inerte a distance=1
        ----------------------------------------------------
        A `distance=1` non ci sono nodi-bridge (l'edge è diretto), quindi
        il filtro threshold non ha effetto. Le righe `distance=1` di tutta
        la griglia avranno valori IDENTICI a parità di Q, D — non è un bug.
        A `distance=2` un solo bridge è soggetto a threshold; a `distance=3`
        due bridge → effetto threshold più pronunciato.

        ESEMPIO
        -------
        Q = ["Q937"]
        D = ["Q183", "Q142", "Q31"]
        df = scorer.kg_components_grid(Q, D)
        # df.head():
        #   distance  threshold  connected_ratio  purity_ratio  kg_score
        # 0        1        500              0.0           0.0       0.0
        # 1        2        500              1.0       0.33333    0.3333
        # 2        3        500              1.0       0.66667    0.6667
        # ...

        TIMING
        ------
        ~300-1200 ms su Q×D piccoli (6 query SQL × ~50-200 ms).
        Il loop Python finale è O(|coppie raggiungibili|) — trascurabile.
        """
        distances = list(distances)
        thresholds = list(thresholds)

        # --- Validazione input ---
        # Solo {1, 2, 3} sono supportate (la query ha 4 CTE max).
        for k in distances:
            assert k in (1, 2, 3), (
                f"distance {k} non valida — supportate solo (1, 2, 3)"
            )

        # --- Edge case: Q o D vuoto → DataFrame di zeri ---
        # Senza early exit faremmo registrate Arrow tables vuote e
        # query inutili. Ritorna direttamente tutte le righe a 0.
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

        # --- Loop esterno sulle threshold (1 query SQL ciascuna) ---
        # La threshold cambia il filtro `WHERE neighbor_degree <= ?` quindi
        # serve una query distinta per ognuna. Per CIASCUNA otteniamo
        # la mappa (q, d) → min_dist e poi deriviamo TUTTE le distance
        # in Python, gratis.
        for t in thresholds:
            pair_to_mindist = self._reachable_pairs_min_dist(Q, D, t)

            # --- Loop interno sulle distance (Python puro, no SQL) ---
            # Per ogni k ∈ distances, filtra le coppie con min_dist ≤ k
            # e ricostruisci Q_reached / D_reached.
            for k in distances:
                Q_reached = {
                    q for (q, _), md in pair_to_mindist.items() if md <= k
                }
                D_reached = {
                    d for (_, d), md in pair_to_mindist.items() if md <= k
                }
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
    # Grid API: batch over multiple (query, passage) pairs → DataFrame
    # ------------------------------------------------------------------------

    def kg_components_grid_batch(
        self,
        pairs: list[tuple[str, list[str], str, list[str]]],
        distances: Iterable[int] = DEFAULT_DISTANCES,
        thresholds: Iterable[int | None] = KGScorer.DEFAULT_THRESHOLDS,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Wrapper batch su lista di coppie → DataFrame "lungo" pronto per analisi.

        SCOPO
        -----
        Comodo quando hai centinaia/migliaia di coppie (query, passage)
        da scorare. Internamente loopa su `kg_components_grid` per ogni
        coppia e concatena i risultati, aggiungendo le colonne identifier.

        INPUT
        -----
        pairs : list[tuple[query_id, Q, passage_id, D]]
            Ogni elemento:
                query_id    : str — identifier della query (es. NQ id)
                Q           : list[str] — question_qids
                passage_id  : str — identifier del passage (es. wiki id)
                D           : list[str] — entity QIDs del passage
        distances, thresholds : come `kg_components_grid`.
        verbose : bool — stampa progress bar (1 riga ogni N coppie).

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

        Numero righe: `len(pairs) × len(distances) × len(thresholds)`.

        ESEMPIO USO
        -----------
            pairs = [
                ("nq_42", ["Q937"], "wiki_123", ["Q183", "Q142"]),
                ("nq_42", ["Q937"], "wiki_456", ["Q31"]),
                ...
            ]
            df = scorer.kg_components_grid_batch(pairs)

            # Pivot per analisi:
            df.pivot_table(
                index=["query_id", "passage_id"],
                columns=["distance", "threshold"],
                values="kg_score",
            )

        TIMING
        ------
        Lineare in `len(pairs)`. Per 1000 coppie × 18 celle griglia:
        ~5-20 minuti single-thread. Per workload reali considerare MP
        (vedi modalità `read_only=True` nel docstring di __init__).
        """
        pairs = list(pairs)
        sub_dfs: list[pd.DataFrame] = []

        # Flush periodico per non perdere tutto su un crash a metà loop.
        # NB: niente try/except qui — se una coppia fallisce l'errore
        # propaga, perché probabilmente indica un bug nei dati upstream.
        for idx, (query_id, Q, passage_id, D) in enumerate(pairs):
            sub = self.kg_components_grid(Q, D, distances, thresholds)
            # `insert(0, ...)` mette le colonne id in testa per leggibilità.
            sub.insert(0, "passage_id", passage_id)
            sub.insert(0, "query_id", query_id)
            sub_dfs.append(sub)

            if verbose and (idx + 1) % 50 == 0:
                print(f"  processed {idx+1}/{len(pairs)} pairs", flush=True)

        # Concat finale: ignore_index per avere un range index pulito.
        # Se sub_dfs è vuoto (pairs vuoto) ritorna DataFrame vuoto con
        # le colonne giuste (lo schema viene da kg_components_grid).
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
    """Sanity check end-to-end di KGScorerAdvanced.

    COSA FA
    -------
    1. Init in modalità default (RW, persiste in data/kg.duckdb). Se il
       file esiste già le tabelle non vengono ricostruite — init ~1s.
       Se non esiste, builda da n1.parquet + edges.parquet — ~5 min.
    2. Carica la prima query e il primo passage dai file curated.
    3. Chiama `_reachable_pairs_min_dist` — diagnostica della struttura output.
    4. Chiama `kg_components_grid` con default — stampa il DataFrame 3×6.
    5. Chiama `kg_components_grid_batch` con 2 coppie sintetiche — verifica
       lo schema "lungo" e la presenza di query_id/passage_id.
    6. (Opzionale, commentato) re-open in `read_only=True` per simulare
       un worker MP.

    OUTPUT ATTESO
    -------------
    Init (file esistente):                 ~1 s
    _reachable_pairs_min_dist:             ~50-200 ms
    kg_components_grid (3×6 = 18 righe):   ~300-1200 ms
    kg_components_grid_batch (2 coppie):   ~600-2400 ms
    """
    import pyarrow.parquet as pq

    queries_path = REPO_ROOT / "data" / "NQ_answer" / "queries_curated.jsonl"
    passages_path = REPO_ROOT / "data" / "NQ_answer" / "passage_entities_curated.parquet"

    print("=" * 70)
    print("KGScorerAdvanced smoke test")
    print("=" * 70)

    # --- Fase 1: init (persistito su disco) ---
    print()
    t0 = time.perf_counter()
    scorer = KGScorerAdvanced()
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

    # --- Fase 3: diagnostica min_dist a threshold default 5000 ---
    print(f"\n_reachable_pairs_min_dist(Q, D, t=5000):")
    t0 = time.perf_counter()
    mind = scorer._reachable_pairs_min_dist(Q, D, threshold=5000)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  {len(mind)} reachable pairs in {elapsed_ms:.1f} ms")
    # Mostra la distribuzione min_dist per intuizione visiva.
    if mind:
        dist_counts = {1: 0, 2: 0, 3: 0}
        for v in mind.values():
            dist_counts[v] += 1
        print(f"  distribution: dist=1: {dist_counts[1]}, "
              f"dist=2: {dist_counts[2]}, dist=3: {dist_counts[3]}")

    # --- Fase 4: griglia completa 3×6 ---
    t0 = time.perf_counter()
    df = scorer.kg_components_grid(Q, D)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"\nkg_components_grid(Q, D)  ({elapsed_ms:.1f} ms):")
    # Stampa il DataFrame intero (è solo 18 righe).
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False))

    # --- Fase 5: batch su 2 coppie sintetiche ---
    # Riusiamo la stessa Q ma due "passage" diversi (D vs D[:5] per simulare
    # un secondo passage più piccolo). Test sintattico, non semantico.
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