# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # HDT — smoke test e helper di lookup
#
# Test del dump HDT Wikidata (`latest-all-06-Jan-2022.hdt`, 166 GB).
# Verifichiamo che i lookup base funzionino, misuriamo i tempi reali, e
# costruiamo l'helper `neighbors_q` che useremo nel BFS-3-onde dello
# Step 3 (KG Subgraph Construction).
#
# **Prerequisiti** (da WSL2/Ubuntu):
# - `pyHDT` installato (vedi script di setup separato)
# - File HDT presente in `data/Wikidata_service/latest-all-06-Jan-2022.hdt` (relativo alla repo root)
#
# **Nota tecnica importante**: alla prima apertura pyHDT costruisce un
# index file `.hdt.index.v1-1` accanto al `.hdt`. Su 166 GB richiede
# ~30-60 min di CPU+I/O. Le aperture successive sono <1s.

# %% [markdown]
# ## 0. Configurazione e verifica file

# %%
import time
from pathlib import Path

# Resolve repo root by walking up until we hit pyproject.toml.
# Works both as a .py script (uses __file__) and from the paired .ipynb (falls back to cwd).
try:
    _start = Path(__file__).resolve().parent
except NameError:
    _start = Path.cwd().resolve()
REPO_ROOT = next(
    (p for p in [_start, *_start.parents] if (p / "pyproject.toml").is_file()),
    None,
)
assert REPO_ROOT is not None, f"Could not locate repo root (pyproject.toml) above {_start}"

HDT_PATH = REPO_ROOT / "data" / "Wikidata_service" / "latest-all-06-Jan-2022.hdt"

# Wikidata URI prefixes (full form, not shorthand — HDT stores full URIs)
WD_ENTITY = "http://www.wikidata.org/entity/"        # wd:Qxxx and wd:Pxxx live here
WDT_DIRECT = "http://www.wikidata.org/prop/direct/"  # wdt:Pxxx — the truthy/direct edges

# Sanity: file must exist and be ~166 GB. If smaller, the copy/move was likely incomplete.
assert HDT_PATH.exists(), f"HDT file not found at expected path: {HDT_PATH}"
size_gb = HDT_PATH.stat().st_size / (1024 ** 3)
print(f"HDT path:   {HDT_PATH}")
print(f"HDT size:   {size_gb:.1f} GB")
assert size_gb > 100, f"File too small ({size_gb:.1f} GB); download likely incomplete"


# %% [markdown]
# ## 1. Caricamento HDT
#
# Prima apertura: lenta (build dell'index, 30-60 min).
# Successive: istantanee.
#
# Mostriamo le statistiche di base del dataset per orientarci sulla scala:
# - `total_triples`: numero totale di triple nel dump (atteso: ~16-20 miliardi sul full Wikidata 2022)
# - `nb_subjects` / `nb_predicates` / `nb_objects`: cardinalità distinta delle posizioni

# %%
from hdt import HDTDocument  # noqa: E402

t0 = time.perf_counter()
doc = HDTDocument(str(HDT_PATH))
load_time = time.perf_counter() - t0
print(f"HDT loaded in {load_time:.1f}s")
print(f"  total triples:       {doc.total_triples:,}")
print(f"  distinct subjects:   {doc.nb_subjects:,}")
print(f"  distinct predicates: {doc.nb_predicates:,}")
print(f"  distinct objects:    {doc.nb_objects:,}")


# %% [markdown]
# ## 2. Smoke test 1 — uscenti di Q42 (Douglas Adams)
#
# Item "normale" non-hub. Atteso: poche centinaia di triple uscenti, lookup in pochi ms.
#
# **API pyHDT**: `doc.search_triples(s, p, o)` accetta stringhe URI (full form)
# o `""` per wildcard. Restituisce `(iterator, count)` dove `count` è la
# cardinalità del pattern *senza dover enumerare* — molto utile per fare
# count rapidi.

# %%
q42 = f"{WD_ENTITY}Q42"

t0 = time.perf_counter()
_iter, n_out = doc.search_triples(q42, "", "")
elapsed = (time.perf_counter() - t0) * 1000
print(f"Q42 outgoing (any predicate, any object): {n_out:,} triples in {elapsed:.2f}ms")


# %% [markdown]
# ## 3. Smoke test 2 — l'hub che ha rotto SPARQL
#
# `?n wdt:P31 wd:Q5` = "tutto ciò che è instance of human" = 10M+ righe.
# Su `query.wikidata.org` questa query (anche solo il count) andava in timeout
# costantemente. Su HDT è un lookup diretto nell'indice POS (predicate-object-subject):
# il count è praticamente immediato.

# %%
q5 = f"{WD_ENTITY}Q5"
p31 = f"{WDT_DIRECT}P31"

t0 = time.perf_counter()
_iter, n_p31_in_q5 = doc.search_triples("", p31, q5)
elapsed = (time.perf_counter() - t0) * 1000
print(f"?n wdt:P31 wd:Q5 (incoming P31 to Q5): {n_p31_in_q5:,} in {elapsed:.2f}ms")


# %% [markdown]
# ## 4. Smoke test 3 — uscenti P31 di Q5
#
# Q5 è una classe. Le uscenti P31 sono "Q5 è instance of *cosa*" — pochissime
# (es. Q5 P31 Q16889133 "common name for taxon" o simili).
# Enumeriamole per davvero per vedere il contenuto.

# %%
t0 = time.perf_counter()
out_iter, n = doc.search_triples(q5, p31, "")
elapsed = (time.perf_counter() - t0) * 1000
print(f"wd:Q5 wdt:P31 ?n: {n} triples in {elapsed:.2f}ms")
for s, p, o in out_iter:
    qid = o.rsplit("/", 1)[-1]
    print(f"  Q5 → P31 → {qid}")


# %% [markdown]
# ## 5. Helper — vicini Q-target di un'entità
#
# Funzione riutilizzabile che è il building block del BFS-3-onde.
# Per un dato QID restituisce `(outgoing, incoming)` dove ognuno è un dict
# `{predicate_uri: [list of neighbor QID URIs]}`.
#
# **Filtri applicati**:
# - Solo predicati `wdt:` (truthy direct), escludendo i nodi statement reificati
# - Solo neighbor che sono Q-entity *vere* (non statement nodes tipo `Q42-uuid`)
#
# Replicano esattamente quello che facevamo via SPARQL FILTER, ma fatti a
# livello Python sui risultati HDT — molto più veloce perché non c'è
# materializzazione lato server.

# %%
def is_q_entity(uri: str) -> bool:
    """True iff URI is a pure Wikidata Q-entity (not a statement node, not a literal).

    Statement nodes have the form 'http://www.wikidata.org/entity/Q42-UUID'
    and we want to exclude those — they are reified statement IDs, not concepts.
    """
    if not uri.startswith(WD_ENTITY):
        return False
    last = uri[len(WD_ENTITY):]                              # e.g. "Q42" or "Q42-abc-uuid"
    if not last.startswith("Q"):
        return False
    return "-" not in last and last[1:].isdigit()            # pure Q\d+


def neighbors_q(
    doc: "HDTDocument",
    qid: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return (outgoing, incoming) Q-neighbours of a Wikidata entity.

    Args:
        doc: opened HDTDocument
        qid: Q-id like "Q42" (without prefix)

    Returns:
        (out, in_) where each is {predicate_uri: [neighbour_uri, ...]}
        — only wdt: predicates, only Q-entity neighbours.
    """
    s = f"{WD_ENTITY}{qid}"
    out: dict[str, list[str]] = {}
    in_: dict[str, list[str]] = {}

    # Outgoing: ?p ?o where subject=qid
    out_iter, _ = doc.search_triples(s, "", "")
    for _, p, o in out_iter:
        if p.startswith(WDT_DIRECT) and is_q_entity(o):
            out.setdefault(p, []).append(o)

    # Incoming: ?s ?p where object=qid
    in_iter, _ = doc.search_triples("", "", s)
    for sn, p, _ in in_iter:
        if p.startswith(WDT_DIRECT) and is_q_entity(sn):
            in_.setdefault(p, []).append(sn)

    return out, in_


# %% [markdown]
# ## 6. Test helper su Q42
#
# Verifica che la funzione restituisca dati sensati su un item normale.
# Stampiamo i top 10 predicati per ciascuna direzione, ordinati per fan-out.

# %%
t0 = time.perf_counter()
out, in_ = neighbors_q(doc, "Q42")
elapsed = time.perf_counter() - t0
total_out = sum(len(v) for v in out.values())
total_in = sum(len(v) for v in in_.values())
print(f"Q42 neighbours computed in {elapsed:.2f}s")
print(f"  outgoing: {len(out)} predicates, {total_out} edges")
print(f"  incoming: {len(in_)} predicates, {total_in} edges")
print()
print("Top-10 outgoing predicates by fan-out:")
for p, ns in sorted(out.items(), key=lambda kv: -len(kv[1]))[:10]:
    pid = p.rsplit("/", 1)[-1]
    sample = ns[0].rsplit("/", 1)[-1]
    print(f"  {pid:>8}: {len(ns):>5}   e.g. {sample}")
print()
print("Top-10 incoming predicates by fan-out:")
for p, ns in sorted(in_.items(), key=lambda kv: -len(kv[1]))[:10]:
    pid = p.rsplit("/", 1)[-1]
    sample = ns[0].rsplit("/", 1)[-1]
    print(f"  {pid:>8}: {len(ns):>5}   e.g. {sample}")


# %% [markdown]
# ## 7. Stress test — enumerazione vs solo count
#
# Il *count* di "incoming P31 di Q5" è istantaneo (vedi cella 3).
# Ma se proviamo a enumerare effettivamente i ~10M URI in Python, l'overhead
# Python-side diventa il collo di bottiglia. Misuriamo per capire se
# l'enumerazione completa è praticabile o se serve evitarla.
#
# Se questa cella richiede troppo tempo (es. >60s), la lezione operativa è:
# **per gli hub usare il count, non enumerare.** Per il BFS questo significa
# sapere a priori quali nodi non vanno espansi.
#
# **Cella opzionale** — interrompila se serve.

# %%
t0 = time.perf_counter()
iter_p31_q5, n = doc.search_triples("", p31, q5)
print(f"Pattern declared, count = {n:,}. Now enumerating...")

# Enumerate but only count by hand to compare with declared count
counted = 0
sample = []
for s, p, o in iter_p31_q5:
    counted += 1
    if counted <= 5:
        sample.append(s.rsplit("/", 1)[-1])
    if counted % 1_000_000 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {counted:,} enumerated in {elapsed:.1f}s ({counted/elapsed:,.0f} triples/s)")

elapsed = time.perf_counter() - t0
print(f"\nEnumeration complete: {counted:,} triples in {elapsed:.1f}s")
print(f"Average rate: {counted/elapsed:,.0f} triples/s")
print(f"Sample first 5 instances: {sample}")
