# REPO_INVENTORY

Mappa di notebook, script e utility del progetto.
Snapshot al **2026-05-03** (branch `develop`).
Documento vivo: aggiornare quando si aggiunge/rimuove un file o si cambia stato.

Stato di un file:
- **CORE**: parte attiva della pipeline produttiva, attualmente in uso.
- **DIAGNOSTIC**: utility di analisi/ablation, non in pipeline ma utile.
- **TOOLING**: script di setup/manutenzione (patch, build helper), eseguito raramente.
- **WIP**: lavoro in corso, non ancora stabilizzato.
- **LEGACY**: archeologia mantenuta per riferimento, non eseguire.
- **ONE-SHOT**: eseguito una sola volta in passato per editing/migrazione, ora inerte.

---

## 1. Notebook (root del repo)

| File | Stato | Step pipeline | Scopo |
|------|-------|---------------|-------|
| `01_corpus_preparation.ipynb` | CORE | 0 + 1 | Download corpus HF (`florin-hf/wiki_dump2018_nq_open`) → `data/wikipedia_2018_clean/articles_clean.tsv`; segmentazione sentence-aligned in passaggi 100-word → `data/wikipedia_2018_sentence_aligned/psgs_w100_sentence.tsv` (~42M passaggi, 25 GB). Output consumato da `03_embedding.ipynb`. |
| `02_nq_filtering.ipynb` | CORE | 2 | Filtra `florin-hf/nq_open_gold`: token ≤ 5 + entità ReFiNed su question + answer. Output `data/NQ_question/qa_all_entities.jsonl` (31.372 query). Chiama `scripts/patch_refined.py` via subprocess. |
| `03_embedding.ipynb` | CORE | 1b | Encoding Contriever (mean pooling) di tutti i ~42M passaggi; build di 9 shard FAISS (`IndexFlatIP`) in `data/faiss_index/`. |
| `04_answer_preparation.ipynb` | CORE | 4 | Top-100 retrieval per le 1000 query del subset; entity linking dei passaggi recuperati; output `data/NQ_answer/top100_*.parquet` + `passage_entities*.parquet`. |
| `answer_curation.ipynb` | CORE | 4.5 | Identifica 344 query con 0 entità nei loro passaggi; produce `data/NQ_answer/curation_results.jsonl` con mapping originale→sostituta. |
| `apply_curation.ipynb` | CORE | 4.5 | Applica lo swap delle 344 query → produce `queries_curated.jsonl`, `top100_curated.parquet`, `passage_entities_curated.parquet`. |
| `base/preprocessing.ipynb` | LEGACY | — | Vecchio notebook Colab. Riferimento storico. **Non eseguire** (vedi CLAUDE.md). |

---

## 2. Script `scripts/`

### 2.1 Pipeline KG (CORE)

| File | Layer | Ambiente | Scopo |
|------|-------|----------|-------|
| `hdt_export_per_predicate.py` | Layer 1 | WSL2 (pyHDT) | Esporta tutte le triple Q-Q `wdt:*` dal dump Wikidata HDT, una pass per predicato. Output `data/db/edges.parquet` (661.471.158 righe). Sostituisce il deprecato wildcard iterator (vedi PROJECT_NOTES §4.8). |
| `node_stats.py` | Layer 1.5 | Windows venv (polars streaming) | Calcola `in_degree`/`out_degree`/`total_degree` per ogni QID che appare in `edges.parquet`. Output `data/db/node_stats.parquet`. |
| `build_labels.py` | Layer 1.6 | WSL2 (pyHDT) | Lookup `rdfs:label@en` per ogni QID dataset → `data/db/labels.parquet`. Per ispezione human-readable. |
| `build_n1.py` | Layer 3 | Windows venv (DuckDB) | Precompute 1-hop neighborhood + degree del vicino, per i QID di `seeds ∪ passage_entities`. Output `data/n1/n1.parquet` (~93M righe). Sostituisce BFS-N3 deprecato (vedi §4.10). |

> **Layer 4** è stato **fuso e spostato in `utils/kg.py`** il 2026-05-03 (vedi §3 e §4.11 di PROJECT_NOTES). I file `scripts/kg.py` e `scripts/kg_advanced.py` sono stati rimossi: il primo importava cross-script, il secondo ereditava dal primo — entrambi i pattern sono stati sostituiti da una sola classe `KGScorer` self-contained in `utils/`.

### 2.2 Diagnostica (DIAGNOSTIC)

| File | Ambiente | Scopo |
|------|----------|-------|
| `ablation_diagnostic.py` | Windows venv | Per ogni threshold ∈ {500,1000,2000,5000,10000,∞}: classifica le query in clean/mixed/all-hub e calcola `mean |N1_filtered|`. Output `data/n1/ablation_summary.parquet` + `ablation_invalidated_per_t.jsonl`. |
| `seed_degree_stats.py` | Windows venv | Distribuzione `total_degree` sul pool di seed (question_qids + answer_variant_qids). Usato per dimensionare la threshold di hub-banning. |
| `verify_completeness.py` | WSL2 (pyHDT) | Verifica che `edges.parquet` copra tutte le triple `wdt:*` Q-Q dell'HDT. Output `data/db/verification.json`. |
| `check_qids.py` | Windows venv (polars) | Pre-flight: verifica che tutti i QID dai dataset matchino `^Q\d+$`. |
| `hdt_query_test.py` | WSL2 (pyHDT) — jupytext | Smoke test HDT + helper di lookup. Paired `.py`/`.ipynb` (vedi `pyproject.toml [tool.jupytext]`). |

### 2.3 Tooling (CORE — eseguito raramente)

| File | Scopo |
|------|-------|
| `patch_refined.py` | Patch source-level di ReFiNed V1 per Windows + Python 3.12+ + transformers 4.48+. Idempotente. **Chiamato in automatico da `02_nq_filtering.ipynb` via subprocess** — non rimuovere. Eseguibile anche manuale: `python scripts/patch_refined.py [--check]`. |

### 2.4 Legacy / archeologia

| File | Motivo |
|------|--------|
| `build_n3.py` | BFS-N3 con hub-banning. Deprecato 2026-04-28 dopo 89% TIMEOUT (vedi PROJECT_NOTES §4.10). Sostituito da `build_n1.py`. |
| `build_test_queries.py` | Generatore di test query SPARQL con VALUES variabili. Pre-pivot HDT (vedi §4.7). Non più usato. |

### 2.5 One-shot già applicati

| File | Scopo storico |
|------|---------------|
| `patch_answer_preparation.py` | Inserì la sezione 3b (subset 1k) in `04_answer_preparation.ipynb`. Già applicato; il notebook è ora committato con la modifica. |
| `_extract_cells.py` | Estrasse celle da `04_answer_preparation.ipynb` per editing offline. |
| `_extract_curation_cells.py` | Stesso, per `answer_curation.ipynb`. |
| `_copy_nb.py` | Helper triviale: copia notebook → JSON per editing manuale. |

---

## 3. Utility `utils/`

| File | Scopo | Importato da |
|------|-------|--------------|
| `utils/__init__.py` | Vuoto. Rende `utils` un package Python. | — |
| `utils/text_processing.py` | `segment_article`, `_init_file_worker`, `file_segment_worker`. Estratto da `01_corpus_preparation.ipynb` per essere importabile dai worker `multiprocessing` (su Windows usa `spawn`, le funzioni notebook-defined non si picklano). | `01_corpus_preparation.ipynb` |
| `utils/kg.py` | **CORE Layer 4** — fusione di ex `scripts/kg.py` + `scripts/kg_advanced.py`. Classe unica `KGScorer` con persistenza DuckDB su disco (`data/kg.duckdb`, default), modalità `read_only` per worker MP, query unificata `min_dist`, API griglia `kg_components_grid`/`kg_components_grid_batch` che ritornano `pd.DataFrame`. Diagnostico `is_reachable` mantenuto. Smoke test invocabile via `python -m utils.kg`. | (futuri notebook KG-rerank) |

---

## 4. Cross-reference degli import

| Importa | Importato da |
|---------|--------------|
| `utils.text_processing` | `01_corpus_preparation.ipynb` |
| `utils.kg` | (futuri notebook KG-rerank — già eseguibile via `python -m utils.kg`) |
| `scripts/patch_refined.py` | invocato via `subprocess` da `02_nq_filtering.ipynb` |

Nessun altro script ha dipendenze incrociate. Ogni script in `scripts/` è self-contained ed eseguito da terminale; tutto il codice condiviso vive in `utils/`.

---

## 5. Diagnosi e problema attuale (RISOLTO 2026-05-03)

`scripts/` non è un package Python. Il vecchio `scripts/kg_advanced.py` faceva `from scripts.kg import ...`, che falliva con `ModuleNotFoundError` quando lanciato come standalone.

**Risolto** spostando il codice condivisibile in `utils/kg.py` (`utils/` è già package). Pattern adottato:

- **Codice riusabile** → in `utils/` (già package, già funziona da notebook)
- **Script eseguibili** → in `scripts/`, self-contained, importano da `utils/` (mai cross-import tra script)

---

## 6. Piano di riorganizzazione (proposto, NON ancora applicato)

```
dl-RAG-denseAndKG/
├── notebooks/                          # i .ipynb (da decidere se spostare o lasciare in root)
│   ├── 01_01_corpus_preparation.ipynb
│   ├── 02_nq_filtering.ipynb
│   ├── 03_embedding.ipynb
│   ├── 04_answer_preparation.ipynb
│   ├── 04b_answer_curation.ipynb
│   └── 04c_apply_curation.ipynb
├── utils/                              # libreria importabile (già package)
│   ├── __init__.py
│   ├── paths.py                        # NEW: REPO_ROOT, N1_PATH, EDGES_PATH, _find_repo_root
│   ├── text_processing.py              # esistente
│   └── kg.py                           # NEW: KGScorer unificata (kg.py + kg_advanced.py fusi)
├── scripts/                            # script eseguibili, self-contained
│   ├── pipeline/
│   │   ├── hdt_export_per_predicate.py
│   │   ├── node_stats.py
│   │   ├── build_labels.py
│   │   └── build_n1.py
│   ├── diagnostic/
│   │   ├── ablation_diagnostic.py
│   │   ├── seed_degree_stats.py
│   │   ├── verify_completeness.py
│   │   ├── check_qids.py
│   │   └── hdt_query_test.py
│   ├── tooling/
│   │   └── patch_refined.py            # mantenuto (chiamato da subprocess dal notebook)
│   └── legacy/
│       ├── build_n3.py
│       ├── build_test_queries.py
│       ├── patch_answer_preparation.py
│       ├── _extract_cells.py
│       ├── _extract_curation_cells.py
│       └── _copy_nb.py
└── base/
    └── preprocessing.ipynb             # già qui
```

### Principi guida

1. **Fondere `kg.py` + `kg_advanced.py` in `utils/kg.py`** con un'unica classe `KGScorer` (persistenza disco di default, API griglia + diagnostici inclusi). Niente eredità, niente cross-import. Smoke test diventa `scripts/diagnostic/kg_smoke.py`.
2. **`utils/paths.py`** centralizza `_find_repo_root` e i path di dati — usato da chiunque ne abbia bisogno.
3. **Nessuno script in `scripts/` importa da altri script.** Tutto il codice condiviso vive in `utils/`.
4. **Notebook**: da decidere se spostare in `notebooks/` (richiede aggiornare i path relativi tipo `Path.cwd() / "data"` in 6 notebook) o lasciarli in root (più pragmatico).
5. **`legacy/` invece di cancellare**: mantieni storia + non rompi link in PROJECT_NOTES. Eventualmente cancellabili dopo che il paper è chiuso.

### Sequenza di esecuzione concordata

1. ~~Salvare questo report.~~ (fatto)
2. ~~Commit e push dello stato attuale (snapshot pre-riorg).~~ (fatto)
3. ~~Fusione `kg.py` + `kg_advanced.py` → `utils/kg.py`.~~ (fatto 2026-05-03)
4. **In corso**: aggiornamento progressivo dei notebook per i nuovi path/import, con verifica funzionale step-by-step.
5. Riorganizzazione interattiva del repo (spostamento file in `pipeline/`/`diagnostic/`/`tooling/`/`legacy/`).

---

## 7. Note operative

- **Aggiornare questo file** quando si aggiunge/rimuove uno script, si cambia stato di un file, o si esegue uno spostamento di cartelle. Coerente con la regola di tracking di CLAUDE.md.
- **Aggiornare anche PROJECT_NOTES.md** sezione "Struttura del Progetto" se la riorganizzazione viene applicata.