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
| `05_answer_curation.ipynb` | CORE | 4.5 | Identifica 344 query con 0 entità nei loro passaggi; produce `data/NQ_answer/curation_results.jsonl` con mapping originale→sostituta. |
| `06_apply_curation.ipynb` | CORE | 4.5 | Applica lo swap delle 344 query → produce `queries_curated.jsonl`, `top100_curated.parquet`, `passage_entities_curated.parquet`, `query_embeddings_curated.npy`. |
| `base/preprocessing.ipynb` | LEGACY | — | Vecchio notebook Colab. Riferimento storico. **Non eseguire** (vedi CLAUDE.md). |

---

## 2. Script `scripts/`

A partire dal 2026-05-04, gli script sono organizzati in 4 sottocartelle (`pipeline/`, `diagnostic/`, `tooling/`, `legacy/`) — vedi §6 per il layout completo.

### 2.1 Pipeline KG (CORE) — `scripts/pipeline/`

| File | Layer | Ambiente | Scopo |
|------|-------|----------|-------|
| `pipeline/hdt_export_per_predicate.py` | Layer 1 | WSL2 (pyHDT) | Esporta tutte le triple Q-Q `wdt:*` dal dump Wikidata HDT, una pass per predicato. Output `data/db/edges.parquet` (661.471.158 righe). Sostituisce il deprecato wildcard iterator (vedi PROJECT_NOTES §4.8). |
| `pipeline/node_stats.py` | Layer 1.5 | Windows venv (polars streaming) | Calcola `in_degree`/`out_degree`/`total_degree` per ogni QID che appare in `edges.parquet`. Output `data/db/node_stats.parquet`. |
| `pipeline/build_labels.py` | Layer 1.6 | WSL2 (pyHDT) | Lookup `rdfs:label@en` per ogni QID dataset → `data/db/labels.parquet`. Per ispezione human-readable. |
| `pipeline/build_n1.py` | Layer 3 | Windows venv (DuckDB) | Precompute 1-hop neighborhood + degree del vicino, per i QID di `seeds ∪ passage_entities`. Output `data/n1/n1.parquet` (~93M righe). Sostituisce BFS-N3 deprecato (vedi §4.10). |

> **Layer 4** è stato **fuso e spostato in `utils/kg.py`** il 2026-05-03 (vedi §3 e §4.11 di PROJECT_NOTES). I file `scripts/kg.py` e `scripts/kg_advanced.py` sono stati rimossi: il primo importava cross-script, il secondo ereditava dal primo — entrambi i pattern sono stati sostituiti da una sola classe `KGScorer` self-contained in `utils/`.

### 2.2 Diagnostica (DIAGNOSTIC) — `scripts/diagnostic/`

| File | Ambiente | Scopo |
|------|----------|-------|
| `diagnostic/ablation_diagnostic.py` | Windows venv | Per ogni threshold ∈ {500,1000,2000,5000,10000,∞}: classifica le query in clean/mixed/all-hub e calcola `mean |N1_filtered|`. Output `data/n1/ablation_summary.parquet` + `ablation_invalidated_per_t.jsonl`. |
| `diagnostic/seed_degree_stats.py` | Windows venv | Distribuzione `total_degree` sul pool di seed (question_qids + answer_variant_qids). Usato per dimensionare la threshold di hub-banning. |
| `diagnostic/verify_completeness.py` | WSL2 (pyHDT) | Verifica che `edges.parquet` copra tutte le triple `wdt:*` Q-Q dell'HDT. Output `data/db/verification.json`. |
| `diagnostic/check_qids.py` | Windows venv (polars) | Pre-flight: verifica che tutti i QID dai dataset matchino `^Q\d+$`. |
| `diagnostic/hdt_query_test.py` | WSL2 (pyHDT) — jupytext | Smoke test HDT + helper di lookup. Paired `.py`/`.ipynb` (vedi `pyproject.toml [tool.jupytext]`). |

### 2.3 Tooling (CORE — eseguito raramente) — `scripts/tooling/`

| File | Scopo |
|------|-------|
| `tooling/patch_refined.py` | Patch source-level di ReFiNed V1 per Windows + Python 3.12+ + transformers 4.48+. Idempotente. **Chiamato in automatico da `02_nq_filtering.ipynb` e `04_answer_preparation.ipynb` via subprocess** — non rimuovere. Eseguibile anche manuale: `python scripts/tooling/patch_refined.py [--check]`. |

### 2.4 Legacy / archeologia — `scripts/legacy/`

| File | Motivo |
|------|--------|
| `legacy/build_n3.py` | BFS-N3 con hub-banning. Deprecato 2026-04-28 dopo 89% TIMEOUT (vedi PROJECT_NOTES §4.10). Sostituito da `pipeline/build_n1.py`. |
| `legacy/build_test_queries.py` | Generatore di test query SPARQL con VALUES variabili. Pre-pivot HDT (vedi §4.7). Non più usato. |
| `legacy/patch_answer_preparation.py` | One-shot. Inserì la sezione 3b (subset 1k) in `04_answer_preparation.ipynb`. Già applicato; il notebook è ora committato con la modifica. |
| `legacy/_extract_cells.py` | One-shot. Estrasse celle da `04_answer_preparation.ipynb` per editing offline. |
| `legacy/_extract_curation_cells.py` | One-shot. Stesso, per `05_answer_curation.ipynb`. |
| `legacy/_copy_nb.py` | One-shot. Helper triviale: copia notebook → JSON per editing manuale. |

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
| `scripts/tooling/patch_refined.py` | invocato via `subprocess` da `02_nq_filtering.ipynb` e `04_answer_preparation.ipynb` |

Nessun altro script ha dipendenze incrociate. Ogni script in `scripts/` è self-contained ed eseguito da terminale; tutto il codice condiviso vive in `utils/`.

---

## 5. Diagnosi e problema attuale (RISOLTO 2026-05-03)

`scripts/` non è un package Python. Il vecchio `scripts/kg_advanced.py` faceva `from scripts.kg import ...`, che falliva con `ModuleNotFoundError` quando lanciato come standalone.

**Risolto** spostando il codice condivisibile in `utils/kg.py` (`utils/` è già package). Pattern adottato:

- **Codice riusabile** → in `utils/` (già package, già funziona da notebook)
- **Script eseguibili** → in `scripts/`, self-contained, importano da `utils/` (mai cross-import tra script)

---

## 6. Layout del repo (applicato 2026-05-04)

```
dl-RAG-denseAndKG/
├── 01_corpus_preparation.ipynb         # notebook in root (decisione 2026-05-04: NON spostati in notebooks/)
├── 02_nq_filtering.ipynb
├── 03_embedding.ipynb
├── 04_answer_preparation.ipynb
├── 05_answer_curation.ipynb
├── 06_apply_curation.ipynb
├── utils/                              # libreria importabile (package)
│   ├── __init__.py                     # re-export KGScorer
│   ├── text_processing.py
│   └── kg.py                           # KGScorer unificata (Layer 4)
├── scripts/                            # script eseguibili, self-contained, mai cross-import
│   ├── pipeline/                       # build pipeline KG (Layer 1-3)
│   │   ├── hdt_export_per_predicate.py
│   │   ├── node_stats.py
│   │   ├── build_labels.py
│   │   └── build_n1.py
│   ├── diagnostic/                     # ispezione e verifica (eseguibile a piacimento)
│   │   ├── ablation_diagnostic.py
│   │   ├── seed_degree_stats.py
│   │   ├── verify_completeness.py
│   │   ├── check_qids.py
│   │   └── hdt_query_test.py
│   ├── tooling/                        # supporto runtime (chiamato da subprocess dai notebook)
│   │   └── patch_refined.py
│   └── legacy/                         # archeologia (one-shot già applicati + deprecated)
│       ├── build_n3.py
│       ├── build_test_queries.py
│       ├── patch_answer_preparation.py
│       ├── _extract_cells.py
│       ├── _extract_curation_cells.py
│       └── _copy_nb.py
└── base/
    └── preprocessing.ipynb             # vecchio notebook Colab, solo riferimento
```

### Principi guida

1. **Codice riusabile** → in `utils/` (già package, importabile dovunque). **Script eseguibili** → in `scripts/<dominio>/`, self-contained, mai cross-import.
2. **Notebook in root** (decisione 2026-05-04): spostarli in `notebooks/` avrebbe richiesto di aggiornare ~50 path relativi tipo `Path.cwd() / "data"` in 6 notebook — pragmaticamente non vale la pena.
3. **`legacy/` invece di cancellare**: preserva storia git + link in PROJECT_NOTES. Eventualmente cancellabile dopo chiusura paper.
4. **Risoluzione path negli script**: tutti usano `_find_repo_root()` che fa walk-up a `pyproject.toml` — robusto a futuri rename di cartella.

### Cronologia

1. ~~Salvare questo report.~~ (fatto)
2. ~~Commit e push dello stato attuale (snapshot pre-riorg).~~ (fatto)
3. ~~Fusione `kg.py` + `kg_advanced.py` → `utils/kg.py`.~~ (fatto 2026-05-03)
4. ~~Audit + rinomina con prefisso `0X_` di tutti i 6 notebook attivi.~~ (fatto 2026-05-04)
5. ~~Riorganizzazione `scripts/` in `pipeline/`/`diagnostic/`/`tooling/`/`legacy/`.~~ (fatto 2026-05-04)

---

## 7. Note operative

- **Aggiornare questo file** quando si aggiunge/rimuove uno script, si cambia stato di un file, o si esegue uno spostamento di cartelle. Coerente con la regola di tracking di CLAUDE.md.
- **Aggiornare anche PROJECT_NOTES.md** sezione "Struttura del Progetto" se la riorganizzazione viene applicata.