# Method — verifica dei dati e delle scelte

> Sezione 3 di `relazione/bozza/main.tex`. Numeri ricalcolati il 2026-06-08 con
> `documents/verify_report_numbers.py` (blocco METHOD). Claim qualitativi ancorati a codice/note.

## 1. Dati numerici

| Claim | Valore reale | Fonte / verifica |
|-------|--------------|------------------|
| "~42M passages" | **41,995,761** (42.0M) | `data/faiss_index/shard_*_ids.npy`: somma delle lunghezze dei 9 array id = passaggi codificati. Snippet sotto. Cfr. `PROJECT_NOTES.md` §6. |
| "$6.61\times10^8$ edges" | **661,471,158** (661.5M) | `data/db/edges.parquet` metadata `num_rows` (vedi `abstract.md` §2.1). |
| "Q30, USA, has 2.35M edges" | **2,353,407** (2.35M) | `data/db/node_stats.parquet`, riga `qid='Q30'`, `total_degree`. Snippet sotto. |
| grid "90 rerank cells plus baseline" (= 91) | **91** = 1 + 90 (3 dist × 6 thr × 5 α) | `08_llm_eval.ipynb` §2; conteggio `judgments_*.jsonl`. |
| "1000 NQ-open queries" | **1000** righe per condizione | ogni `judgments_*.jsonl`; `queries_curated.jsonl`. |

```python
# 42M passages (legge solo l'header dei .npy, non i dati)
import numpy as np, pathlib
ids = sorted(pathlib.Path("data/faiss_index").glob("shard_*_ids.npy"))
print(sum(np.load(p, mmap_mode="r").shape[0] for p in ids))   # -> 41995761

# Q30 degree
import duckdb
print(duckdb.sql("SELECT total_degree FROM 'data/db/node_stats.parquet' WHERE qid='Q30'").fetchone())
# -> (2353407,)
```

> **Nota editoriale**: 2,353,407 ≈ **2.35M**. Il `.tex` scrive "2.35M edges" (scelta
> 2026-06-08, per coerenza con 661.5M).

## 2. Scelte metodologiche (claim qualitativi)

| Claim nel testo | Dove è implementato / verifica |
|-----------------|--------------------------------|
| **Departure 1** — segmentazione sentence-aligned + padding a 100 parole (vs `psgs_w100` DPR a code ragged, non paddati) | `01_corpus_preparation.ipynb`; `utils/text_processing.py` (`segment_article`); razionale e "perché non DPR" in `PROJECT_NOTES.md` §4.2. |
| Corpus = release Wikipedia Dec-2018 di [cuconasu2024power] | `florin-hf/wiki_dump2018_nq_open`, `PROJECT_NOTES.md` §2 Step 1 / §4.1. |
| Contriever + FAISS `IndexFlatIP`, ~42M | `03_embedding.ipynb`; `PROJECT_NOTES.md` Step 1b.4. |
| Filtro query: risposte ≤5 token + entità ReFinED in domanda e **tutte** le varianti | `02_nq_filtering.ipynb`; `PROJECT_NOTES.md` §2 Step 2 (31,372 query → subset 1000 curato). |
| **HDT locale** (Jan 2022, ~166 GB) via pyHDT; SPARQL pubblico abbandonato dopo timeout dei `COUNT` su hub (es. Q5, *human*) | `PROJECT_NOTES.md` §4.7 (timeout empirici su `wd:Q5`); script `scripts/pipeline/hdt_export_per_predicate.py`. |
| Export per-predicato di tutte le triple Q–Q `wdt:*` | `scripts/pipeline/hdt_export_per_predicate.py`; `PROJECT_NOTES.md` §4.8. |
| **Meet-in-the-middle**: dist=1 lookup, dist=2 intersezione N1, dist=3 edge-probe N1(q)×N1(d); threshold filtra i *bridge*, non gli endpoint | `utils/kg.py` `is_reachable` / `_reachable_pairs_min_dist` (CTE `d1/d2/d3a/d3b`); `PROJECT_NOTES.md` §4.10, §6.3. |
| "mai materializza il vicinato di un hub" | conseguenza diretta dell'algoritmo (solo membership/join), cfr. Q30 2.35M archi mai espansi. |
| `kg_score = connected_ratio × purity_ratio` con le due definizioni | `utils/kg.py` (`connected_ratio`, `purity_ratio`, `kg_score`); proposal; `PROJECT_NOTES.md` §2 Step 5. |
| Fusione `final = (1−α)·dense_norm + α·kg_score`, **α = peso KG** | `08_llm_eval.ipynb` §4 (`final_α = (1-α)·dense_norm + α·kg_score`). **Verificato**: `alpha_9` = 90% KG = condizione peggiore (vedi `results.md`). |
| `dense_norm` = rescaling **max-only** per-query (`s/max_q s`), cap a 1, niente min-max | `07_kg_rerank.ipynb` §6 e `08_llm_eval.ipynb` §4 (`scale_by_max_per_group`); razionale in `PROJECT_NOTES`/memoria `project_dense_norm_maxonly`. |
| Grid: k∈{1,2,3}, t∈{500,1k,2k,5k,10k,∞}, α∈{0.1,0.3,0.5,0.7,0.9} | `08_llm_eval.ipynb` §2 (`ALPHAS_TO_TEST`, `THR_VARIANTS`, `CELL_DIST_VARIANTS`). |
| **Departure 2** — prompt derivato da Power of Noise (`src/prompt_dataset.py`) con due deviazioni: (a) one-shot exemplar per il modello **base** non-instruct, (b) **domanda spostata SOPRA i documenti** (il paper la mette dopo) | `08_llm_eval.ipynb` §4.1: commento "two deliberate deviations: 1. One-shot example prepended... 2. Question moved ABOVE Documents (vs paper's Question-after)"; modello `meta-llama/Llama-2-7b-hf` (base). NB: il "top-1 passage a Document [5]" è la convenzione `reversed(passage_ids)` (tail-bias, allineata al paper), NON una delle due deviazioni. |
| **Departure 3** — LLM-as-judge **Qwen2.5-7B-Instruct** (modello diverso, per evitare self-preference), sostituisce l'exact-match substring | `09_llm_judge.ipynb` intro + §4 (`JUDGE_MODEL_NAME`, `JUDGE_SYSTEM_PROMPT`; motivazione "judge MUST be a different model"). |

**Esito**: tutti i numeri confermati; tutte le scelte metodologiche tracciate a codice/note.