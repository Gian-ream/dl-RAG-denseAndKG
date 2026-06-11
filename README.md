# Distractor Filtering Through Open KG Concept Proximity for Dense RAG Systems

Can *pure graph topology* on Wikidata — bare ≤3-hop reachability, no relation
semantics — filter the distracting passages that dense retrieval feeds to a RAG
reader? We test this on NQ-open with a Contriever → Llama-2 pipeline, reranking
the top-100 passages by a topological score computed over a 661.5M-edge entity
graph, across a grid of 90 (distance × hub-threshold × weight) settings.

**The answer is negative, and explained**: the KG signal never beats the dense
baseline (0.364) beyond noise (best cell 0.372) and actively degrades accuracy
as its weight grows (down to 0.290); 13/90 conditions are significant under
exact McNemar + Bonferroni — all worse. The mechanism: on a graph as dense as
Wikidata, 2–3-hop reachability saturates, so the score stops discriminating.

📄 **Report**: [`relazione/bozza/main.pdf`](relazione/bozza/main.pdf) — 2-page
body + appendix (pipeline, environment, design choices, failed attempts).

## Repository structure

```
01..09_*.ipynb        # the pipeline, as 9 notebooks in execution order
utils/                # importable library (KGScorer, text processing)
scripts/
├── pipeline/         # Wikidata graph build, Layers 1–3 (Linux/WSL2)
├── diagnostic/       # ablation / verification utilities
├── tooling/          # patch_refined.py (auto-run by the notebooks)
└── legacy/           # deprecated attempts, kept for reference
documents/            # PROJECT_NOTES (full decision log), DATA_DICTIONARY,
                      # REPO_INVENTORY, GLOSSARIO, per-section fact-check files,
                      # verify_*.py (recompute every number in the report)
relazione/            # LaTeX report (bozza/) + course template
data/                 # gitignored except artifacts < 100 MB (see below)
```

## Pipeline

Nine notebooks, each reading the artifacts of the previous ones from `data/`:

1. `01_corpus_preparation` — download corpus, segment into sentence-aligned 100-word passages (~42M)
2. `02_nq_filtering` — filter NQ-open (≤5-token answers + ReFinED entities) → 31,372 queries
3. `03_embedding` — Contriever-encode all passages, build 9 FAISS shards
4. `04_answer_preparation` — top-100 retrieval + passage entity linking (1000-query subset)
5. `05_answer_curation` — find the 344 queries whose top-100 carries no entities
6. `06_apply_curation` — apply the substitutions → `*_curated.*` files
7. `07_kg_rerank` — KG reachability (connected/purity ratios) over the full grid
8. `08_llm_eval` — Llama-2-7B generates answers for all 91 conditions (~20 h GPU)
9. `09_llm_judge` — Qwen2.5-7B judges every answer; accuracy + McNemar/Bonferroni

The Wikidata graph itself is built once by `scripts/pipeline/` (HDT → edges →
degrees → 1-hop neighbourhoods), which run on Linux/WSL2 via `pyHDT`.

## Setup

Main environment (Windows + [uv](https://docs.astral.sh/uv/)):

```bash
uv sync          # Python >=3.10, pinned deps, PyTorch cu128
```

Then:

- **HuggingFace token**: put a read token in a `.hf_token` file at the repo
  root (gitignored). Required only for the licence-gated Llama-2 (notebook 08);
  the other notebooks use it just for faster downloads.
- **faiss-gpu**: no Windows pip wheel exists, so it lives in a separate
  miniconda install and is bridged into the uv venv at runtime
  (`os.add_dll_directory` + `sys.path.insert` at the top of notebooks 03/04 —
  **adapt those conda paths to your machine**).
- **ReFinED patch**: ReFinED V1 breaks on Windows + Python 3.12 +
  transformers 4.48 (three bugs, see the report appendix).
  `scripts/tooling/patch_refined.py` fixes the installed package and is invoked
  automatically by the notebooks that load the model — nothing to do manually.

## Data

| What | Where | Size |
|---|---|---|
| Outputs needed to **verify the results** (per-query judgments, top-100, KG scores, curation, gold answers) | committed in this repo under `data/` | ~0.6 GB |
| Full `data/` folder (FAISS shards, corpus TSV, edge list, DuckDB, model caches, …) | [Google Drive](https://drive.google.com/drive/folders/1RlQuowaWmVF0fe2fcAqVBBaT4dmsnU7s) | ~614 GB |
| Wikidata HDT dump + prebuilt index (`latest-all-06-Jan-2022.hdt` + `.index.v1-1`) | [hdt-dumps](https://hdt-dumps.cluster.ai.wu.ac.at/dumps/) → `data/Wikidata_service/` | 166 + 107 GB |
| Corpus (gold-augmented Wikipedia Dec-2018) | [`florin-hf/wiki_dump2018_nq_open`](https://huggingface.co/datasets/florin-hf/wiki_dump2018_nq_open) — auto-downloaded by notebook 01 | — |
| Queries (NQ-open + gold) | [`florin-hf/nq_open_gold`](https://huggingface.co/datasets/florin-hf/nq_open_gold) — auto-downloaded by notebook 02 | — |

The prebuilt HDT **index** is required: without it pyHDT tries to build the
index in RAM, which exceeds a default WSL2 memory cap.

## Reproducing the results

**Re-run the judge (1 GPU).** Notebook `09_llm_judge` works out of the box
from a fresh clone: its inputs are committed and Qwen2.5-7B-Instruct is
downloaded automatically (~15 GB, ungated). It performs 91,000 judgments, so a
full pass takes time; it is resumable and skips already-judged rows.

**Full rerun (discouraged).** Every artifact except the HDT dump and its index
can be re-created by the notebooks themselves, in order 01 → 09, plus the
`scripts/pipeline/` layers for the graph. The computational cost is high:
depending on the stage and the machine, a single notebook can take days.
Reference machine: RTX 5070 Ti (16 GB), 64 GB RAM, 24 cores; WSL2 Ubuntu 24.04.

## Documentation

- `documents/PROJECT_NOTES.md` — the full decision log (in Italian): every
  design choice, failed attempt and pivot, in chronological order.
- `documents/DATA_DICTIONARY.md` — schema, producer and consumers of every
  data artifact.
- `documents/REPO_INVENTORY.md` — status and purpose of every notebook/script.
- `documents/{abstract,method,results,…}.md` — per-section fact-check of the
  report (claim → source → verification snippet).

## Statement on the use of AI

This project was developed in close collaboration with Claude (Anthropic) —
see the dedicated statement in the report for the full breakdown of roles.