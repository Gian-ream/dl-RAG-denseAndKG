# Related work / bibliografia — verifica

> Sezione 2 di `relazione/bozza/main.tex`. Stato al 2026-06-08: la bibliografia contiene
> **esattamente le 5 citazioni del proposal**; nessun altro riferimento. I tool restano
> nominati nel testo senza citazione formale.

## Bibliografia = le 5 citazioni del proposal

| # proposal | Citazione | bibkey | Come compare nel report |
|------------|-----------|--------|--------------------------|
| [1] | Cuconasu et al., *The Power of Noise*, SIGIR 2024 | `cuconasu2024power` | citato inline (più volte) |
| [2] | Ayoola et al., *ReFinED*, NAACL 2022 | `ayoola2022refined` | citato inline (entity linking) |
| [3] | Bast, Hertel, Prange, *A Fair and In-Depth Evaluation of End-to-End Entity Linking Systems*, EMNLP 2023 | `bast2023fair` | citato inline (accanto a ReFinED) |
| [4] | Naveed et al., *A Comprehensive Overview of LLMs*, arXiv:2307.06435, 2023 | `naveed2023llm` | citato inline (intro, "grounds LLMs") |
| [5] | Peng et al., *Graph RAG: A Survey*, arXiv:2408.08921, 2024 | `peng2024graphrag` | **solo in bibliografia** via `\nocite` (per scelta: "GraphRAG" non nominato nel testo) |

Verifica: `references.bib` ha 5 voci; bibtex compila 5 `\bibitem`, 0 citazioni non risolte.
Metadati [3]/[4] verificati via web (EMNLP 2023 ACL Anthology; arXiv:2307.06435).

## Tool nominati SENZA citazione (non nel proposal)

Contriever, corpus DPR, HDT, Wikidata, Natural Questions, Llama 2, Qwen2.5 — citati per nome
nel testo, senza riferimento bibliografico (vincolo "solo citazioni del proposal").
"GraphRAG" e "Personalized PageRank" **non** sono nominati nel testo (scelta dell'autore).

## Implicazione

Dataset (NQ), modelli (Llama 2, Qwen2.5) e librerie (Contriever, HDT) appaiono senza citazione.
È conseguenza voluta del vincolo. Se un docente si aspetta la citazione di dataset/modelli,
va riconsiderato (richiederebbe riferimenti fuori dalla lista del proposal).
