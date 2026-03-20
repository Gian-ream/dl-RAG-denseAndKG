# Distractor Filtering Through Open KG Concept Proximity for Dense RAG Systems

## Regole di Comportamento

### Lingua
- **Documentazione**: scritta e aggiornata in italiano, man mano che il progetto evolve
- **Commenti nel codice**: tutti in inglese
- **Questo documento**: viene aggiornato progressivamente ad ogni step completato

### Metodologia di lavoro (Claude)
- **STOP prima del codice**: Claude NON scrive codice finché non ha spiegato cosa farà e perché, e non ha ricevuto conferma esplicita
- **Passo passo**: ogni implementazione viene fatta incrementalmente, una parte alla volta
- **Quadro generale prima del codice**: prima di scrivere codice, Claude spiega:
  1. **Dove siamo** nella pipeline (riferimento allo step del proposal)
  2. **Cosa faremo** e perchè
  3. **Come si collega** al resto del sistema
- **Spiegazione delle operazioni**: ogni blocco di codice viene accompagnato da spiegazione in italiano di cosa fa e perche e stato scritto cosi
- **Commenti "why" nel codice**: oltre al "what", i commenti spiegano il *motivo* delle scelte non banali (es. `# 3-hop because beyond this the graph becomes too noisy for filtering`)
- **Riferimenti al proposal**: quando un pezzo di codice implementa un punto specifico, viene indicato esplicitamente (es. "Questo implementa lo Step 3 — KG Subgraph Construction del proposal")
- **"Perche non X?"**: documentare anche le alternative scartate, non solo quelle scelte. Aiuta a capire il ragionamento
- **Mini-recap a inizio sessione**: all'inizio di ogni sessione di lavoro, riassumere brevemente dove siamo e cosa faremo

### Didattica
- **Operazioni matriciali/tensoriali**: quando si incontrano operazioni su matrici, vettori o tensori, Claude le spiega con esempi concreti (dimensioni, significato di ogni asse, cosa rappresenta il risultato)
- **Quiz periodici**: ogni tanto Claude propone mini-quiz per verificare la comprensione di concetti chiave appena trattati (es. "Cosa succede se il connected_ratio e 0?", "Perche normalizziamo gli embeddings?")
- **Glossario vivo**: il file `documents/GLOSSARIO.md` viene aggiornato man mano che si incontrano nuovi termini

---

## 1. Panoramica del Progetto

**Research question**: La prossimita concettuale su Wikidata puo ridurre i documenti distrattori nel dense retrieval e migliorare l'accuracy del RAG rispetto a baseline puramente semantiche?

**Idea chiave**: I documenti recuperati tramite dense retrieval possono essere semanticamente simili alla query ma non contenere la risposta (distrattori). Usando la topologia del knowledge graph Wikidata (prossimita a 3-hop), possiamo identificare e penalizzare questi distrattori nel reranking.

**Dataset**: Natural Questions Open (NQ-open)
**Evaluation**: Exact Match accuracy

---

## 2. Pipeline Completa

### Step 1 — Corpus Preparation
- **Input**: `florin-hf/wiki_dump2018_nq_open` (HuggingFace) — corpus Wikipedia Dec 2018 aumentato con gold documents NQ
  - Basato sul dump Wikipedia **Dec 20, 2018** (stesso di DPR, Karpukhin et al. 2020)
  - Integra gold documents dal dataset NQ con deduplicazione, ~21M documenti
  - Colonne: `text`, `title` (articoli interi, non pre-segmentati)
  - Usato da Silvestri et al. in "The Power of Noise"
- **Fonte**: HuggingFace dataset `florin-hf/wiki_dump2018_nq_open`
- **Processing**: Download da HF → cache locale come TSV → segmentazione sentence-aligned → indicizzazione con Contriever embeddings
- **Output**: `psgs_w100_sentence.tsv` (passaggi sentence-aligned da 100 parole) + indice FAISS
- **Storico**: inizialmente si usava `psgs_w100.tsv` (corpus DPR pre-segmentato da repo Silvestri), ricostruendo articoli con Polars e rimuovendo padding DPR. Ora si parte direttamente da articoli interi via HF.

### Step 2 — Query Filtering
- **Input**: `florin-hf/nq_open_gold` — NQ-open arricchito con gold documents da `wiki_dump2018_nq_open` (Silvestri et al.). 83,104 query totali (train 72,209 + validation 8,006 + test 2,889), split uniti.
- **Filtering** (pipeline a 2 stadi):
  1. **Token count ≤ 5** (tokenizer Contriever/BERT wordpiece, `add_special_tokens=False`): criterio ALL (tutte le varianti risposta devono passare). Risultato: 76,406 query (91.9%). Split answers (1.4%) sacrificate per coerenza.
  2. **Entity linking ReFiNed**: modello `questions_model` (fine-tuned WebQSP), entity_set `wikipedia` (~6M entità). Criterio: domanda ha ≥1 entità AND tutte le varianti risposta hanno ≥1 entità. Risultato: 31,372 query (41.1% delle 76,406).
- **Output**: `data/NQ_question/qa_all_entities.jsonl` (31,372 query filtrate con QID) + `data/NQ_question/qa_entities_general.jsonl` (76,406 query con entity info completa)
- **Notebook**: `nq_filtering.ipynb`

### Step 3 — KG Subgraph Construction (Wikidata Preparation)
- **Input**: Entita estratte dalle query filtrate + entita nei top-100 documenti recuperati
- **Processing**: Per ogni entita, estrazione vicinato 3-hop su Wikidata via SPARQL API
- **Output**: Grafo locale (networkx o similare) con le connessioni tra entita

### Step 4 — Baseline (Contriever-only)
- **Input**: Query filtrate
- **Processing**: Dense retrieval con Contriever -> top-5 documenti -> Llama2-7B generazione
- **Output**: Risposte generate + accuracy baseline

### Step 5 — KG-Enhanced Reranking
- **Input**: Top-100 documenti da dense retrieval + grafo locale Wikidata
- **Scoring**:
  ```
  kg_score = connected_ratio * purity_ratio

  connected_ratio = (# query entities con >= 1 doc entity entro 3-hop) / (# total query entities)
  purity_ratio = (# doc entities entro 3-hop di qualsiasi query entity) / (# total doc entities)

  final_score = alpha * contriever_score + (1 - alpha) * kg_score    [alpha = 0.5]
  ```
- **Processing**: Reranking dei 100 documenti -> top-5 -> Llama2-7B generazione
- **Output**: Risposte generate + accuracy KG-enhanced

### Step 6 — Evaluation
- **Metrica**: Exact Match (la risposta contiene almeno una delle risposte corrette da NQ-open)
- **Confronto**: Baseline vs KG-Enhanced

---

## 3. Librerie e Strumenti

### 3.1 Dense Retrieval
| Libreria | Uso | Note |
|----------|-----|------|
| **Contriever** (facebook/contriever) | Embedding dei passaggi e delle query | Modello unsupervised dense retriever di Meta |
| **FAISS** (faiss-cpu / faiss-gpu) | Indicizzazione e ricerca nearest-neighbor | Usato per retrieval efficiente sui vettori Contriever |

### 3.2 Entity Linking
| Libreria | Uso | Note |
|----------|-----|------|
| **ReFiNed** (amazon-science/ReFinED) | Entity linking end-to-end | Mappa menzioni nel testo a entita Wikidata (QID). Zero-shot capable. |

### 3.3 Knowledge Graph
| Strumento | Uso | Note |
|-----------|-----|------|
| **Wikidata SPARQL API** | Estrazione vicinato 3-hop per entita | Endpoint: `https://query.wikidata.org/sparql` |
| **NetworkX** | Rappresentazione e analisi del grafo locale | Per calcolo shortest path e prossimita |
| **SPARQLWrapper** | Client Python per query SPARQL | Interfaccia programmatica all'endpoint Wikidata |

### 3.4 Generazione
| Libreria | Uso | Note |
|----------|-----|------|
| **Llama2-7B** | Generazione risposte dato contesto | Via HuggingFace transformers o vLLM |
| **transformers** | Loading e inference del modello | HuggingFace |

### 3.5 Dataset e Preprocessing
| Libreria | Uso | Note |
|----------|-----|------|
| **Polars** | Elaborazione corpus Wikipedia (13 GB) | DataFrame in Rust con binding Python. Lazy evaluation + parallelismo nativo su tutti i core + streaming (non carica tutto in RAM). Usato per ricostruire articoli dal corpus DPR e ri-segmentazione sentence-aligned. |
| **datasets** (HuggingFace) | Caricamento NQ-open e Wikipedia | `datasets.load_dataset()` |
| **pandas** | Manipolazione dati tabulari | Preprocessing e analisi esplorativa |
| **pyarrow** | Formato colonnare in memoria | Backend per pandas/datasets/Polars |
| **kagglehub** | Download dataset Kaggle | ~~Usato nel vecchio notebook per Wikipedia dump~~ — non più necessario, corpus preso dal repo Silvestri |

### 3.6 Utility
| Libreria | Uso | Note |
|----------|-----|------|
| **tqdm** | Progress bar | Per loop lunghi |
| **logging** / Logger custom | Logging operazioni | Il vecchio notebook aveva una classe Logger custom; valuteremo se tenerla o usare `logging` standard |

---

## 4. Osservazioni sull'Implementazione

### 4.1 Corpus: evoluzione delle scelte
- **Vecchio notebook** (`base/preprocessing.ipynb`): era su Google Colab, usava dataset Kaggle `jjinho/wikipedia-20230701` (~442K articoli) e HuggingFace `HuggingFaceFW/clean-wikipedia`. Rimane come **riferimento storico**, non più eseguito.
- **Cambio corpus (2026-02-24)**: il dataset Kaggle 2023 è stato **scartato** perché non allineato con NQ-open (le cui risposte provengono da Wikipedia Dec 2018). Inizialmente sostituito con `psgs_w100.tsv` (corpus DPR pre-segmentato dal repo Silvestri).
- **Passaggio a HuggingFace (2026-03-10)**: scoperto che Silvestri et al. ([repo GitHub](https://github.com/florin-git/The-Power-of-Noise)) pubblicano il corpus come dataset HF `florin-hf/wiki_dump2018_nq_open` — articoli interi con gold documents NQ integrati e deduplicati. Questo elimina la necessità di ricostruire articoli da passaggi DPR e rimuovere padding. Il notebook è stato semplificato di conseguenza.
- **Dual-corpus strategy abbandonata (2026-03-10)**: inizialmente si pensava di mantenere sia il corpus DPR meccanico sia quello sentence-aligned per un'ablation study. Con il passaggio a HF (articoli interi), ricreare il taglio meccanico DPR richiederebbe un secondo splitter ad hoc — lavoro extra non previsto dal proposal. Si procede solo con segmentazione sentence-aligned.

### 4.2 Sentence-aligned segmentation (decisione 2026-02-25)
- **Problema riscontrato**: il taglio meccanico DPR a 100 parole causa chunk senza soggetto esplicito (es. l'articolo PAEEK ha un chunk che inizia con "he Cyprus Basketball Federation..." — il soggetto è nel chunk precedente). Questo penalizza entity linking (ReFiNed) e comprensione LLM.
- **Nota storica**: nel flusso precedente (partendo da `psgs_w100.tsv`) era necessario rimuovere il padding DPR dall'ultimo chunk di ogni articolo prima della segmentazione. Con il passaggio al dataset HF (articoli interi) questo step non è più necessario.
- **Strategia di segmentazione scelta**: applicare la stessa filosofia DPR a ogni segmento:
  1. Selezionare frasi complete finché `total_words < 100`
  2. Paddare lo spazio rimanente (`100 - total_words` parole) con le prime parole del primo segmento dell'articolo
  3. Risultato: esattamente 100 parole per segmento, frasi intere, padding contestuale
- **Perché padding dal primo segmento**: coerente con DPR originale (che fa lo stesso sull'ultimo chunk). Dà a ogni passaggio un'ancora esplicita sull'identità dell'articolo.
- **Alternative scartate**:
  - Lunghezza variabile (80-120 parole senza padding): perde la proprietà di lunghezza fissa
  - Prefisso "From TITLE:" sintetico: Contriever non è stato addestrato su questo formato, effetto imprevedibile sugli embedding
  - Titolo come campo separato (metadato): elegante ma non confrontabile con DPR che inietta il titolo nel testo

### 4.3 Scelte architetturali
- **Perche solo prossimita topologica (3-hop) e non semantica del grafo**: semplifica l'implementazione e testa un'ipotesi specifica — la struttura del grafo da sola e informativa per distinguere documenti rilevanti da distrattori
- **Perche alpha = 0.5**: peso uguale a segnale semantico e segnale topologico come punto di partenza; potenzialmente tunable
- **Perche top-100 -> top-5**: 100 documenti danno margine sufficiente per il reranking; 5 e il contesto tipico per LLM

### 4.4 Parallelizzazione file-based per la ri-segmentazione
- **Problema**: `Pool.imap` con dati testuali richiede di serializzare (pickle) ~11 GB di testo verso i worker e ~11 GB di risultati indietro. Il pickle è single-threaded nel processo main e domina il tempo di esecuzione.
- **Soluzione scelta**: approccio shared-nothing basato su file. Il corpus viene partizionato in 100 frammenti TSV su disco (~32K articoli ciascuno). I worker ricevono solo un indice intero via IPC (~400 bytes totali!), leggono/scrivono file indipendentemente.
- **Vantaggi**: zero pickle di dati testuali; resumability gratis (worker skippa se output esiste già); load balancing dinamico via `Pool.imap` su 100 task.
- **Alternative scartate**:
  - `Pool.imap` con dati via pipe: ~22 GB di pickle, main process collo di bottiglia
  - Macro-batch `Pool.map` (1 chunk/core): meno IPC calls ma stessa quantità di dati serializzati
  - Shared memory (`multiprocessing.shared_memory`): richiede serializzazione manuale di stringhe in buffer raw, complessità alta senza guadagno proporzionale

### 4.5 ReFiNed V1 — problemi di compatibilità e workaround
- **Installazione**: ReFiNed non è su PyPI. Si installa da GitHub: `pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip`
- **Bug 1 — `strftime("%s")` su Windows**: il downloader S3 (`refined/resource_management/aws.py`) usa `strftime("%s")` che è un'estensione Unix-only. Su Windows causa `ValueError: Invalid format string`. **Fix**: monkey-patch a runtime che sovrascrive `S3Manager.download_file_if_needed` usando `.timestamp()` (cross-platform).
- **Bug 2 — `add_special_tokens` con transformers recenti**: ReFiNed passa `add_special_tokens=False` come kwarg a `AutoTokenizer.from_pretrained()` in più punti (`general_utils.py:127`, `data_lookups.py:80`). Nelle versioni recenti di `transformers` (≥4.x), `add_special_tokens` è un metodo del tokenizer e passarlo come kwarg causa `AttributeError`. **Fix**: patch sorgente che rimuove il kwarg.
- **Bug 3 — `re.compile()` senza raw string**: `loaders.py` usa escape sequences in pattern regex senza `r"..."`, causando `SyntaxWarning` in Python 3.12+. **Fix**: patch sorgente che aggiunge il prefisso `r`.
- **Entity set `wikidata` vs `wikipedia`**: `entity_set="wikidata"` scarica ~20 GB di embeddings pre-calcolati per 33M entità. `entity_set="wikipedia"` (~6M entità) è molto più leggero (~9 GB totali) e sufficiente per NQ-open (tutte le risposte provengono da Wikipedia). Restituisce comunque QID Wikidata.
- **Cache locale**: i dati del modello vengono salvati in `data/refined_cache/` (gitignored) per evitare re-download.
- **Patch applicati a livello sorgente** tramite `scripts/patch_refined.py` — lo script modifica direttamente i file installati nella venv. Va ri-eseguito dopo `uv sync` o reinstallazione del pacchetto. Il notebook `nq_filtering.ipynb` invoca lo script automaticamente prima del caricamento del modello.

### 4.6 Rate limiting SPARQL
- L'endpoint Wikidata ha limiti di rate. Sara necessario:
  - Caching aggressivo delle query gia fatte
  - Batch delle richieste dove possibile
  - Eventuale download di un dump Wikidata locale per query intensive

---

## 5. Flusso Operativo

```
[NQ-open dataset] ──> Query Filtering (ReFiNed + token count)
                            │
                            ▼
                    [Filtered Queries]
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   [Wikipedia Corpus]                [Entity Linking]
         │                               │
         ▼                               ▼
   [FAISS Index]                  [Wikidata SPARQL]
   (Contriever)                        │
         │                             ▼
         ▼                    [Local KG Subgraph]
   [Top-100 Retrieval]               │
         │                            │
         ├────── Baseline ──> Top-5 ──> Llama2-7B ──> Exact Match
         │                                                  │
         └──── KG Reranking ──> Top-5 ──> Llama2-7B ──> Exact Match
                    │                                       │
                    └── final_score = α·dense + (1-α)·kg    │
                                                            ▼
                                                    [Confronto Accuracy]
```

---

## 6. Stato Avanzamento

| Step | Stato | Note |
|------|-------|------|
| Corpus Preparation | Completato | Corpus da HF `florin-hf/wiki_dump2018_nq_open` (~21M articoli con gold NQ). Segmentazione sentence-aligned completata: 23,910,209 passaggi da 100 parole in `data/wikipedia_2018_sentence_aligned/psgs_w100_sentence.tsv` (14.5 GB). Approccio file-based shared-nothing (100 frammenti, ~22s su 24 core). |
| Query Filtering | **Completato** | Notebook `nq_filtering.ipynb`. Dataset `florin-hf/nq_open_gold` (83,104 query, 3 split uniti). Token filter ≤5 (Contriever tokenizer, ALL variants): 76,406 query. Entity linking ReFiNed (`questions_model`, entity_set `wikipedia`): 31,372 query con entità sia in domanda che in TUTTE le varianti risposta (41.1%). Output: `data/NQ_question/qa_all_entities.jsonl` (filtrate) + `qa_entities_general.jsonl` (tutte con entity info). |
| KG Subgraph Construction | Da fare | Notebook `wikidata_preparation.ipynb` da popolare |
| Baseline Contriever-only | Da fare | |
| KG-Enhanced Reranking | Da fare | |
| Evaluation | Da fare | |

---

## 7. Struttura del Progetto

```
dl-RAG-denseAndKG/
├── documents/
│   ├── DL Project proposal ... .pdf    # Proposal originale
│   ├── PROJECT_NOTES.md                # Questo documento
│   └── GLOSSARIO.md                    # Terminologia tecnica
├── base/
│   └── preprocessing.ipynb             # Vecchio notebook Colab (riferimento)
├── utils/
│   ├── __init__.py
│   └── text_processing.py             # segment_article, _init_file_worker, file_segment_worker
├── scripts/
│   └── patch_refined.py               # Patch sorgente per ReFiNed V1 (Windows + Python 3.12+ + transformers 4.x)
├── data/
│   ├── wikipedia_2018_clean/
│   │   ├── articles_clean.tsv          # Articoli interi da HF (cache locale, ~3.2M articoli)
│   │   └── ordered_fragments/          # 100 frammenti input per parallelizzazione
│   │       └── frag_{0..99}.tsv        # ~32K articoli ciascuno (title, text)
│   ├── wikipedia_2018_sentence_aligned/
│   │   ├── ordered_fragments/          # 100 frammenti output dei worker
│   │   │   └── frag_{0..99}.tsv        # passaggi flat (text, title)
│   │   └── psgs_w100_sentence.tsv      # Corpus sentence-aligned finale (id, text, title)
│   ├── NQ_question/
│   │   ├── qa_all_entities.jsonl       # 31,372 query filtrate (Q+A hanno entità)
│   │   └── qa_entities_general.jsonl   # 76,406 query post token filter (con entity info)
│   └── refined_cache/                  # Cache locale modello ReFiNed (~9 GB)
├── nq_filtering.ipynb                  # Step 2 — Query Filtering (token + entity linking)
├── wikidata_preparation.ipynb          # Notebook principale (Step 3+)
├── main.py                             # Entry point (da definire)
└── .venv/                              # Virtual environment locale
```

---

*Ultimo aggiornamento: 2026-03-20*