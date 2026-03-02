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
- **Input**: `psgs_w100.tsv` — corpus DPR standard (Dense Passage Retrieval, Karpukhin et al. 2020)
  - Wikipedia dump **Dec 20, 2018**, pulito e segmentato in **~21M passaggi di 100 parole** (taglio meccanico, no sentence-alignment)
  - Formato TSV: colonne `id`, `text`, `title` (~13 GB)
  - Questo file è il **benchmark de facto** per open-domain QA: usato da Contriever, FiD, ATLAS, RAG, Silvestri et al.
  - Dec 2018 perché NQ-open è stato annotato su quella versione — le risposte gold provengono da lì
- **Fonte**: Repository del paper di Silvestri
- **Processing**: Caricamento passaggi → indicizzazione con Contriever embeddings
- **Output**: Indice FAISS con embeddings Contriever
- **Nota**: Inizialmente era stato caricato per errore il dataset Kaggle `jjinho/wikipedia-20230701` (Wikipedia 2023). Sostituito con il corpus DPR Dec 2018 per allineamento con NQ-open.

### Step 2 — Query Filtering
- **Input**: NQ-open dataset completo
- **Filtering**:
  - Risposte con <= 5 token
  - Sia domanda che risposta contengono entita Wikidata riconosciute da ReFiNed
- **Output**: Subset filtrato di query valide

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

### 4.1 Corpus: da Kaggle 2023 a Silvestri Dec 2018
- **Vecchio notebook** (`base/preprocessing.ipynb`): era su Google Colab, usava dataset Kaggle `jjinho/wikipedia-20230701` (~442K articoli) e HuggingFace `HuggingFaceFW/clean-wikipedia`. Conteneva analisi separatori `==...==`, classe Logger custom, funzioni batch processing. Rimane come **riferimento storico**, non più eseguito.
- **Cambio corpus (2026-02-24)**: il dataset Kaggle 2023 è stato **scartato** perché non allineato con NQ-open (le cui risposte provengono da Wikipedia Dec 2018). Sostituito con il corpus pre-split dal repository del paper di Silvestri.
- **Dual-corpus strategy (2026-02-24)**: il taglio meccanico DPR causa "semantic bleeding" (frasi spezzate tra chunk adiacenti), penalizzando entity linking e qualità del contesto per il LLM. Decisione: mantenere **entrambi** i corpus:
  - `psgs_w100.tsv` — DPR standard (baseline confrontabile con la letteratura)
  - `psgs_w100_sentence.tsv` — ri-segmentazione sentence-aligned (variante sperimentale)
  - Questo abilita un'ablation study: quanto miglioramento viene dalla segmentazione vs. dal KG reranking?
- **Perché Polars per la ricostruzione**: il corpus DPR è un TSV da ~13 GB / ~21M righe. Polars (scritto in Rust) offre lazy evaluation, parallelismo automatico su tutti i core, e streaming — processa il file senza caricarlo interamente in RAM. Alternativa pandas scartata: single-thread, richiederebbe ~26 GB di RAM per il DataFrame completo.

### 4.2 Sentence-aligned segmentation (decisione 2026-02-25)
- **Problema riscontrato**: il taglio meccanico DPR a 100 parole causa chunk senza soggetto esplicito (es. l'articolo PAEEK ha un chunk che inizia con "he Cyprus Basketball Federation..." — il soggetto è nel chunk precedente). Questo penalizza entity linking (ReFiNed) e comprensione LLM.
- **Artefatto padding DPR**: l'ultimo chunk di ogni articolo viene imbottito con parole dall'inizio dell'articolo (titolo + prime frasi) per raggiungere esattamente 100 parole. Questo padding va rimosso prima della ri-segmentazione.
- **Strategia di ri-segmentazione scelta**: applicare la stessa filosofia DPR a ogni segmento, non solo all'ultimo:
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

### 4.5 Rate limiting SPARQL
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
| Corpus Preparation | In corso | Ricostruzione articoli da DPR completata (Polars group_by, 3,232,908 articoli). Padding DPR rimosso con `multiprocessing.Pool.imap` (~87s su 16 core): 3,193,309 articoli con padding, 169M parole rimosse (8.1%). Articoli puliti salvati in `data/wikipedia_2018_clean/articles_clean.tsv` (11.2 GB). Ri-segmentazione sentence-aligned implementata con approccio file-based shared-nothing (zero IPC sui dati: 100 frammenti su disco, worker leggono/scrivono file). **Da eseguire** nel notebook. Output atteso: `data/wikipedia_2018_sentence_aligned/psgs_w100_sentence.tsv`. |
| Query Filtering | Da fare | |
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
│   └── text_processing.py             # strip_dpr_padding, segment_article, file_segment_worker
├── data/
│   ├── wikipedia_2018/
│   │   └── psgs_w100.tsv              # Corpus DPR originale (12.8 GB, ~21M passaggi)
│   ├── wikipedia_2018_clean/
│   │   ├── articles_clean.tsv          # Articoli puliti (11.2 GB, ~3.2M articoli)
│   │   └── ordered_fragments/          # 100 frammenti input per parallelizzazione
│   │       └── frag_{0..99}.tsv        # ~32K articoli ciascuno (title, text)
│   └── wikipedia_2018_sentence_aligned/
│       ├── ordered_fragments/          # 100 frammenti output dei worker
│       │   └── frag_{0..99}.tsv        # passaggi flat (text, title)
│       └── psgs_w100_sentence.tsv      # Corpus sentence-aligned finale (id, text, title)
├── wikidata_preparation.ipynb          # Notebook principale
├── main.py                             # Entry point (da definire)
└── .venv/                              # Virtual environment locale
```

---

*Ultimo aggiornamento: 2026-03-02*