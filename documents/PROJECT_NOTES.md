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
- **Output**: `psgs_w100_sentence.tsv` (passaggi sentence-aligned da 100 parole) + indice FAISS shardato

### Step 1b.4 — Passage Encoding & FAISS Indexing
- **Input**: `psgs_w100_sentence.tsv` (23.9M passaggi, 14.5 GB)
- **Notebook**: `03_embedding.ipynb`
- **Processing**:
  1. Caricamento corpus con Polars (`pl.read_csv` con `schema_overrides`)
  2. Per ogni passaggio: concatenazione `title + " " + text` (stesso formato di Silvestri et al.)
  3. Encoding con Contriever su GPU (batch 512, mean pooling su token non-padding)
  4. Sharding: 9 shard da ~5M vettori (ultimo shard ~2M). Corpus effettivo: ~42M passaggi × 768 × 4B ≈ 129 GB, non entra in RAM/VRAM
  5. Costruzione indici `faiss.IndexFlatIP` (brute-force exact inner product)
- **Output** (in `data/faiss_index/`):
  - `shard_XX.npy`: embedding float32 (5M × 768)
  - `shard_XX_ids.npy`: mapping posizione → passage ID (per risalire al testo nel TSV)
  - `shard_XX.faiss`: indice FAISS serializzato
- **Nota chiave**: FAISS contiene solo vettori numerici, niente testo. Per recuperare il testo serve: posizione FAISS → `shard_ids` → passage ID → corpus TSV
- **Resume support**: shard già completati vengono skippati automaticamente
- **Storico**: inizialmente si usava `psgs_w100.tsv` (corpus DPR pre-segmentato da repo Silvestri), ricostruendo articoli con Polars e rimuovendo padding DPR. Ora si parte direttamente da articoli interi via HF.

### Step 2 — Query Filtering
- **Input**: `florin-hf/nq_open_gold` — NQ-open arricchito con gold documents da `wiki_dump2018_nq_open` (Silvestri et al.). 83,104 query totali (train 72,209 + validation 8,006 + test 2,889), split uniti.
- **Filtering** (pipeline a 2 stadi):
  1. **Token count ≤ 5** (tokenizer Contriever/BERT wordpiece, `add_special_tokens=False`): criterio ALL (tutte le varianti risposta devono passare). Risultato: 76,406 query (91.9%). Split answers (1.4%) sacrificate per coerenza.
  2. **Entity linking ReFiNed**: modello `questions_model` (fine-tuned WebQSP), entity_set `wikipedia` (~6M entità). Criterio: domanda ha ≥1 entità AND tutte le varianti risposta hanno ≥1 entità. Risultato: 31,372 query (41.1% delle 76,406).
- **Output**: `data/NQ_question/qa_all_entities.jsonl` (31,372 query filtrate con QID) + `data/NQ_question/qa_entities_general.jsonl` (76,406 query con entity info completa)
- **Notebook**: `02_nq_filtering.ipynb`

### Step 3 — KG Subgraph Construction (Wikidata Preparation)
- **Input**:
  - Query entities: `data/NQ_answer/queries_curated.jsonl` (1000 query, `question_qids` + `answer_variant_qids`)
  - Passage entities: `data/NQ_answer/passage_entities_curated.parquet` (90.667 passaggi unici, lista QID per passaggio)
- **Storage del grafo Wikidata**: **dump HDT pre-built locale** (`latest-all-06-Jan-2022.hdt`, ~166 GB) interrogato via `pyHDT` in WSL2. SPARQL endpoint pubblico abbandonato dopo prove empiriche di timeout sui count per iper-hub (vedi 4.7).
- **Processing**: BFS a 3 onde con deduplicazione globale dei nodi visitati. Per ogni seed l'helper `neighbors_q(doc, qid)` (in `scripts/hdt_query_test.py`) restituisce vicini Q-target uscenti+entranti, indicizzati per predicato.
- **Output**: Grafo locale (NetworkX) con archi `(src_qid, predicate, dst_qid)`, salvato come parquet incrementale onda-per-onda per resumability

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
- **Patch applicati a livello sorgente** tramite `scripts/patch_refined.py` — lo script modifica direttamente i file installati nella venv. Va ri-eseguito dopo `uv sync` o reinstallazione del pacchetto. Il notebook `02_nq_filtering.ipynb` invoca lo script automaticamente prima del caricamento del modello.

### 4.6 Rate limiting SPARQL
- L'endpoint Wikidata ha limiti di rate. Sara necessario:
  - Caching aggressivo delle query gia fatte
  - Batch delle richieste dove possibile
  - Eventuale download di un dump Wikidata locale per query intensive

### 4.7 SPARQL endpoint pubblico → dump HDT locale (decisione 2026-04-25)

**Piano iniziale**: lookup tramite `https://query.wikidata.org/sparql` con strategia "pre-onda count + onda estrazione", organizzata in fallback a 4 livelli (count globale → distinct predicates → per-predicato → skip nodo).

**Problema empirico verificato sull'endpoint pubblico**:
- `SELECT DISTINCT ?p WHERE { ?n ?p wd:Q5 }` → **timeout** (nemmeno la lista dei predicati distinti)
- `SELECT (COUNT(?n) AS ?c) WHERE { ?n wdt:P31 wd:Q5 }` → **timeout** (anche solo il count, query minimale, niente FILTER)

Q5 ("essere umano") come oggetto di P31 ha ~10M+ righe e il server pubblico non riesce a contarle entro il limite di 60s. Questo invalida tutto lo schema di fallback: per coppie *(hub, predicato-popolare)* anche il livello L4 (count per singolo predicato) fallisce.

**Pivot**: passaggio al dump HDT pre-built `latest-all-06-Jan-2022.hdt` (~166 GB), URL `https://hdt-dumps.cluster.ai.wu.ac.at/dumps/`. Lookup locali tramite `pyHDT` da Python in **WSL2/Ubuntu** (Windows nativo non supporta la build C++ del binding pulitamente). I count per pattern triple su HDT sono sub-millisecondi: vanno direttamente all'indice POS pre-costruito.

**Conseguenze sul piano**:
- Niente blacklist predicati popolari, niente fallback, niente rate limit
- BFS 3-hop con espansione completa (anche su Q5, Q30) diventa fattibile
- Tempo stimato per i ~100k seed: poche ore in totale (vs 1-2 giorni col pubblico)
- Snapshot gennaio 2022 va bene: ReFiNed è anch'esso 2022, le QID seed sono stabili, il grafo concettuale post-2022 non è rilevante per il task

**Alternative scartate**:
- **Triple store self-hosted** (Blazegraph/Virtuoso/QLever): l'utente l'aveva provato in passato, esploso a 5 TB in fase di ingestione (working space per la build degli indici 3x-5x il dataset finale)
- **Build HDT da `latest-truthy.nt.gz`**: 12-24h di build + disco transitorio. Il pre-built copre già i nostri scopi
- **Linked Data Fragments**: per pattern singoli funziona ma più lento, ed endpoint pubblico per Wikidata non sempre affidabile
- **`hdt` package PyPI su Windows nativo**: build C++ richiede Visual Studio Build Tools, deps native (`pybind11` non dichiarata correttamente nel `pyproject.toml` upstream, lib `libserd`/`libraptor`), troppo fragile

**Materiale archeologico** (esperimenti SPARQL precedenti, tenuti per riferimento ma non più nella pipeline):
- `data/Wikidata_service/queryPredicati.csv` — lista 13.431 predicati Wikidata
- `data/Wikidata_service/test_queries/query_NNNNN.sparql` — file di test count batched (100→5000 predicati su Q5)
- `scripts/build_test_queries.py` — generatore dei file sopra

### 4.8 Layer 1 export: bug pyHDT wildcard iterator → fix per-predicate (2026-04-27 / 2026-04-28)

**Sintomo**. La prima versione di `scripts/hdt_export.py` usava `doc.search_triples("", "", "")` (wildcard puro) per enumerare tutte le triple e filtrare in-loop le Q-Q `wdt:*`. L'iteratore ha mostrato due anomalie:

1. **Overshoot**: yieldava 8.85B+ triple contro le 3.65B dichiarate da `doc.total_triples` (header HDT). Probabile causa: l'header conta una sezione, l'iteratore wildcard traversa anche labels/descrizioni/reificazioni `wd:statement:*`/proprietà.
2. **Gap silenziosi**: pur in overshoot, MANCAVANO righe Q-Q `wdt:*` genuine. Ctrl+C al 242% di iterazione con buffer parziale (max 999.999 righe) non flushato.

**Diagnosi**. `verify_completeness.py` Phase 1 ha confermato che la API HDT *predicate-restricted* (`search_triples("", "wdt:Pxxx", "")`) restituisce conteggi **esatti** (`iter_total == hdt_count` per P7481/P3174/P2860). Il bug è solo nel wildcard puro su dump di questa dimensione.

**Fix**. Nuovo script `scripts/hdt_export_per_predicate.py` che enumera `P1...P15000` e per ogni `Pxxx` con count > 0 itera in modo bounded. Risultato (run ~1h):
- 661.471.158 righe Q-Q `wdt:*` (vs 661.000.000 del wildcard parziale, +471k recuperate)
- 1.431 predicati con dati
- iterazione deterministica, validabile per-predicato

Verifica empirica del nuovo parquet (DuckDB) sui 3 predicati sanity-checked: 1, 1.200, 284.766.906 — match esatto con `iter_qq` di Phase 1.

**File**. Lo script wildcard buggato (`scripts/hdt_export.py`) è stato **cancellato**. Il vecchio parquet è conservato come artifact storico in `data/db/edges_v1_wildcard_partial.parquet`; quello canonico è `data/db/edges.parquet` (versione per-predicato).

**Lesson learned**. Le API "wildcard scan" su dump grandi possono fallire silenziosamente. Quando un index permette di restringere via predicate/type, **preferire l'iterazione vincolata e validare con header counts**.

### 4.9 Layer 3 — gestione seed-hub: option A1 (degree-threshold seed-skip) (decisione 2026-04-28)

**Sintomo**. Primo dry-run di `scripts/build_n3.py` con threshold=5000, 100 seed sampled, 12 worker: tutti i seed hub-popolari (Q21=England, Q145=UK, …) andavano in `TIMEOUT` a 60s con `reach=0 ban=0`. Il guard `total_degree > threshold` esistente nel BFS si applicava ai *vicini* incontrati durante l'espansione, ma il seed *stesso* veniva sempre espanso → wave 1 di un seed mega-hub (es. Q30=USA, 2.3M edges) da sola supera 60s prima ancora di hit-tare il primo guard.

**Diagnosi**. Script diagnostico `scripts/seed_degree_stats.py` (DuckDB in-memory + JOIN su `node_stats.parquet`) sulla seed pool (1.416 distinct = unione `question_qids ∪ answer_variant_qids`):

- distribuzione: p50=71, p90=3.760, p95=17.064, p99=379.800, max=2.353.407 (Q30=USA)
- a soglia 5.000: **124 seed (8.8%)** sono hub
- top-30: tutti **paesi** (USA, UK, DE, FR, CN, IN, …) + Catholic Church, Paris, London

Estensione del diagnostic alla classificazione **per-query** usando solo `question_qids` (non answer-variants — al retrieval-time abbiamo solo la query, non la risposta):

- 1.000 query, **media 1.0 question_qids/query** (max 2) → ogni query è essenzialmente determinata dal suo unico concept di partenza
- a soglia 5.000: **822 `clean` (82.2%)**, 4 `mixed` (0.4%), **174 `all-hub` (17.4%)**

Asimmetria seed-vs-query: 8.8% di hub-seeds → 17.4% di query "tainted", perché i seed hub-popolari (Q30, Q145, Q142) compaiono come unico question_qid in molte query.

**Decisione — option A1: degree-threshold seed-skip**. Confronto con la letteratura (HippoRAG NeurIPS 2024, GraphSAGE NeurIPS 2017, PullNet EMNLP 2019, Resource Allocation Index Zhou et al. 2009, PPR family) ordinato per implementation effort:

1. **A1 — degree-threshold seed-skip** — 5 min, 3 righe (scelta)
2. PDB — predicate-direction blacklist (P17/P31/P131/P276 incoming) — 1-2h + tuning iterativo
3. Resource Allocation weighted overlap (`Σ 1/deg(z)` invece di `|∩|`) — 2-3h + downstream rewrite + serve A1 sotto comunque
4. PPR à la HippoRAG — giorni (sparse matrix, scoring continuo, downstream da rifare)

A1 è anche **prassi metodologica** in WebQSP (Yih et al. ACL 2016) e CWQ (Talmor & Berant NAACL 2018): split eval set per `clean`/`hub` subset e riportare metriche separate. Le 174 query `all-hub` riceveranno KG-score=0 → fallback denso puro.

**Implementazione**. Guard all'inizio di `bfs_3_waves`:

```python
seed_deg = get_degrees_batch([seed_qid], db).get(seed_qid, 0)
if seed_deg > threshold:
    seed_lbl = get_label(seed_qid, doc, label)
    return {}, [(seed_qid, seed_lbl, seed_deg, 0)]
```

I seed-hub vengono registrati in `banned_hubs.parquet` con `origin_qid == hub_qid` e `first_seen_dist=0` — distinguibili dai banned-mid-BFS via filtro SQL. Nuovo status worker `seed_hub` separato da `ok` e `timeout`, con marker `SKIPHUB` nel log per-seed e counter dedicato `n_seed_hub` nel summary finale.

**Limite onesto da riportare nel paper**: ~17% delle query ricadono nella categoria "popular entity" (paesi, religioni, città capitali) dove il KG-overlap con N3 a 3-hop non è discriminativo (qualsiasi passaggio mainstream "tocca" Italy / USA / UK). Per quelle query la pipeline ricade sul pure dense retrieval. Le metriche andranno riportate separatamente per `clean` (n=822) vs `all-hub` (n=174).

**Alternative scartate**:

- **A2 — drop hard delle query non-clean**: avrebbe ridotto eval set da 1000 a 822 e introdotto **selection bias** sistematico (rimuove proprio le query su entità popolari → KG-rerank apparirebbe migliore di quanto sia in produzione, hide failure mode su traffico real-world)
- **PDB — predicate-direction blacklist**: elegante ma richiede tuning iterativo dei predicati da bannare (P17/P31/P131/P276/P19/…) + non risolve seed estremi tipo Q30 con 506k incoming P17. Tenuto come **ablation v2**
- **PPR/HippoRAG**: rewrite completo (sparse matrix, scoring continuo), incompatibile con set-overlap metric scelta nel proposal
- **Resource Allocation weighted overlap**: serve A1 sotto comunque (hub estremi ammazzano BFS prima del weighting), poi `Σ 1/deg(z)` sostituisce `|∩|`. Tenuto come **ablation v3**

### 4.10 BFS-N3 abbandonato → switch a N1+per-pair (decisione 2026-04-28)

**Sintomo**. Dry-run di `scripts/build_n3.py` post-fix seed-hub guard (vedi §4.9), 100 seed sample, 12 worker, threshold=5000:

- 7/100 SKIPHUB (guard funziona — Q21, Q148, Q241, Q298, Q664, Q771, Q8222 — paesi/regioni/ONG)
- 4/100 OK (reach 3.6k-9.1k, t 5-52s — borderline anche quando "completa")
- **89/100 TIMEOUT** a 60s con `reach=0 ban=0`

I timeout colpiscono seed con degree well below 5000 (Q1558, Q119, Q23666, Q132616, Q4447, …). Sono entità "ordinarie" (persone, eventi, opere), non hub.

**Diagnosi**. Il problema non è il degree del seed, è la **somma esponenziale di wave 2-3**. Anche per seed con degree ~100-1000:

- Wave 1: 100-1000 nodi (ok)
- Wave 2: degree-batch lookup, hub-ban a 5000, ma i restanti 50-90% non-hub espandono ognuno ~100-1000 vicini → 10k-100k unici dopo dedup
- Wave 3: 10k × 100-1000 = milioni di triple HDT da iterare → 60s+ anche con I/O parallelo a 12 worker

L'hub-banning node-level cattura solo i casi estremi. Il **costo cumulativo** della BFS-3-onde su un grafo denso come Wikidata supera SEED_TIMEOUT_S per la maggioranza dei seed, indipendentemente dal degree del seed stesso.

**Decisione: switch ad architettura B (N1 precompute + per-pair check)**.

Insight chiave (intuizione dell'utente): la metrica `connected_ratio`/`purity_ratio` chiede *reachability booleana* (`∃ d ∈ D : dist(q,d) ≤ 3`), NON l'intero `N3(q)`. Computare l'intero N3 è overkill; per la metrica binaria basta:

- **dist=1**: `d ∈ N1(q)` — set lookup, O(1)
- **dist=2**: `N1(q) ∩ N1(d) ≠ ∅` — set intersection, O(min |N1|)
- **dist=3**: `∃ edge (x,y) : x ∈ N1(q), y ∈ N1(d)` — SQL indicizzata su `edges.parquet`

Costo:

- N1 precompute via DuckDB su `edges.parquet`: ~30-60 min (NO HDT, NO BFS)
- Per-pair check al scoring: ~1-5ms × ~500k coppie = 8-40 min
- Totale ~1-2h vs ~5-10h sperate (mai ottenute) della BFS-N3

**Multi-threshold ablation gratis**. La nuova architettura abilita ablation senza ricomputare N1: precompute N1 unfiltered, filtra al volo per soglie multiple `[500, 1000, 2000, 5000, 10000, ∞]`. Permette il claim del paper: "fino a t=X i seed-hub non causano degradazione metrica, oltre cambia così".

**Letteratura — questa è prassi standard per reachability su grafi grandi**:

- **Cohen et al., SODA 2003** — *Reachability and Distance Queries via 2-Hop Labels*. Base teorica del meet-in-the-middle: precomputi label per ogni nodo, query in O(1). I nostri N1 sono la versione "cheap" (1-hop label) di quel framework.
- **Bidirectional BFS** (CLRS) — costo `O(b^(d/2))` invece di `O(b^d)`. Per il nostro 3-hop dimezza l'esponente.
- **Lao & Cohen, ACL 2010 (PRA)** — query on-demand su KG senza precompute del subgraph completo.

**Cosa cambia operativamente**:

- `scripts/build_n3.py` deprecato (da rinominare `build_n3_BFS_DEPRECATED.py` per chiarezza, mantenuto per riferimento storico)
- Nuovo `scripts/build_n1.py` (Layer 3 alternativo, output `data/n1/n1.parquet`)
- Nuovo `scripts/ablation_diagnostic.py` (output `data/n1/ablation_summary.parquet` + `data/n1/ablation_invalidated_per_t.jsonl`)
- `scripts/kg.py` v2 (Layer 4) con per-pair check + multi-threshold

**Alternative scartate**:

- Aumentare SEED_TIMEOUT_S a 300s+: sposta il problema senza risolverlo. 1500 seeds × 300s ÷ 12 worker ≈ 10h non garantiti
- Cap esplicito su frontier wave-2 size: hack, non principled, non riproducibile
- Restare su BFS accettando 50%+ TIMEOUT: non viable, numeri non attendibili sul KG-rerank

**Lezione**. Per metriche binarie di reachability su grafo denso, la versione *space-eager* (BFS precompute completo) è categoricamente più costosa della *space-lazy* (precompute minimo, calcola al volo). Gli hub-seed sono gestiti naturalmente: non enumeriamo MAI tutti i 2.3M vicini di Q30, controlliamo solo "il `d` specifico è tra di essi?" — set lookup costante.

### 4.11 Layer 4 — query unificata min_dist + persistenza disco + fusione in utils/kg.py (decisione 2026-05-03)

**Contesto**. La prima versione di `scripts/kg.py` (Layer 4) restituiva `(Q_reached, D_reached)` per una sola configurazione `(threshold, max_distance=3)` per chiamata. Per ablation a griglia `distance × threshold` (3 × 6 = 18 celle) richiederebbe 18 query SQL per coppia (Q, D), con grosso overlap di lavoro: la query a `max_dist=3` esegue tutto quello che farebbe la query a `max_dist=2` e `max_dist=1`.

**Decisione iniziale (poi superata)**. Era stato creato `scripts/kg_advanced.py` con `KGScorerAdvanced(KGScorer)` per introdurre le ottimizzazioni in modo additivo. Pattern problematico: `from scripts.kg import ...` falliva quando lanciato standalone perché `scripts/` non è un package Python. Vedi REPO_INVENTORY §5.

**Decisione finale**. Fusione di entrambi in **`utils/kg.py`** con un'unica classe `KGScorer` self-contained. Niente eredità, niente cross-import. `utils/` è già un package quindi l'import `from utils.kg import KGScorer` funziona da qualunque punto del repo (notebook o terminale). Aggiunte due ottimizzazioni indipendenti:

**(1) Query unificata con min_dist**. Le 4 CTE (`d1`, `d2`, `d3a`, `d3b`) proiettano in più una colonna letterale `dist`; l'aggregato finale `MIN(dist) GROUP BY q, d` collassa cammini multipli sulla stessa coppia restituendo la distanza minima. Risultato: mappa `(q, d) → min_dist ∈ {1, 2, 3}`. In Python si deriva il risultato per ogni `max_distance ∈ {1, 2, 3}` filtrando `min_dist ≤ k` — zero lavoro SQL ripetuto sulla dimensione distanza.

Riduzione effettiva su griglia 3×6:
- Approccio loop: 18 query SQL → 6 × (1+2+4) = 42 esecuzioni di CTE
- Approccio min_dist: 6 query SQL (1 per threshold) → 6 × 4 = 24 esecuzioni di CTE
- Speed-up ~1.75x sul lavoro di scansione, ~3x sull'overhead di parsing/planning DuckDB

NB: la threshold cambia il filtro `WHERE neighbor_degree <= ?`, quindi 6 query distinte restano necessarie. Annotare anche `max_bridge_degree` per fattorizzare le threshold rischia esplosione di cardinalità intermedia — non implementato.

**(2) Persistenza su disco** (`data/kg.duckdb`). Il `__init__` ora supporta tre modalità:
- `db_path=None` → in-memory (fallback debug, comportamento di `KGScorer`)
- `read_only=True` → apre il file esistente in RO; errore se mancante
- default → RW, builda solo le tabelle non già presenti nel file (idempotente)

Prima init: ~5 min, ~10-15 GB su disco. Init successive: ~1s (skip rebuild). I worker MP aprono in `read_only=True`, condividono il page cache OS — niente RAM duplicata tra processi.

**API griglia**:
- `kg_components_grid(Q, D, distances, thresholds) -> pd.DataFrame` — single pair
- `kg_components_grid_batch(pairs, distances, thresholds) -> pd.DataFrame` — lista `(query_id, Q, passage_id, D)`, ritorna DataFrame "lungo" con colonne `[query_id, passage_id, distance, threshold, connected_ratio, purity_ratio, kg_score]` pronto per `pivot_table`.

**API single-configuration mantenute** per backward compatibility (debug, log per query): `connected_ratio`, `purity_ratio`, `kg_score`, `kg_components`. Tutte appoggiate sullo stesso helper `_reachable_pairs_min_dist` — nessuna duplicazione di SQL. `kg_score_multi` rimosso (sottocaso di `kg_components_grid` con `distances=(3,)`).

**Nota interpretativa**. A `distance=1` la threshold è inerte (no bridge, endpoint sempre preservati): tutte le righe `distance=1` saranno identiche al variare del threshold. A `distance=2` un solo bridge è soggetto a filtro; a `distance=3` due bridge → effetto threshold più pronunciato.

**File `scripts/kg.py` e `scripts/kg_advanced.py`**: rimossi. Storia preservata in git.

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
| FAISS Indexing | Completato | Notebook `03_embedding.ipynb`. Encoding con Contriever (batch 512, GPU) in 9 shard. Indici `IndexFlatIP` (exact inner product). Output in `data/faiss_index/shard_XX.{npy,faiss}` + `shard_XX_ids.npy`. |
| Query Filtering | Completato | Notebook `02_nq_filtering.ipynb`. Dataset `florin-hf/nq_open_gold` (83,104 query, 3 split uniti). Token filter ≤5 (Contriever tokenizer, ALL variants): 76,406 query. Entity linking ReFiNed (`questions_model`, entity_set `wikipedia`): 31,372 query con entità sia in domanda che in TUTTE le varianti risposta (41.1%). Output: `data/NQ_question/qa_all_entities.jsonl` (filtrate) + `qa_entities_general.jsonl` (tutte con entity info). |
| Answer preparation + curation | Completato | Notebook `answer_preparation.ipynb` (top-100 retrieval per query, 1000 query subset) + `answer_curation.ipynb` (sostituzione query con passaggi a 0 entità) + `apply_curation.ipynb` (apply 344 sostituzioni). Output: `data/NQ_answer/{queries_curated.jsonl, top100_curated.parquet, passage_entities_curated.parquet, query_embeddings_curated.npy}` — 1000 query, 100k righe top-100, 90.667 passaggi unici tutti con ≥1 entità. |
| **KG Subgraph Construction** | **In corso** | Layer 1 (edges.parquet 661.5M righe) + Layer 1.5 (node_stats.parquet con top hub Q13442814/Q1860/Q5) completati al 2026-04-28. Layer 1.6 (labels) in lancio. Bug wildcard iterator e recovery per-predicate documentati in §4.8. Pendenze attive in §6.1. |
| Baseline Contriever-only | Da fare | |
| KG-Enhanced Reranking | Da fare | Userà `connected_ratio` e `purity_ratio` su set di QID a distanza ≤ k via Layer 4 (vedi 6.3). |
| Evaluation | Da fare | |

### 6.1 Pendenze attive — Step 3 (KG Subgraph Construction)

Stato al **2026-04-25**, in ordine dipendenza:

1. **[fatto, 2026-04-25]** Download dump HDT `latest-all-06-Jan-2022.hdt` (~166 GB), sorgente `https://hdt-dumps.cluster.ai.wu.ac.at/dumps/`. File spostato dentro la repo in `data/Wikidata_service/latest-all-06-Jan-2022.hdt` (path gitignored). Accessibile da WSL2 via `/mnt/c/Users/Utente/Documents/PycharmProjects/dl-RAG-denseAndKG/data/Wikidata_service/`.
2. **[fatto, 2026-04-25]** Setup WSL2 + Ubuntu 24.04 LTS (noble). 31 GB RAM, 24 core, Python 3.12.3 di sistema. File HDT raggiungibile da `/mnt/c/...` con read+write OK.
3. **[fatto, 2026-04-25]** Setup pyHDT in Ubuntu. **Recipe** (da memorizzare per riproducibilità):
   ```bash
   sudo apt install -y build-essential cmake libserd-0-0 libserd-dev python3-dev
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
   uv venv ~/.venvs/dl-rag-wsl --python 3.12 && source ~/.venvs/dl-rag-wsl/bin/activate
   uv pip install --upgrade pip wheel setuptools pybind11
   CXXFLAGS="-include cstdint" CFLAGS="-include cstdint" \
     uv pip install hdt --no-build-isolation --no-cache
   python -c "from hdt import HDTDocument; print('ok')"
   ```
   Due workaround necessari per `hdt 2.3` (PyPI, packaging non aggiornato dal 2018):
   - `pybind11` non dichiarato in `build-system.requires` → installarlo prima e usare `--no-build-isolation`
   - `hdt-cpp 1.3.3` (vendored) usa `uint64_t` senza `#include <cstdint>` → GCC 13 non lo include più transitivamente → flag `-include cstdint` come pre-include forzato
4. **[fatto, 2026-04-25]** Smoke test: scaricato index pre-built `latest-all-06-Jan-2022.hdt.index.v1-1` (115 GB, da `https://hdt-dumps.cluster.ai.wu.ac.at/dumps/`) per evitare build locale che andava OOM (working set indexer ≈ 30 GB anon-rss su 32 GB cap WSL2). Apertura HDT 30s prima volta, <1s successive. Numeri reali su Q42 (150 edges, healthy) e Q5 (9.7M incoming P31, mega hub) confermano necessità di filtraggio strutturale.
5. **[fatto, 2026-04-26]** Architettura 4-layer decisa, vedi §6.3.
6. **[fatto, 2026-04-28]** Layer 1 — `scripts/hdt_export_per_predicate.py`: enumera `P1...P15000`, per ogni `Pxxx` con count > 0 itera in modo bounded e filtra Q-Q `wdt:*`, scrive `data/db/edges.parquet`. Run completo ~1h, **661.471.158 righe** finali su 1.431 predicati. La prima versione (wildcard) è stata cancellata dopo aver scoperto bug di overshoot + gap silenziosi: storia in §4.8.
7. **[fatto, 2026-04-28]** Layer 1.5 — `scripts/node_stats.py`: degree in/out per ogni QID, output `data/db/node_stats.parquet`. Top-10 hub coerenti con la struttura attesa di Wikidata: Q13442814 scholarly article (37.4M in_degree), Q1860 EN, Q5 human, Q1264450 (da identificare via labels), Q6581097 male, Q4167836 Wikimedia category, Q16521 taxon, Q523 star, Q7432 species, Q30 USA. Out-degree ≤ 450 per gli hub (tipico delle "type entities").
8. **[in corso]** Layer 1.6 — `scripts/build_labels.py`: lookup `rdfs:label@en` per ogni QID di interesse, scrive `data/db/labels.parquet`. ~5-10 min.
9. **[abbandonato 2026-04-28]** Layer 3 BFS-N3 — `scripts/build_n3.py`: dry-run con 100 seed, threshold=5000, 12 worker, seed-hub guard attivo (vedi §4.9) → **89/100 TIMEOUT** (4 OK, 7 SKIPHUB) per esplosione cumulativa wave 2-3 anche su seed non-hub. Decisione: switch ad architettura N1+per-pair (vedi §4.10). Script da rinominare `build_n3_BFS_DEPRECATED.py` per chiarezza, mantenuto come riferimento storico.
10. **[da fare]** Layer 3 N1 — `scripts/build_n1.py`: per ogni QID in `seeds ∪ passage_entities` (~140k), estrai `N1` da `edges.parquet` via DuckDB JOIN. Output `data/n1/n1.parquet` (schema long: `qid`, `neighbor`, `neighbor_degree`). Nessun HDT, nessuna BFS. Tempo stimato: ~30-60 min su una sola pass DuckDB.
11. **[bloccato da 10]** Layer 3 ablation — `scripts/ablation_diagnostic.py`: per ogni threshold in `[500, 1000, 2000, 5000, 10000, ∞]`, classifica le query (clean/mixed/all-hub) usando solo `question_qids` (vedi §4.9) + statistiche `|N1_filtered|`. Output:
    - `data/n1/ablation_summary.parquet` — una riga per threshold con counts e medie
    - `data/n1/ablation_invalidated_per_t.jsonl` — per ogni threshold, lista delle query invalidate con `question_qids`, `max_degree`, `labels`
12. **[bloccato da 10,11]** Layer 4 v2 — `scripts/kg.py`: modulo runtime con `connected_ratio(Q, D, threshold)` e `purity_ratio(Q, D, threshold)`. Carica `n1.parquet` in dict in-memory, filtra al volo per threshold, edge-probe via DuckDB su `edges.parquet` per `dist=3`. Multi-threshold scoring built-in.
13. **[bloccato da 12]** Pulizia residuale: rimuovere `pybind11` e `hdt` dal `pyproject.toml` Windows se rimasti (l'architettura B non usa più HDT al di fuori di Layer 1/1.5/1.6 già completati).

**Decisione architetturale 2026-04-28** — workflow `build_labels.py` ↔ `build_n3.py`. Versione precedente prevedeva: (1) `build_labels.py` su dataset QIDs; (2) `build_n3.py` produce `banned_hubs.parquet` con `hub_label` da `labels.parquet`; (3) re-run `build_labels.py` per estendere con hub QIDs. Problemi: al passo (2), la maggior parte degli hub (Q5, Q4167836, ecc.) non sono dataset QIDs → `hub_label = None` → re-run di (3) NON popola comunque la colonna in `banned_hubs.parquet` (servirebbe un JOIN downstream). **Fix**: `build_n3.py` ora cerca `rdfs:label@en` direttamente da HDT quando un hub non è in `labels.parquet` (cache warm), e memoizza il risultato. Costo aggiuntivo per run: <1s su poche centinaia di hub deduplicati. `build_labels.py` semplificato: rimossa `collect_banned_hub_qids` e dipendenza da `banned_hubs_*.parquet`. Workflow ora lineare: build_labels una volta, build_n3 una volta, fine.

### 6.2 Decisioni architetturali (chiuse)

- **~~Strategia di pruning~~ (deprecato 2026-04-28, vedi §4.10)**: ~~hub-banning a expansion-time via degree threshold (default 5000) durante BFS-3-onde~~. Approccio abbandonato dopo dry-run con 89% TIMEOUT.
- **Architettura B — N1 precompute + per-pair check (decisa 2026-04-28)**: per ogni QID in `seeds ∪ passage_entities`, precompute solo `N1` (1-hop neighborhood) da `edges.parquet`. Al scoring, reachability `dist(q,d) ≤ 3` calcolata on-demand via meet-in-the-middle: dist=1 lookup, dist=2 set-intersection, dist=3 SQL edge-probe su `edges.parquet`. Hub-handling naturale (hub-seed non enumerati, controlliamo solo membership). Multi-threshold ablation built-in.
- **Multi-threshold ablation (decisa 2026-04-28)**: per ogni threshold in `[500, 1000, 2000, 5000, 10000, ∞]` calcoliamo `connected_ratio_t` e `purity_ratio_t` filtrando `N1` al volo. File di output dedicati a ablation_summary + ablation_invalidated_per_t.
- **Multi-distance**: schema Layer 3 include `min_distance ∈ {1,2,3}`. Permette esperimenti su distanza 1/2/3 senza ricomputare BFS.
- **Direzione**: non-direzionale (in+out collassati nello stesso BFS).
- **Filtro target**: solo Q-entity vere (escludendo statement nodes `Q42-uuid`).
- **Operazioni runtime**: set-theoretic — `connected_ratio` e `purity_ratio` (vedi §6.3.4).
- **Niente seed-class blacklist**: i seed vengono da entity linking ReFiNed sui passaggi → individui, non classi. Validazione runtime via inspection del file `banned_hubs_*.parquet` (se vediamo classi tra gli hub frequenti, le possiamo blacklistare ex-post).
- **Niente predicate-direction blacklist**: scartata in favore del hub-banning per maggiore trasparenza ("vedo cosa escludo, non escludo per regola"). Tenuta come ablation v2 se serve recuperare il 17% di query `all-hub`.
- **Seed-hub skip — option A1 (decisa 2026-04-28)**: se il seed stesso ha `total_degree > threshold`, BFS skip immediato (return empty N3 + 1 banned row con `origin == hub_qid`, `first_seen_dist=0`). Per le ~17% query interessate (174/1000 a soglia 5000), KG-score=0 → fallback dense-only. Alternativa "drop hard delle query non-clean" scartata per **selection bias** (avrebbe rimosso sistematicamente le query su entità popolari, gonfiando i numeri KG-rerank). Confronto con la letteratura (HippoRAG/PPR, Resource Allocation, PDB) ordinato per effort in §4.9.

### 6.3 Architettura 4-layer KG (Step 3 + Step 4)

```
Layer 1     data/db/edges.parquet                       # 661.471.158 Q-Q wdt:*, una tantum (~1h via per-predicate)
Layer 1.5   data/db/node_stats.parquet                  # qid, in_deg, out_deg, total_deg
Layer 1.6   data/db/labels.parquet                      # qid, label_en
Layer 3a    data/n1/n1.parquet                          # qid, neighbor, neighbor_degree (~140k qids × ~N1 size)
Layer 3b    data/n1/ablation_summary.parquet            # threshold, n_clean, n_mixed, n_all_hub, mean_n1, ...
Layer 3c    data/n1/ablation_invalidated_per_t.jsonl    # per threshold: lista query invalidate con dettagli
Layer 4     scripts/kg.py v2                            # connected_ratio(Q,D,t), purity_ratio(Q,D,t), edge-probe SQL
```

**[abbandonato]** ~~Layer 3 BFS-N3: hop_sets_tNNNN.parquet, banned_hubs_tNNNN.parquet~~ — vedi §4.10 per ragioni del switch.

**Layer 1** (data) è immutabile. Qualsiasi cambio di filtro/threshold ricomputa solo Layer 3 (~30 min).

**Layer 2** (view) è un costrutto SQL DuckDB: `CREATE VIEW factual_edges AS SELECT * FROM edges WHERE ...`. Non ha file fisico, è solo una proiezione. Usato per esperimenti di query (es. "neighbors esclusi predicate X").

**Layer 3** è precomputato per le entità del dataset (passage QIDs + question QIDs + answer QIDs). Schema con `min_distance` permette di simulare a runtime distanze 1, 2, o 3 senza ricomputare.

**Layer 4 v2** carica `n1.parquet` in `dict[qid, dict[neighbor, degree]]` (~500 MB-1 GB in RAM). Per `dist=3` apre una connessione DuckDB read-only su `edges.parquet`. Operazioni:

```python
def reachable_within_3(q: str, d: str, n1, edges_db, threshold: int) -> bool:
    """True iff dist(q, d) ≤ 3 nel grafo filtrato a deg ≤ threshold."""
    n1_q = {n for n, deg in n1.get(q, {}).items() if deg <= threshold}
    n1_d = {n for n, deg in n1.get(d, {}).items() if deg <= threshold}
    # dist=1
    if d in n1_q: return True
    # dist=2
    if not n1_q.isdisjoint(n1_d): return True
    # dist=3: edge-probe su edges.parquet
    return edges_db.execute(
        "SELECT 1 FROM edges WHERE subject = ANY(?) AND object = ANY(?) LIMIT 1",
        [list(n1_q), list(n1_d)]
    ).fetchone() is not None

def connected_ratio(Q: set[str], D: set[str], n1, edges_db, threshold: int) -> float:
    if not Q: return 0.0
    return sum(1 for q in Q if any(reachable_within_3(q, d, n1, edges_db, threshold) for d in D)) / len(Q)

def purity_ratio(Q: set[str], D: set[str], n1, edges_db, threshold: int) -> float:
    if not D: return 0.0
    return sum(1 for d in D if any(reachable_within_3(q, d, n1, edges_db, threshold) for q in Q)) / len(D)
```

Costo per query-doc pair:
- `dist=1`/`dist=2`: O(|N1|) set ops, microsecondi
- `dist=3`: ~1-5 ms per coppia con DuckDB indicizzato (solo se le precedenti falliscono)
- Multi-threshold scoring: stessa N1 in dict, basta riapplicare il filtro

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
├── utils/                              # Libreria importabile (package)
│   ├── __init__.py                     # Re-export: from utils import KGScorer
│   ├── text_processing.py              # segment_article, _init_file_worker, file_segment_worker
│   └── kg.py                           # Layer 4 — KGScorer unificato (persistenza data/kg.duckdb, griglia DataFrame, MP-ready). Fusione di ex scripts/kg.py + scripts/kg_advanced.py il 2026-05-03 (vedi §4.11)
├── scripts/                            # Script eseguibili, self-contained (mai cross-import; codice condiviso vive in utils/)
│   ├── patch_refined.py                # Patch sorgente per ReFiNed V1 (Windows + Python 3.12+ + transformers 4.x)
│   ├── build_test_queries.py           # [archeologia SPARQL] generatore test queries con VALUES variabili
│   ├── hdt_query_test.py               # Smoke test HDT (jupytext .py)
│   ├── hdt_export_per_predicate.py     # Layer 1 — export per-predicate Q-Q wdt:* da HDT (WSL only)
│   ├── verify_completeness.py          # Sanity check + comparison HDT counts vs parquet (WSL only)
│   ├── node_stats.py                   # Layer 1.5 — degree in/out per QID (Windows venv, polars streaming)
│   ├── build_labels.py                 # Layer 1.6 — lookup rdfs:label@en via HDT (WSL only)
│   ├── seed_degree_stats.py            # Diagnostic — distribuzione degree dei seed + classificazione clean/mixed/all-hub (Windows venv)
│   ├── build_n3.py                     # ~~Layer 3 BFS-N3~~ — DEPRECATO 2026-04-28 dopo 89% TIMEOUT (vedi §4.10)
│   ├── build_n1.py                     # Layer 3 N1 — precompute 1-hop neighborhoods via DuckDB (Windows venv)
│   └── ablation_diagnostic.py          # Layer 3 ablation — multi-threshold analysis su N1 (Windows venv)
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
│   ├── NQ_answer/
│   │   ├── queries_subset.jsonl        # 1000 query originali subset
│   │   ├── queries_curated.jsonl       # 1000 query post-curation (344 sostituite)
│   │   ├── top100_merged.parquet       # Top-100 originali per le 1000 query
│   │   ├── top100_candidates.parquet   # Top-100 per 5000 query candidate (pool curation)
│   │   ├── top100_curated.parquet      # Top-100 finale post-curation
│   │   ├── passage_entities.parquet    # Entità per passaggi delle 1000 query originali
│   │   ├── passage_entities_curated.parquet  # Entità per passaggi nel top-100 curato (90.667 unici)
│   │   ├── query_embeddings_curated.npy  # (1000, 768) Contriever embeddings
│   │   ├── curation_results.jsonl      # 344 mapping originale→sostituta
│   │   ├── shard_{00..08}/             # Shard di lavoro per top-100 retrieval (parquet)
│   │   ├── passage_entities/           # Chunk passage entities (parquet, originali)
│   │   └── curation_chunks/            # Chunk passage entities (parquet, candidati)
│   ├── Wikidata_service/               # [archeologia SPARQL] esperimenti pre-pivot HDT + dump HDT
│   │   ├── latest-all-06-Jan-2022.hdt          # Dump Wikidata HDT (~166 GB, gitignored)
│   │   ├── latest-all-06-Jan-2022.hdt.index.v1-1  # Indice HDT pre-built (~115 GB, gitignored)
│   │   ├── queryPredicati.csv                  # 13.431 predicati Wikidata con URI entity-form
│   │   └── test_queries/                       # query_NNNNN.sparql con VALUES da 100 a 5000 predicati
│   ├── db/                             # Output parquet della pipeline KG
│   │   ├── edges.parquet                       # Layer 1 — 661.471.158 triple Q-Q wdt:* (per-predicate)
│   │   ├── edges_v1_wildcard_partial.parquet   # Artifact storico, run wildcard parziale (661M, vedi §4.8)
│   │   ├── node_stats.parquet                  # Layer 1.5 — degree per QID
│   │   ├── labels.parquet                      # Layer 1.6 — label EN (in arrivo)
│   │   └── verification.json                   # Output di verify_completeness.py
│   ├── n1/                             # Output Layer 3 architettura B (vedi §4.10)
│   │   ├── n1.parquet                          # Layer 3a — qid, neighbor, neighbor_degree
│   │   ├── ablation_summary.parquet            # Layer 3b — counts e medie per threshold
│   │   └── ablation_invalidated_per_t.jsonl    # Layer 3c — query invalidate per threshold con dettagli
│   ├── kg.duckdb                       # Layer 4 — DuckDB persistito (n1+edges+indice), generato al primo init di KGScorer (vedi §4.11)
│   ├── refined_cache/                  # Cache locale modello ReFiNed (~9 GB)
│   └── faiss_index/                    # Output di 03_embedding.ipynb
│       ├── shard_XX.npy                # Embedding float32 (5M × 768 per shard)
│       ├── shard_XX_ids.npy            # Mapping posizione FAISS → passage ID
│       └── shard_XX.faiss              # Indice FAISS IndexFlatIP
├── 01_corpus_preparation.ipynb         # Step 0-1 — Download corpus HF + sentence-aligned segmentation (output: data/wikipedia_2018_sentence_aligned/psgs_w100_sentence.tsv)
├── 02_nq_filtering.ipynb               # Step 2 — Query Filtering (token + ReFiNed entity linking)
├── 03_embedding.ipynb                  # Step 1b.4 — Passage Encoding & FAISS Indexing
├── answer_preparation.ipynb            # Step 4 — Top-100 retrieval per query subset
├── answer_curation.ipynb               # Step 4.5 — Identificazione query sostituibili (passaggi 0-entity)
├── apply_curation.ipynb                # Step 4.5 — Apply 344 sostituzioni → file _curated.*
└── .venv/                              # Virtual environment locale Windows (uv)
```

**Nota infrastrutturale**: il dump HDT `latest-all-06-Jan-2022.hdt` (~166 GB) è in `data/Wikidata_service/` (gitignored), accessibile da WSL via `/mnt/c/Users/Utente/Documents/PycharmProjects/dl-RAG-denseAndKG/data/Wikidata_service/`. La pipeline KG gira **da WSL2/Ubuntu**, non dall'ambiente Python Windows: pyHDT richiede compilazione C++ che su Windows non collabora. Path resolution negli script avviene via walk-up fino a `pyproject.toml` — niente path hardcoded a username o posizione di clone.

---

*Ultimo aggiornamento: 2026-04-28 — **BFS-N3 abbandonato dopo dry-run con 89% TIMEOUT** (vedi §4.10). Switch ad architettura B: precompute solo `N1` (1-hop neighborhood) e check reachability `dist≤3` on-demand via meet-in-the-middle (dist=1/2 set ops, dist=3 SQL edge-probe). Multi-threshold ablation built-in. Nuovi script in arrivo: `scripts/build_n1.py` (Layer 3 N1) + `scripts/ablation_diagnostic.py` (analisi per threshold con file di output dedicati). `scripts/build_n3.py` deprecato, da rinominare `build_n3_BFS_DEPRECATED.py`.*