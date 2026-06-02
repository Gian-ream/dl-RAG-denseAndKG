# Glossario

Glossario vivo dei termini tecnici del progetto. Aggiornato man mano che si incontrano nuovi concetti.

---

## A

### Accuracy (Exact Match)
Metrica di valutazione: una risposta e corretta se contiene almeno una delle risposte gold del dataset NQ-open. Binaria: corretto/incorretto.

## C

### Connected Ratio
Componente del kg_score. Misura quante entita della query hanno almeno un'entita del documento raggiungibile entro 3-hop nel grafo Wikidata. Formula: `(# query entities con >= 1 doc entity entro 3-hop) / (# total query entities)`. Valore tra 0 e 1.

### Contriever
Modello di dense retrieval unsupervised sviluppato da Meta (facebook/contriever). Genera embeddings densi per passaggi e query, usati per calcolare similarita semantica tramite prodotto scalare o cosine similarity.

### curated (files `*_curated.*`)
Suffix marking the artifacts produced by the curation step (notebook 06). Curation **substituted** 344 of the 1000 queries (those whose retrieved passages had no linkable entities) with replacement queries from the NQ pool, and re-encoded their embeddings. The `*_curated.*` files (`queries_curated.jsonl`, `top100_curated.parquet`, `passage_entities_curated.parquet`, `query_embeddings_curated.npy`) form a **self-consistent set** indexed positionally 0..999. **Never mix** a `*_curated.*` file with its `*_subset.*` (pre-curation) counterpart — the positional `query_id` would silently point to a different query. See DATA_DICTIONARY for schemas.

## D

### DPR (Dense Passage Retrieval)
Paper di Karpukhin et al. (Facebook AI, 2020) che ha definito lo standard per l'open-domain QA. Ha introdotto: (1) un dual-encoder (query encoder + passage encoder) addestrato con in-batch negatives, (2) il corpus standard `psgs_w100.tsv` — Wikipedia Dec 2018 segmentata in ~21M passaggi di 100 parole. Questo corpus è il benchmark de facto riusato da quasi tutti i paper successivi (Contriever, FiD, ATLAS, RAG, Silvestri).

### Dense Retrieval
Approccio al recupero di documenti che usa rappresentazioni vettoriali dense (embeddings) per trovare documenti simili alla query nello spazio vettoriale, in contrasto con metodi sparse come BM25/TF-IDF.

### Distrattore (Distractor)
Documento che ha alta similarita semantica con la query (quindi viene recuperato dal dense retriever) ma NON contiene la risposta. E il problema centrale che questo progetto cerca di risolvere.

## E

### Embedding
Rappresentazione vettoriale densa di un testo (parola, frase, passaggio) in uno spazio continuo a N dimensioni. Testi semanticamente simili hanno embeddings vicini nello spazio.

### Entity Linking
Task NLP che consiste nel riconoscere menzioni di entita nel testo e collegarle alla voce corrispondente in una knowledge base (nel nostro caso, Wikidata QID).

### Entity Set (ReFiNed)
Insieme di entita candidate per il linking. ReFiNed offre due opzioni: `wikipedia` (~6M entita con pagina Wikipedia, ~9 GB) e `wikidata` (~33M entita, ~20 GB). Per NQ-open usiamo `wikipedia` perche tutte le risposte provengono da Wikipedia. Entrambi restituiscono QID Wikidata.

### edges (DuckDB table / `edges.parquet`)
The full set of Q-Q `wdt:*` triples extracted from the Wikidata HDT dump: 661M rows, schema `subject VARCHAR, object VARCHAR` (predicate discarded). Loaded into `kg.duckdb` as table `edges` (no index — DuckDB uses hash joins). Used by `utils/kg.py` for dist=3 reachability: it bridges the 1-hop neighborhoods of query and passage entities. Produced by `scripts/pipeline/hdt_export_per_predicate.py` (Layer 1). See DATA_DICTIONARY.

## F

### FAISS (Facebook AI Similarity Search)
Libreria di Meta per ricerca efficiente di nearest-neighbor su vettori densi. Permette di indicizzare milioni di embeddings e trovare i piu simili a una query in tempi sub-lineari. FAISS contiene **solo vettori numerici**, niente testo o metadati: per risalire al testo originale serve un mapping esterno (nel nostro caso i file `shard_XX_ids.npy`).

### FAISS IndexFlatIP
Tipo di indice FAISS che usa il prodotto interno (Inner Product) come metrica di similarita. "Flat" = ricerca brute-force esatta (confronta la query con tutti i vettori, nessuna approssimazione). Su vettori normalizzati, il prodotto interno equivale alla cosine similarity. Nel progetto usiamo un indice per shard (~5M vettori ciascuno).

## H

### Hop (nel grafo)
Un "salto" lungo un arco del grafo. Se A e collegato a B e B a C, allora C e a 2-hop da A. Nel progetto usiamo 3-hop come soglia di prossimita.

## J

### JSONL (JSON Lines)
Formato di file dove ogni riga e un oggetto JSON indipendente. Usato nel progetto per salvare i risultati dell'entity linking (`qa_all_entities.jsonl`, `qa_entities_general.jsonl`). Vantaggi: append-friendly, leggibile riga per riga senza caricare tutto in memoria.

### jupytext (percent format, `# %%`)
Tool that keeps a notebook in two paired files: a `.py` (source of truth) and a `.ipynb` (runnable). In the `py:percent` format, special comment markers delimit cells:
- `# %%` → start of a **code cell**
- `# %% [markdown]` → start of a **markdown cell** (the following `#`-prefixed lines are the rendered text)

We edit only the `.py`, then sync with `jupytext --to ipynb --update <file>.py` (never bare `--sync`, which can lose edits to a timestamp race with the IDE). This is why every notebook in the repo exists as both `0X_name.py` and `0X_name.ipynb`.

## K

### Knowledge Graph (KG)
Grafo strutturato dove i nodi sono entita e gli archi sono relazioni tra esse. Wikidata e un KG aperto e collaborativo.

### KG Score
Score calcolato dalla topologia del grafo: `kg_score = connected_ratio * purity_ratio`. Usato per reranking dei documenti combinandolo con lo score denso.

## I

### Inner Product (prodotto interno)
Operazione tra due vettori: somma dei prodotti elemento per elemento. Per vettori normalizzati (norma = 1), il prodotto interno coincide con la cosine similarity. Usato da FAISS (`IndexFlatIP`) come metrica di similarita tra embedding di query e passaggi.

## L

### Lazy Evaluation (Polars)
Modalità in cui le operazioni non vengono eseguite immediatamente, ma accumulate in un piano logico. Polars ottimizza il piano (riordina operazioni, elimina colonne inutili, fonde passaggi ridondanti) prima di eseguirlo. Analogo al query planner di un database SQL. Si attiva con `scan_csv()` (lazy) invece di `read_csv()` (eager).

### Mean Pooling
Tecnica per ottenere un singolo vettore embedding da una sequenza di token. Si fa la media dei vettori di tutti i token reali (escludendo il padding). Contriever usa mean pooling invece del token CLS (usato da BERT). Formula: somma dei vettori token / numero di token reali.

## M

### mmap (memory-mapped file)
OS technique to access an on-disk file *as if* it were an in-RAM array, without loading it whole. Only the pages you actually touch are read on-demand (lazy). Used in two places:
- `07_kg_rerank` loads each FAISS embedding shard with `np.load(..., mmap_mode="r")`: fancy-indexing the mmap reads only the ~30 MB of rows actually needed instead of the full ~1.6 GB shard.
- DuckDB (`kg.duckdb`) memory-maps its tables, so the OS page cache is shared across read-only worker processes (no RAM duplication).

Trade-off: tiny RAM footprint, but random scattered access causes page faults (slow if not roughly sequential). mmap views must be released (`del`) to free the mapping.

## N

### n1 (DuckDB table / `n1.parquet`)
Precomputed **1-hop adjacency list** of the Wikidata KG, restricted to the QIDs of `seeds ∪ passage_entities`. ~93M rows, schema `qid VARCHAR, neighbor VARCHAR, neighbor_degree UBIGINT` (+ B-tree index on `qid`). Each row = "entity `qid` is directly connected to `neighbor`, which has `neighbor_degree` total connections". It is the workhorse of all reachability queries in `utils/kg.py`:
- dist=1: is `d ∈ N1(q)`?
- dist=2: do `q` and `d` share a common neighbor (with degree ≤ threshold)?
- dist=3: `n1` gives the neighbors of `q` and `d`, `edges` bridges them.

The `neighbor_degree` column is exactly what the **threshold** filters (hub-banning): a bridge node is allowed only if `neighbor_degree ≤ threshold`. At dist=1 there is no intermediate node, so `neighbor_degree` is never consulted → **dist=1 is threshold-invariant**. Produced by `scripts/pipeline/build_n1.py` (Layer 3). See DATA_DICTIONARY.

### NQ-open (Natural Questions Open)
Dataset di domande reali poste a Google, con risposte estratte da Wikipedia. Nella variante "open", il sistema deve trovare la risposta nell'intero corpus (non in un singolo documento dato).

### nq_open_gold (florin-hf/nq_open_gold)
Dataset HuggingFace di Silvestri et al. che arricchisce NQ-open con gold documents estratti dal corpus `wiki_dump2018_nq_open`. 83,104 query totali (train 72,209 + validation 8,006 + test 2,889). Colonne: `question`, `answers` (lista), `text` (gold document), `example_id`, `idx_gold_in_corpus`.

## P

### Polars
Libreria DataFrame scritta in Rust con binding Python. Alternativa a pandas ottimizzata per performance: parallelismo nativo (usa tutti i core), lazy evaluation (ottimizza il piano di esecuzione prima di eseguirlo), streaming (processa file grandi senza caricarli interamente in RAM). Internamente usa Apache Arrow come formato in memoria.

### Purity Ratio
Componente del kg_score. Misura la "purezza" delle entita del documento rispetto alla query: quante entita del documento sono effettivamente vicine (entro 3-hop) a entita della query. Formula: `(# doc entities entro 3-hop di qualsiasi query entity) / (# total doc entities)`. Valore tra 0 e 1.

## Q

### QID (Wikidata)
Identificatore univoco di un'entita su Wikidata (es. Q42 = Douglas Adams, Q64 = Berlino). Formato: lettera Q seguita da un numero.

## R

### RAG (Retrieval-Augmented Generation)
Architettura che combina un retriever (che trova documenti rilevanti) con un generatore (LLM che produce la risposta usando quei documenti come contesto). Riduce le allucinazioni del LLM ancorandolo a fonti esterne.

### ReFiNed
Entity linker end-to-end sviluppato da Amazon Science. Riconosce menzioni di entita nel testo e le collega direttamente a Wikidata QID. Zero-shot capable (funziona su entita mai viste in training).

### Reranking
Processo di riordinamento dei documenti recuperati. Nel nostro caso: si recuperano top-100 con Contriever, poi si riordinano usando un final_score che combina score denso e kg_score, e si tengono i top-5.

## S

### Sharding (FAISS)
Strategia di partizionamento dell'indice FAISS in piu pezzi (shard) per gestire dataset che non entrano in memoria. Nel progetto: ~42M vettori (un embedding per passaggio del corpus sentence-aligned) × 768 dim × 4 bytes ≈ 129 GB, suddivisi in 9 shard da ~4.7M vettori (~14 GB ciascuno). A search time si carica uno shard alla volta su GPU.

### SPARQL
Linguaggio di query per grafi RDF/knowledge graph. Usato per interrogare l'endpoint Wikidata ed estrarre vicinati di entita.

## T

### Token Threshold (≤ 5)
Criterio di filtraggio per le query NQ-open: si tengono solo query le cui risposte hanno al massimo 5 token (misurati con il tokenizer Contriever/BERT wordpiece). Scopo: garantire che le risposte siano entita brevi e fattoriali, adatte al confronto con entita Wikidata.

## W

### WebQSP (WebQuestions Semantic Parses)
Dataset di entity linking su domande in linguaggio naturale. Usato per il fine-tuning del modello `questions_model` di ReFiNed, che risulta piu adatto a testo in stile domanda (lowercase, senza punteggiatura) rispetto al `wikipedia_model` addestrato su prosa Wikipedia.

### Wikidata
Knowledge graph aperto e collaborativo gestito dalla Wikimedia Foundation. Contiene dati strutturati su milioni di entita (persone, luoghi, concetti) con relazioni tipizzate.

### Wordpiece (tokenizzazione)
Algoritmo di tokenizzazione subword usato da BERT (e quindi da Contriever). Spezza parole rare in sotto-unita (es. "Calrissian" → ["cal", "##ris", "##sian"]). Il prefisso `##` indica continuazione della parola precedente. Rilevante nel progetto per il conteggio token delle risposte (soglia ≤ 5).

---

### psgs_w100.tsv
File TSV prodotto dal preprocessing DPR. Contiene ~21M righe, ciascuna un passaggio di ~100 parole da Wikipedia Dec 2018. Colonne: `id` (intero, identificativo passaggio), `text` (contenuto testuale), `title` (titolo dell'articolo di origine). È il corpus standard per NQ-open e benchmarks open-domain QA.

---

*Ultimo aggiornamento: 2026-05-20*