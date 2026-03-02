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

## F

### FAISS (Facebook AI Similarity Search)
Libreria di Meta per ricerca efficiente di nearest-neighbor su vettori densi. Permette di indicizzare milioni di embeddings e trovare i piu simili a una query in tempi sub-lineari.

## H

### Hop (nel grafo)
Un "salto" lungo un arco del grafo. Se A e collegato a B e B a C, allora C e a 2-hop da A. Nel progetto usiamo 3-hop come soglia di prossimita.

## K

### Knowledge Graph (KG)
Grafo strutturato dove i nodi sono entita e gli archi sono relazioni tra esse. Wikidata e un KG aperto e collaborativo.

### KG Score
Score calcolato dalla topologia del grafo: `kg_score = connected_ratio * purity_ratio`. Usato per reranking dei documenti combinandolo con lo score denso.

## L

### Lazy Evaluation (Polars)
Modalità in cui le operazioni non vengono eseguite immediatamente, ma accumulate in un piano logico. Polars ottimizza il piano (riordina operazioni, elimina colonne inutili, fonde passaggi ridondanti) prima di eseguirlo. Analogo al query planner di un database SQL. Si attiva con `scan_csv()` (lazy) invece di `read_csv()` (eager).

## N

### NQ-open (Natural Questions Open)
Dataset di domande reali poste a Google, con risposte estratte da Wikipedia. Nella variante "open", il sistema deve trovare la risposta nell'intero corpus (non in un singolo documento dato).

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

### SPARQL
Linguaggio di query per grafi RDF/knowledge graph. Usato per interrogare l'endpoint Wikidata ed estrarre vicinati di entita.

## W

### Wikidata
Knowledge graph aperto e collaborativo gestito dalla Wikimedia Foundation. Contiene dati strutturati su milioni di entita (persone, luoghi, concetti) con relazioni tipizzate.

---

### psgs_w100.tsv
File TSV prodotto dal preprocessing DPR. Contiene ~21M righe, ciascuna un passaggio di ~100 parole da Wikipedia Dec 2018. Colonne: `id` (intero, identificativo passaggio), `text` (contenuto testuale), `title` (titolo dell'articolo di origine). È il corpus standard per NQ-open e benchmarks open-domain QA.

---

*Ultimo aggiornamento: 2026-02-24*