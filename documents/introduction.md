# Introduction — verifica delle affermazioni

> Sezione 1 di `relazione/bozza/main.tex`. Claim per lo più qualitativi: per ciascuno
> indico **dove** è fondato (paper, codice, note). I numeri citati qui sono verificati
> in `abstract.md` / `method.md` / `results.md`.

## Testo (parafrasi dei claim)

1. RAG fonda gli LLM su testo esterno, ma passaggi ad alto score possono essere **distrattori**
   (simili alla query ma senza risposta) e danneggiano il reader.
2. **Ipotesi**: un passaggio con entità *graph-close* alle entità della query è meno
   probabilmente un distrattore di uno scelto per sola similarità semantica.
3. A differenza di GraphRAG (che usa la semantica completa delle relazioni), testiamo il
   **segnale più debole** — pura prossimità entro *k* hop — per isolarne il contributo.
4. Contributi: (i) pipeline che naviga l'**intero** grafo Wikidata da dump HDT locale e
   decide reachability ≤3-hop con schema *meet-in-the-middle* che **non enumera mai** i
   vicinati degli hub; (ii) risultato negativo pulito con test McNemar esatti e spiegazione
   meccanicistica; (iii) tre raffinamenti metodologici su [cuconasu2024power].

## Verifica

| # | Claim | Fonte / come si verifica |
|---|-------|--------------------------|
| 1 | Distrattori degradano il reader | Tesi centrale di **The Power of Noise** (Cuconasu et al., SIGIR 2024); è anche il presupposto del progetto, `PROJECT_NOTES.md` §1 ("Research question"). |
| 2 | Ipotesi prossimità-KG | `PROJECT_NOTES.md` §1 ("Idea chiave") e proposal (`documents/DL Project proposal ... .pdf`). |
| 3 | Solo topologia, non semantica del grafo | Scelta deliberata documentata in `PROJECT_NOTES.md` §4.3 ("Perché solo prossimità topologica (3-hop) e non semantica del grafo"). |
| 4(i) | Intero Wikidata via HDT + meet-in-the-middle senza enumerare hub | `utils/kg.py` (`is_reachable`, `_reachable_pairs_min_dist`: dist=1 edge diretto, dist=2 bridge condiviso, dist=3 edge-probe N1×N1); decisione in `PROJECT_NOTES.md` §4.10 e §6.3. "Non enumera hub" ⇒ verifica di sola *membership*, mai materializzazione del vicinato (cfr. Q30 con 2.35M archi, `method.md` M2). |
| 4(ii) | Risultato negativo + McNemar esatti | Notebook `09_llm_judge.ipynb` §9; numeri in `results.md` (13/90, tutte peggiorative). |
| 4(iii) | Tre departure metodologiche | padding → `01_corpus_preparation.ipynb` / `PROJECT_NOTES.md` §4.2; prompt → `08_llm_eval.ipynb` §4.1; judge → `09_llm_judge.ipynb`. Dettaglio in `method.md`. |

## Punto aperto

- **Link al codice** (riga ~44 del `.tex`): placeholder `https://github.com/<USER>/dl-RAG-denseAndKG`.
  Da sostituire con l'URL reale del repository.