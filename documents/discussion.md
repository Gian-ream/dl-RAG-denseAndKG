# Discussion and conclusions — verifica

> Sezione 5 di `relazione/bozza/main.tex`. Numeri ricalcolati il 2026-06-08 con
> `documents/verify_report_numbers.py` (blocco DISCUSSION).

## 1. Saturazione della reachability ("connected_ratio ≈ 1")

Claim: "su un grafo denso come Wikidata, la reachability a 2–3 hop satura: quasi ogni
entità documento è raggiungibile da quasi ogni entità query, quindi `connected_ratio ≈ 1`
e lo score perde potere discriminante. Il filtro hub limita il danno ma non recupera segnale."

**Verifica** — `connected_ratio` per coppia (query, passaggio) a **dist=3**, da `kg_pairs_raw.parquet`,
al variare della threshold (∞ è codificato `0`):

| threshold | mean | median | frazione con connected_ratio = 1 |
|-----------|------|--------|----------------------------------|
| ∞ (0)  | **0.992** | 1.000 | **0.987** |
| 10000  | 0.952 | 1.000 | 0.947 |
| 5000   | 0.940 | 1.000 | 0.935 |
| 2000   | 0.920 | 1.000 | 0.914 |
| 1000   | 0.909 | 1.000 | 0.903 |
| 500    | 0.893 | 1.000 | 0.887 |

Lettura: senza filtro hub (∞), il **98.7%** delle coppie ha `connected_ratio` esattamente 1
→ saturazione confermata. Il filtro hub abbassa solo marginalmente (median resta 1.000 fino a
t=500) → "limita il danno ma non recupera segnale" confermato. Coerente col gradiente threshold
di `results.md` §3.

## 2. "About 17% of queries seed on popular entities" (all-hub)

Claim: ~17% delle query partono da entità popolari (paesi, religioni) dove ogni passaggio
mainstream "tocca" il seed → KG-score non informativo → fallback al dense.

**Verifica** — query in cui **tutti** i `question_qids` hanno `total_degree > 5000`
(stessa soglia di `PROJECT_NOTES.md` §4.9), da `queries_curated.jsonl` × `node_stats.parquet`:

- **174 / 1000 = 17.4%** → claim "~17%" confermato.

```python
# verify_report_numbers.py [D1]: per ogni query, all(deg(qid) > 5000 for qid in question_qids)
```

## 3. Semantica delle proprietà collassata

Claim: "la reachability binaria collassa la semantica: P31 (*instance-of*) e P19 (*born-in*)
contano allo stesso modo".

| Proprietà | Etichetta Wikidata | Uso nel claim |
|-----------|--------------------|---------------|
| `P31` | *instance of* | ✓ |
| `P19` | *place of birth* | ✓ ("born-in" è una glossa colloquiale; l'etichetta esatta è *place of birth*) |

Il punto è corretto: `is_reachable` (`utils/kg.py`) tratta ogni predicato `wdt:*` allo stesso
modo (nessun peso per tipo di relazione) — la reachability è puramente topologica.

## 4. Conclusione / future work

| Claim | Fonte |
|-------|-------|
| Future work = scoring relation-aware (PPR à la HippoRAG) o prossimità property-filtered | alternative già discusse in `PROJECT_NOTES.md` §4.9 (ablation v2/v3) come direzioni più costose ma "semantiche". |
| "valore = risultato negativo spiegato + tooling riusabile per navigare Wikidata" | sintesi del progetto (memoria `project_kg_rerank_negative_result`). |

**Esito**: tutti i claim numerici e fattuali della Discussion sono confermati.