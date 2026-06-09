# Results — verifica dei dati

> Sezione 4 di `relazione/bozza/main.tex`. Tutto ricalcolato il 2026-06-08 con
> `documents/verify_report_numbers.py` (blocco RESULTS) dai verdetti per-query
> `data/NQ_answer/llm_eval/judgments_*.jsonl`.

## 1. Numeri nel testo

| Claim | Valore reale | Esito |
|-------|--------------|-------|
| baseline (dense) = 0.364, "in linea con ~0.36 di [cuconasu2024power]" | **0.364** (364/1000) | ✓. Verificato dal paper (arXiv:2401.14887): con Llama-2 + documenti correlati Contriever riportano **0.4068 / 0.3815 / 0.3626** (1/2/4 doc). Il nostro top-5 = 0.364 ≈ il loro 0.3626. **Correzione 2026-06-08**: il report NON dice più "mid-30% band" (frase non presente nel paper). |
| best cell = 0.372 (`α=0.3, dist 2, t=5k`) | **0.372** (`alpha_3_dist2_thr5k`), massimo sulle 90 celle | ✓ |
| "+0.8pp ≈ 8 queries" | 0.372 − 0.364 = **+0.008 = 8/1000** | ✓ |
| worst = 0.290 (`α=0.9, dist 3, t=5k`) | **0.290** (`alpha_9_dist3_thr5k`), minimo sulle 90 celle | ✓ |
| 13/90 significativi (McNemar+Bonferroni), tutte peggiorative | **13/90**, worse=13, better=0 | ✓ (vedi `abstract.md` §2.6) |
| "b+c ≈ 200" (mover count) | min=111, **median=200**, max=249 | ✓ |

## 2. Tabella 1 — cross-check cella per cella

Confronto valori `.tex` vs ricalcolo dai `judgments_*.jsonl` (tutti combaciano):

| α | d2·t500 | d2·t5k | d2·t∞ | d3·t5k | d3·t∞ |
|---|---------|--------|-------|--------|-------|
| 0.1 | 0.362 | 0.357 | 0.354 | 0.366 | 0.347 |
| 0.3 | 0.358 | **0.372** | 0.343 | 0.351 | 0.333 |
| 0.5 | 0.361 | 0.362 | 0.311 | 0.310 | 0.314 |
| 0.9 | 0.354 | 0.337 | 0.295 | <u>0.290</u> | 0.318 |

(grassetto = best globale; sottolineato = worst globale — coerenti col `.tex`.)

## 3. Claim meccanicistici

| Claim | Verifica |
|-------|----------|
| "Threshold gradient": a α=0.9, dist 2, accuracy cala monotòna da 0.354 (t=500) a 0.295 (t=∞) | sequenza ricalcolata **[0.354, 0.350, 0.341, 0.337, 0.329, 0.295]**, monotòna-decrescente = **True** |
| "At dist 1 the threshold is inert — all cells coincide" | per ogni α, le 6 threshold a dist=1 danno accuracy **identica** (verificato per α∈{0.1,0.3,0.5,0.7,0.9}) = **True** |
| "more α (KG weight) → worse" | colonne d2·t∞ e d3·t5k decrescono in α; minimo globale a α=0.9 | 

## 4. Test statistico (ricostruzione indipendente)

McNemar **esatto** = binomtest a due code su `b` (retrieval-only) vs `b+c` (discordanti),
una condizione vs `retrieval`; Bonferroni `α_B = 0.05/90 = 5.56e-4`.

```python
from scipy.stats import binomtest   # vedi verify_report_numbers.py [R6]
# significant = 13/90, tutte con delta_acc<0 (KG peggiore), 0 migliorative
# movers b+c: min 111, median 200, max 249
```

**Esito**: tutti i numeri della sezione Results sono confermati.