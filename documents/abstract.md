# Abstract — testo e verifica dei dati numerici

> Documento di tracciamento per la relazione (`relazione/bozza/main.tex`).
> Per ogni numero citato nell'abstract: **dove** nasce e **come** si verifica.
> Tutti i numeri sotto sono stati **ricalcolati dalle fonti grezze** il 2026-06-08
> con `documents/verify_abstract_numbers.py` (output in fondo), non copiati da printout.

---

## 1. Testo dell'abstract (sorgente LaTeX, verbatim da `main.tex`)

```latex
Dense retrieval can surface \emph{distractors}: passages semantically close to
the query that do not contain the answer, degrading RAG accuracy~\cite{cuconasu2024power}.
We test whether \emph{pure graph topology} on Wikidata can filter them, without
using relation semantics. We build a 661.5M-edge entity graph from a local HDT dump,
score query--document pairs by topological reachability (\texttt{connected\,$\times$\,purity}),
and fuse this with the dense score over a $\text{distance}\times\text{hub-threshold}\times\alpha$
grid (91 conditions), evaluated on 1000 NQ-open queries with an LLM judge. The result
is a clean \emph{negative}: the KG signal never beats the dense baseline (0.364) beyond
noise (best 0.372) and actively degrades accuracy as its weight grows (down to 0.290);
13/90 conditions are significant under McNemar+Bonferroni, all of them worse. We explain
the mechanism (reachability saturates on a dense graph) and document the engineering and
the methodological departures from~\cite{cuconasu2024power}.
```

I numeri da verificare: **661M archi**, **baseline 0.364**, **best 0.372**,
**worst 0.290**, **91 condizioni**, **1000 query**, **13/90 significativi (tutti peggiorativi)**.

---

## 2. Verifica per singolo dato

### 2.1 — "661M-edge entity graph"

- **Valore reale**: `661,471,158` archi Q–Q `wdt:*` (≈ 661.5M → arrotondato a 661M).
- **Fonte**: `data/db/edges.parquet` — output del Layer 1 della pipeline KG, prodotto da
  `scripts/pipeline/hdt_export_per_predicate.py` (export per-predicato dal dump HDT
  `latest-all-06-Jan-2022.hdt`). Cfr. `PROJECT_NOTES.md` §4.8 e §6.1.
- **Come si verifica** (lettura del solo footer del parquet, istantanea):

```python
import pyarrow.parquet as pq
print(pq.ParquetFile("data/db/edges.parquet").metadata.num_rows)
# -> 661471158
```

- **Nota**: lo schema è `[subject, predicate, object]`, una riga per arco. Il conteggio
  combacia col cross-check per-predicato di `verify_completeness.py` documentato in §4.8.

### 2.2 — "dense baseline (0.364)"

- **Valore reale**: `0.364` = `364 / 1000` (verdetti YES del judge sulla condizione `retrieval`).
- **Fonte**: verdetti per-query in `data/NQ_answer/llm_eval/judgments_retrieval.jsonl`
  (campo booleano `verdict_bool`), prodotti dal notebook `09_llm_judge.ipynb`.
  La condizione `retrieval` è il dense-only (top-5 Contriever, nessun KG), definita in
  `08_llm_eval.ipynb` §2.
- **Come si verifica** (accuracy = media dei verdetti, come nel groupby di `09` §8,
  righe ~1547-1555: `df_judge.groupby("condition")["verdict_safe"].agg([...])`):

```python
import json, pandas as pd
v = pd.read_json("data/NQ_answer/llm_eval/judgments_retrieval.jsonl", lines=True)
print(v["verdict_bool"].mean(), f'({v["verdict_bool"].sum()}/{len(v)})')
# -> 0.364 (364/1000)
```

### 2.3 — "best 0.372"

- **Valore reale**: `0.372`, cella `alpha_3_dist2_thr5k` (α=0.3 peso KG, distanza 2, hub-threshold 5k).
  È il **massimo** sulle 90 celle di rerank; nessuna cella supera 0.372.
- **Fonte**: `data/NQ_answer/llm_eval/judgments_alpha_3_dist2_thr5k.jsonl` (e tutte le altre
  `judgments_*.jsonl`), notebook `09_llm_judge.ipynb` §8.
- **Come si verifica** (max dell'accuracy su tutte le condizioni escluso `retrieval`):

```python
import pathlib, pandas as pd
d = pathlib.Path("data/NQ_answer/llm_eval")
acc = {p.stem.replace("judgments_", ""):
       pd.read_json(p, lines=True)["verdict_bool"].mean()
       for p in d.glob("judgments_*.jsonl")}
s = pd.Series(acc).drop("retrieval")
print(s.idxmax(), round(s.max(), 3))   # -> alpha_3_dist2_thr5k 0.372
```

### 2.4 — "down to 0.290" (worst)

- **Valore reale**: `0.290`, cella `alpha_9_dist3_thr5k` (α=0.9 peso KG, distanza 3, threshold 5k).
  È il **minimo** sulle 90 celle.
- **Fonte**: come 2.3 (`judgments_alpha_9_dist3_thr5k.jsonl`).
- **Come si verifica** (stesso `s` della 2.3):

```python
print(s.idxmin(), round(s.min(), 3))   # -> alpha_9_dist3_thr5k 0.290
```

### 2.5 — "91 conditions" / "1000 NQ-open queries"

- **91 condizioni** = 1 baseline (`retrieval`) + 90 celle di rerank (3 distanze × 6 threshold × 5 α).
  Definite in `08_llm_eval.ipynb` §2:

```python
ALPHAS_TO_TEST = [0.1, 0.3, 0.5, 0.7, 0.9]          # 5
# CELL_DIST_VARIANTS = [1, 2, 3]  (3)  ×  THR_VARIANTS = [500,1k,2k,5k,10k,inf]  (6)
# CONDITIONS = ["retrieval"] + 5 × 3 × 6 = 1 + 90 = 91
```
  Verifica diretta: `len(list(d.glob("judgments_*.jsonl")))` → **91**.
- **1000 query**: ogni `judgments_*.jsonl` ha 1000 righe (una per query del subset curato).
  Verifica: `len(pd.read_json(".../judgments_retrieval.jsonl", lines=True))` → **1000**.

### 2.6 — "13/90 conditions significant under McNemar+Bonferroni, all of them worse"

- **Valore reale**: `13 / 90` condizioni significative dopo correzione di Bonferroni
  (`alpha_B = 0.05/90 = 5.56e-4`); di queste **13 sono peggiorative** (delta_acc < 0) e
  **0 migliorative**.
- **Fonte**: notebook `09_llm_judge.ipynb` §9. Test = McNemar **esatto** (binomiale a due code
  sui discordanti `b`/`c`, via `scipy.stats.binomtest`), una condizione vs `retrieval`.
  - Bonferroni: `09` righe ~1869-1877 (`bonferroni_alpha = 0.05 / m`, `m = 90`).
  - Conteggio worse/better: `09` righe ~1999-2010.
- **Come si verifica** (ricostruzione del test dai verdetti per-query):

```python
import pathlib, pandas as pd
from scipy.stats import binomtest
d = pathlib.Path("data/NQ_answer/llm_eval")
V = {p.stem.replace("judgments_", ""):
     pd.read_json(p, lines=True).set_index("query_id")["verdict_bool"].astype(bool)
     for p in d.glob("judgments_*.jsonl")}
ret = V["retrieval"]; conds = [c for c in V if c != "retrieval"]; m = len(conds)
alpha_b = 0.05 / m; n_sig = n_worse = 0
for c in conds:
    a, b = ret.align(V[c], join="inner")
    bb = int((a & ~b).sum()); cc = int((~a & b).sum()); mv = bb + cc
    p = 1.0 if mv == 0 else binomtest(bb, mv, 0.5, alternative="two-sided").pvalue
    if p < alpha_b:
        n_sig += 1
        if b.mean() - a.mean() < 0: n_worse += 1
print(n_sig, "/", m, "| worse:", n_worse)   # -> 13 / 90 | worse: 13
```

---

## 3. Script di verifica unico + output reale

Tutti i controlli sopra sono raccolti in **`documents/verify_abstract_numbers.py`**.
Esecuzione e output (2026-06-08, ambiente `.venv` del progetto):

```
[1] edges.parquet num_rows = 661,471,158
    detected verdict column = 'verdict_bool'  (jsonl columns: ['query_id', 'condition', 'response', 'gold_answers', 'judge_raw', 'verdict_bool'])
[2] baseline (retrieval) accuracy = 0.364  (n_true=364/1000)
[3] BEST  rerank cell = alpha_3_dist2_thr5k    accuracy = 0.372
[4] WORST rerank cell = alpha_9_dist3_thr5k    accuracy = 0.290
    number of rerank conditions = 90
[5] Bonferroni alpha = 0.05/90 = 5.56e-04
    significant after Bonferroni = 13 / 90  (KG worse = 13, KG better = 0)
```

**Esito**: tutti i numeri dell'abstract sono confermati. Nota: in abstract si usa
"661.5M-edge" (cifra esatta 661,471,158), scelta il 2026-06-08 per precisione.