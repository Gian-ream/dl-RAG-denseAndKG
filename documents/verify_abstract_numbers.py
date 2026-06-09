"""Independent re-verification of the numeric claims in the report abstract.

Recomputes, from the raw per-query judge verdicts (NOT from cached printouts):
  - total Q-Q edges exported from the Wikidata HDT dump
  - dense-retrieval baseline accuracy
  - best / worst rerank-cell accuracy (over the 90 KG conditions)
  - number of conditions significant under exact McNemar + Bonferroni

Run:  .venv\\Scripts\\python.exe documents\\verify_abstract_numbers.py
"""
import json
import pathlib

import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLM = ROOT / "data" / "NQ_answer" / "llm_eval"


def load_jsonl(path: pathlib.Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


# 1) Total edges: read the parquet footer only (instant, no full scan).
edges_rows = pq.ParquetFile(ROOT / "data" / "db" / "edges.parquet").metadata.num_rows
print(f"[1] edges.parquet num_rows = {edges_rows:,}")

# 2) Per-condition accuracy from per-query verdicts (judgments_*.jsonl).
ret_cols = list(load_jsonl(LLM / "judgments_retrieval.jsonl").columns)
vcol = next(c for c in ("verdict_bool", "verdict_safe", "verdict", "correct")
            if c in ret_cols)
print(f"    detected verdict column = '{vcol}'  (jsonl columns: {ret_cols})")

verdicts: dict[str, pd.Series] = {}
acc = {}
for p in sorted(LLM.glob("judgments_*.jsonl")):
    cond = p.stem.replace("judgments_", "")
    df = load_jsonl(p)
    v = df.set_index("query_id")[vcol].astype(bool)
    verdicts[cond] = v
    acc[cond] = {"n_true": int(v.sum()), "n_total": int(len(v)),
                 "accuracy": float(v.mean())}

accdf = pd.DataFrame(acc).T
baseline = accdf.loc["retrieval", "accuracy"]
rerank = accdf.drop(index="retrieval")
best, worst = rerank["accuracy"].idxmax(), rerank["accuracy"].idxmin()
print(f"[2] baseline (retrieval) accuracy = {baseline:.3f}  "
      f"(n_true={int(accdf.loc['retrieval','n_true'])}/"
      f"{int(accdf.loc['retrieval','n_total'])})")
print(f"[3] BEST  rerank cell = {best:<22} accuracy = {rerank.loc[best,'accuracy']:.3f}")
print(f"[4] WORST rerank cell = {worst:<22} accuracy = {rerank.loc[worst,'accuracy']:.3f}")
print(f"    number of rerank conditions = {len(rerank)}")

# 3) Exact McNemar (two-sided binomial on discordant pairs) vs retrieval + Bonferroni.
ret_v = verdicts["retrieval"]
m = len(rerank)
alpha_b = 0.05 / m
n_sig = n_worse = n_better = 0
for cond in rerank.index:
    cv = verdicts[cond]
    idx = ret_v.index.intersection(cv.index)
    a = ret_v.loc[idx]              # retrieval correct?
    b = cv.loc[idx]                 # condition correct?
    only_ret = int((a & ~b).sum())  # b in McNemar 2x2
    only_cond = int((~a & b).sum()) # c in McNemar 2x2
    movers = only_ret + only_cond
    p = 1.0 if movers == 0 else binomtest(only_ret, movers, 0.5,
                                          alternative="two-sided").pvalue
    if p < alpha_b:
        n_sig += 1
        if b.mean() - a.mean() < 0:
            n_worse += 1
        else:
            n_better += 1
print(f"[5] Bonferroni alpha = 0.05/{m} = {alpha_b:.2e}")
print(f"    significant after Bonferroni = {n_sig} / {m}  "
      f"(KG worse = {n_worse}, KG better = {n_better})")