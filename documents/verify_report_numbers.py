"""Independent re-verification of the numeric claims in the report body
(Method / Results / Discussion). Recomputes from raw artefacts, not printouts.

Run:  .venv\\Scripts\\python.exe documents\\verify_report_numbers.py
"""
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLM = ROOT / "data" / "NQ_answer" / "llm_eval"
DB = ROOT / "data" / "db"
NQ = ROOT / "data" / "NQ_answer"


def load_jsonl(path: pathlib.Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


print("=" * 70)
print("METHOD")
print("=" * 70)

# M1 — total edges
edges_rows = pq.ParquetFile(DB / "edges.parquet").metadata.num_rows
print(f"[M1] edges.parquet rows = {edges_rows:,}  -> {edges_rows/1e6:.1f}M")

# M2 — Q30 degree
import duckdb  # noqa: E402
con = duckdb.connect()
q30 = con.execute(
    f"SELECT total_degree FROM read_parquet('{(DB/'node_stats.parquet').as_posix()}') "
    "WHERE qid='Q30'").fetchone()[0]
print(f"[M2] Q30 (USA) total_degree = {q30:,}  -> {q30/1e6:.2f}M")

# M3 — corpus passage count via FAISS shard id arrays (mmap header only)
id_files = sorted((ROOT / "data" / "faiss_index").glob("shard_*_ids.npy"))
if id_files:
    total = sum(np.load(p, mmap_mode="r").shape[0] for p in id_files)
    print(f"[M3] passages (sum of {len(id_files)} FAISS shard id arrays) = {total:,}"
          f"  -> {total/1e6:.1f}M")
else:
    print("[M3] FAISS shard id arrays not found; corpus size per PROJECT_NOTES = 41,995,761")

# Per-condition accuracy from per-query verdicts
verdicts = {}
acc = {}
for p in sorted(LLM.glob("judgments_*.jsonl")):
    cond = p.stem.replace("judgments_", "")
    v = load_jsonl(p).set_index("query_id")["verdict_bool"].astype(bool)
    verdicts[cond] = v
    acc[cond] = float(v.mean())
conds = [c for c in acc if c != "retrieval"]
print(f"[M4] conditions = {len(acc)} (= 1 retrieval + {len(conds)} rerank); "
      f"queries per condition = {len(verdicts['retrieval'])}")

print("=" * 70)
print("RESULTS")
print("=" * 70)
baseline = acc["retrieval"]
best = max(conds, key=lambda c: acc[c])
worst = min(conds, key=lambda c: acc[c])
print(f"[R1] baseline={baseline:.3f} | best={best}={acc[best]:.3f} | worst={worst}={acc[worst]:.3f}")
print(f"[R2] best - baseline = {acc[best]-baseline:+.3f} = {round((acc[best]-baseline)*1000)} queries/1000")

# R3 — Table 1 cells
print("[R3] Table 1 cells (accuracy):")
cells = {"dist2": ["thr500", "thr5k", "thrinf"], "dist3": ["thr5k", "thrinf"]}
print("      a   | d2t500 d2t5k  d2tinf | d3t5k  d3tinf")
for a in (1, 3, 5, 9):
    row = [f"{acc[f'alpha_{a}_dist2_thr500']:.3f}", f"{acc[f'alpha_{a}_dist2_thr5k']:.3f}",
           f"{acc[f'alpha_{a}_dist2_thrinf']:.3f}", f"{acc[f'alpha_{a}_dist3_thr5k']:.3f}",
           f"{acc[f'alpha_{a}_dist3_thrinf']:.3f}"]
    print(f"     0.{a}  | {row[0]}  {row[1]}  {row[2]} | {row[3]}  {row[4]}")

# R4 — threshold gradient at alpha=0.9, dist=2
grad = [acc[f"alpha_9_dist2_thr{t}"] for t in ("500", "1k", "2k", "5k", "10k", "inf")]
mono = all(grad[i] >= grad[i+1] for i in range(len(grad)-1))
print(f"[R4] alpha_9 dist2 threshold gradient {grad}  monotone-decreasing={mono}")

# R5 — dist=1 invariance to threshold
inv_ok = True
for a in (1, 3, 5, 7, 9):
    vals = {acc[f"alpha_{a}_dist1_thr{t}"] for t in ("500", "1k", "2k", "5k", "10k", "inf")}
    if len(vals) != 1:
        inv_ok = False
print(f"[R5] dist1 accuracy identical across all 6 thresholds for every alpha = {inv_ok}")

# R6 — McNemar exact + Bonferroni + mover counts
from scipy.stats import binomtest  # noqa: E402
ret = verdicts["retrieval"]
m = len(conds)
alpha_b = 0.05 / m
n_sig = n_worse = 0
movers = []
for c in conds:
    a_, b_ = ret.align(verdicts[c], join="inner")
    bb = int((a_ & ~b_).sum()); cc = int((~a_ & b_).sum()); mv = bb + cc
    movers.append(mv)
    p = 1.0 if mv == 0 else binomtest(bb, mv, 0.5, alternative="two-sided").pvalue
    if p < alpha_b:
        n_sig += 1
        if b_.mean() - a_.mean() < 0:
            n_worse += 1
movers = np.array(movers)
print(f"[R6] Bonferroni alpha=0.05/{m}={alpha_b:.2e} | significant={n_sig}/{m} (worse={n_worse})")
print(f"[R7] movers b+c: min={movers.min()} median={int(np.median(movers))} max={movers.max()} "
      f"(claim 'b+c ~ 200')")

print("=" * 70)
print("DISCUSSION")
print("=" * 70)

# D1 — ~17% all-hub queries (all question_qids have total_degree > 5000)
queries = [json.loads(l) for l in open(NQ / "queries_curated.jsonl", encoding="utf-8")]
all_qids = sorted({q for r in queries for q in (r.get("question_qids") or [])})
deg = dict(con.execute(
    "SELECT qid, total_degree FROM read_parquet('"
    f"{(DB/'node_stats.parquet').as_posix()}') WHERE qid IN "
    f"({','.join(repr(q) for q in all_qids)})").fetchall())
THR = 5000
n_allhub = 0
for r in queries:
    qq = r.get("question_qids") or []
    if qq and all(deg.get(q, 0) > THR for q in qq):
        n_allhub += 1
print(f"[D1] all-hub queries (every question_qid deg>{THR}) = {n_allhub}/{len(queries)} "
      f"= {100*n_allhub/len(queries):.1f}%  (claim '~17%')")

# D2 — saturation: connected_ratio at dist=3, threshold=inf
kgp = NQ / "kg_pairs_raw.parquet"
if kgp.exists():
    cols = pq.ParquetFile(kgp).schema.names
    if "connected_ratio" in cols and "distance" in cols and "threshold" in cols:
        # inf threshold is encoded as 0 in kg_pairs_raw (utils/kg.py _resolve_threshold)
        thr_inf = 0
        cr = con.execute(
            f"SELECT avg(connected_ratio), median(connected_ratio), "
            f"avg((connected_ratio=1.0)::INT) "
            f"FROM read_parquet('{kgp.as_posix()}') WHERE distance=3 AND threshold={thr_inf}").fetchone()
        print(f"[D2] connected_ratio @ dist=3, threshold=inf (encoded 0): "
              f"mean={cr[0]:.3f}, median={cr[1]:.3f}, frac==1={cr[2]:.3f}  (claim 'connected_ratio ~ 1')")
    else:
        print(f"[D2] kg_pairs_raw columns = {cols} (adjust query)")
else:
    print("[D2] kg_pairs_raw.parquet not found; saturation claim from PROJECT_NOTES")