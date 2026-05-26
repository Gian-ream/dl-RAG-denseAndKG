# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 09 — LLM-as-judge (Qwen2.5-7B-Instruct) over the 91 conditions
#
# ## Big picture
#
# Pipeline position:
# `... → 07_kg_rerank → 08_llm_eval → **09_llm_judge**`
#
# **Goal**: for each `(query_id, condition)` produced by notebook 08,
# decide with a strong-but-cheap judge LLM whether the response generated
# by Llama-2 contains at least one of the **gold answers** for that query.
# Output: one boolean per row.
#
# Why an LLM judge instead of substring match: NQ answers come as a list
# of accepted variants (`["10 miles", "ten miles"]`, `["Paul Telfer",
# "Telfer"]`) and the generator's response is often verbose or paraphrased
# ("The song is performed by..." / "10 MILES (from Document [4])"). Plain
# substring match collapses on case, plurals, punctuation, paraphrases, and
# preambles. An LLM judge handles all of these natively.
#
# **Why Qwen2.5-7B-Instruct (and not Llama-2 again)**: judge MUST be a
# different family from the generator to avoid pet-grades-itself bias, AND
# it must be at least as instruction-tuned as the generator — a base-level
# model would noise the verdicts. Qwen2.5-7B-Instruct (Apache 2.0, no
# license gating) is strong and free. Pinned to Qwen2.5 (not Qwen3)
# because our installed transformers (4.48) does not yet ship the
# `qwen3` architecture class.
#
# **Scale**: 91 conditions × 1000 queries = **91000 binary judgments**.
#
# ## I/O
#
# **Inputs** (`data/NQ_answer/`):
# - `queries_curated.jsonl`                      — gold answers per query
# - `llm_eval/llm_responses_{condition}.jsonl`   — 91 files from 08
#
# **Outputs** (`data/NQ_answer/llm_eval/`):
# - `judgments_{condition}.jsonl`                — 91 files, schema:
#   `{query_id, condition, response, gold_answers, judge_raw, verdict_bool}`
# - `judgments_summary.parquet`                  — aggregated TRUE counts +
#   accuracy + McNemar p-value per (condition vs retrieval)
#
# **Model**: `Qwen/Qwen2.5-7B-Instruct` (HuggingFace, Apache 2.0, no gating).

# %% [markdown]
# ## 1 · Setup

# %%
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


# %%
# Determinism setup. Must run BEFORE any CUDA initialization so that
# CUBLAS_WORKSPACE_CONFIG is picked up by the CUDA context at startup.
# Same rationale as notebook 08 — see comments there for details.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"determinism: seed={SEED}, "
      f"deterministic_algorithms=True (warn_only), cudnn=deterministic")


# %%
# Walk-up to repo root via pyproject.toml — same pattern as other notebooks.
def find_repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError("Could not find repo root (pyproject.toml)")


REPO_ROOT = find_repo_root()
NQ_DIR = REPO_ROOT / "data" / "NQ_answer"
LLM_EVAL_DIR = NQ_DIR / "llm_eval"

QUERIES_PATH = NQ_DIR / "queries_curated.jsonl"
JUDGMENTS_SUMMARY_PATH = LLM_EVAL_DIR / "judgments_summary.parquet"

print(f"REPO_ROOT: {REPO_ROOT}")
for p in [QUERIES_PATH, LLM_EVAL_DIR]:
    flag = "OK" if p.exists() else "MISSING"
    print(f"  [{flag}] {p.relative_to(REPO_ROOT)}")


# %%
# Load HF token from .hf_token (gitignored) and expose it via env var so that
# AutoTokenizer / AutoModelForCausalLM in section 5 can authenticate
# (Qwen2.5 is Apache 2.0 with no gating, but downloading still benefits
# from a logged-in token for rate-limit reasons).
HF_TOKEN_PATH = REPO_ROOT / ".hf_token"
if HF_TOKEN_PATH.is_file():
    os.environ["HF_TOKEN"] = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
    print(f"HF_TOKEN loaded from {HF_TOKEN_PATH.name} (len={len(os.environ['HF_TOKEN'])})")
elif "HF_TOKEN" in os.environ:
    print(f"HF_TOKEN already in env (len={len(os.environ['HF_TOKEN'])})")
else:
    print("WARNING: no HF_TOKEN found (download may work anyway since Qwen3 is open).")

# %% [markdown]
# ## 2 · Configuration

# %%
# Qwen naming note: Qwen2.5 series keeps the explicit "-Instruct" suffix
# for the chat-tuned variant (vs the bare "Qwen/Qwen2.5-7B" base model).
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Inference knobs
BATCH_SIZE = 8                # Qwen2.5-7B at 4-bit ~5 GB → can afford larger batches than Llama-2-7B
JUDGE_MAX_NEW_TOKENS = 10     # binary YES/NO needs <5 tokens; 10 = safety buffer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Conditions: mirror exactly what 08 produced (retrieval + 5 alphas × 18 cells)
CELL_DIST_VARIANTS: list[int] = [1, 2, 3]
THR_VARIANTS: list[tuple[int, str]] = [
    (500, "thr500"),
    (1000, "thr1k"),
    (2000, "thr2k"),
    (5000, "thr5k"),
    (10000, "thr10k"),
    (0, "thrinf"),
]
ALPHAS_TO_TEST = [0.1, 0.3, 0.5, 0.7, 0.9]
CONDITIONS: list[str] = ["retrieval"]
for _dist in CELL_DIST_VARIANTS:
    for _thr_value, _thr_label in THR_VARIANTS:
        for _alpha in ALPHAS_TO_TEST:
            CONDITIONS.append(f"alpha_{int(round(_alpha * 10))}_dist{_dist}_{_thr_label}")
print(f"conditions: {len(CONDITIONS)}  "
      f"(retrieval + {len(CELL_DIST_VARIANTS)}×{len(THR_VARIANTS)}×{len(ALPHAS_TO_TEST)})")
print(f"first 6: {CONDITIONS[:6]}")
print(f"last 6:  {CONDITIONS[-6:]}")

# Sanity print
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: no GPU available, will be very slow on CPU")

# %% [markdown]
# ## 3 · Load inputs

# %% [markdown]
# ### 3.1 · Queries with gold answers

# %%
queries: list[dict] = [json.loads(line) for line in QUERIES_PATH.open(encoding="utf-8")]
print(f"queries loaded: {len(queries):,}")
# Sanity: every query must have a non-empty 'answers' field (gold).
n_no_gold = sum(1 for q in queries if not q.get("answers"))
print(f"  queries with NO gold answers: {n_no_gold}")

# %%
# Lookup query_id (str, positional 0..999) → list of gold answer strings
gold_by_qid: dict[str, list[str]] = {
    str(i): (q.get("answers") or []) for i, q in enumerate(queries)
}
print(f"gold lookup built: {len(gold_by_qid)} queries, "
      f"avg {sum(len(v) for v in gold_by_qid.values()) / len(gold_by_qid):.2f} variants/query")

# %% [markdown]
# ### 3.2 · Responses from notebook 08 (all 91 conditions)

# %%
# Load each llm_responses_{condition}.jsonl into one long DataFrame.
# Schema after load: (query_id, condition, question, response, passage_ids).
rows: list[dict] = []
for condition in CONDITIONS:
    path = LLM_EVAL_DIR / f"llm_responses_{condition}.jsonl"
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        continue
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["condition"] = condition
            rows.append(d)
    print(f"  [OK] {path.name}: loaded")

df_resp = pd.DataFrame(rows)
print(f"\ntotal response rows: {len(df_resp):,}  "
      f"(expected: {len(CONDITIONS)} × 1000 = {len(CONDITIONS) * 1000})")

# %%
df_resp.head(1)

# %%
# Attach gold answers for each row. Rows with response=None (overflow skips
# from 08) get dropped — there's nothing to judge.
df_resp["gold_answers"] = df_resp["query_id"].map(gold_by_qid)
n_null_resp = df_resp["response"].isna().sum()
print(f"rows with response=null (overflow from 08): {n_null_resp:,}")
df_resp = df_resp.dropna(subset=["response"]).reset_index(drop=True)
print(f"rows kept for judging: {len(df_resp):,}")

# %% [markdown]
# ## 4 · Build judge prompts (Qwen3 chat template, structured)

# %% [markdown]
# Two messages: a strict `system` defining the task, and a `user`
# containing the (question, gold list, response) triple. Qwen2.5 chat
# template wraps with `<|im_start|>system / <|im_start|>user /
# <|im_start|>assistant` and closes turns with `<|im_end|>`. No thinking
# mode to disable (that's a Qwen3-only feature).

# %%
JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge. Given a question, a list of acceptable answers, "
    "and a model response, decide whether the response contains at least one "
    "of the acceptable answers (verbatim or as a clear paraphrase). "
    "Reply with exactly YES or NO, nothing else."
)


def build_judge_user_message(question: str, gold_answers: list[str], response: str) -> str:
    """User-turn message for one judgment: triple (Q, gold list, response)."""
    gold_json = json.dumps(gold_answers, ensure_ascii=False)
    return (
        f"Question: {question}\n"
        f"Acceptable answers: {gold_json}\n"
        f"Response: {response}\n\n"
        f"Does the response contain at least one of the acceptable answers? "
        f"Reply YES or NO."
    )


# Demo: show one user message + how it looks chat-templated (after tokenizer loads)
_demo_row = df_resp.iloc[0]
_demo_user = build_judge_user_message(
    _demo_row["question"], _demo_row["gold_answers"], _demo_row["response"]
)
print("=== example judge user message ===")
print(_demo_user)

# %% [markdown]
# ## 5 · Init Qwen3-8B (4-bit NF4 via bitsandbytes)

# %%
print(f"Loading tokenizer for {JUDGE_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)

# Qwen uses <|endoftext|> as eos and has a pad_token of its own usually; if
# missing, alias to eos for batching. padding_side="left" mandatory for
# causal LM batched generation (same reason as in 08).
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"tokenizer loaded. pad_token={tokenizer.pad_token!r}, "
      f"padding_side={tokenizer.padding_side}, eos_token_id={tokenizer.eos_token_id}")

# %%
# 4-bit NF4 quantization — same config as 08 (compute_dtype=bfloat16,
# double_quant on). On the RTX 5070 Ti (16 GB), Qwen3-8B in 4-bit fits
# in ~5 GB → plenty of headroom for larger batches.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model {JUDGE_MODEL_NAME} in 4-bit NF4 on {DEVICE}...")
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    JUDGE_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
print(f"model loaded in {time.perf_counter() - t0:.1f}s")
print(f"model dtype: {model.dtype}")
if torch.cuda.is_available():
    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM used: {used_gb:.2f} GB")

# %%
# Generation config: deterministic short answer. eos_token_id is the
# native Qwen eos (typically <|im_end|>); we don't need a newline stop
# because the model is instruction-tuned and will close the turn cleanly.
gen_config = GenerationConfig(
    max_new_tokens=JUDGE_MAX_NEW_TOKENS,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"generation config: max_new_tokens={JUDGE_MAX_NEW_TOKENS}, greedy")

# %% [markdown]
# ### 5.1 · Sanity: 1 prompt, 1 generation

# %%
_demo_messages = [
    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
    {"role": "user", "content": _demo_user},
]
_demo_prompt_text = tokenizer.apply_chat_template(
    _demo_messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("=== example chat-templated judge prompt (first 800 chars) ===")
print(_demo_prompt_text[:800])

# %%
_demo_inputs = tokenizer(_demo_prompt_text, return_tensors="pt").to(DEVICE)
print(f"prompt tokens: {_demo_inputs.input_ids.shape[1]}")

t0 = time.perf_counter()
with torch.no_grad():
    _demo_output = model.generate(**_demo_inputs, generation_config=gen_config)
elapsed = time.perf_counter() - t0

_demo_input_len = _demo_inputs.input_ids.shape[1]
_demo_judge_raw = tokenizer.decode(
    _demo_output[0][_demo_input_len:], skip_special_tokens=True
).strip()
print(f"\ngeneration: {elapsed:.1f}s")
print(f"question: {_demo_row['question']}")
print(f"gold:     {_demo_row['gold_answers']}")
print(f"response: {_demo_row['response']!r}")
print(f"judge:    {_demo_judge_raw!r}")

# %% [markdown]
# ## 6 · Batched judgment + per-condition save (resumable)

# %%
def chunked(iterable: list, n: int) -> Iterable[list]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def process_condition_judge(
    condition: str,
    df_resp_cond: pd.DataFrame,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Judge all responses for ONE condition, append to out_path JSONL.

    Idempotent: existing query_ids in out_path are skipped (resume after crash).
    Output row schema: {query_id, condition, response, gold_answers,
                        judge_raw, verdict_bool}.
    """
    done_qids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                done_qids.add(json.loads(line)["query_id"])
        print(f"  [{condition}] resume: {len(done_qids):,} already judged, skipping")

    todo = df_resp_cond[~df_resp_cond["query_id"].isin(done_qids)].to_dict("records")
    if not todo:
        print(f"  [{condition}] all done")
        return

    print(f"  [{condition}] judging {len(todo):,} rows in batches of {batch_size}")
    t_cond_start = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(chunked(todo, batch_size)):
            # Build chat-templated prompts (one per response)
            prompts = []
            for r in batch:
                user_msg = build_judge_user_message(
                    r["question"], r["gold_answers"], r["response"]
                )
                messages = [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt_text)

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)

            input_len = inputs.input_ids.shape[1]
            for i, r in enumerate(batch):
                response_tokens = outputs[i][input_len:]
                judge_raw = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                verdict_bool = parse_yes_no(judge_raw)
                fout.write(json.dumps({
                    "query_id": r["query_id"],
                    "condition": condition,
                    "response": r["response"],
                    "gold_answers": r["gold_answers"],
                    "judge_raw": judge_raw,
                    "verdict_bool": verdict_bool,
                }, ensure_ascii=False) + "\n")
            fout.flush()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) * batch_size >= len(todo):
                done_count = min((batch_idx + 1) * batch_size, len(todo))
                elapsed_min = (time.perf_counter() - t_cond_start) / 60
                rate = done_count / max(elapsed_min, 1e-6)
                eta_min = (len(todo) - done_count) / max(rate, 1e-6)
                print(
                    f"    [{condition}] {done_count}/{len(todo)}  "
                    f"({elapsed_min:.1f} min elapsed, {eta_min:.1f} min ETA, "
                    f"~{rate:.0f} prompts/min)",
                    flush=True,
                )

    print(f"  [{condition}] done in {(time.perf_counter() - t_cond_start) / 60:.1f} min")


# Robust YES/NO parser: tolerates "Yes", "YES.", "yes\n", "Yes, the response..."
# Returns True for YES, False for NO, None for unparseable.
YES_NO_PATTERN = re.compile(r"^\s*(yes|no)\b", flags=re.IGNORECASE)


def parse_yes_no(raw: str) -> bool | None:
    m = YES_NO_PATTERN.match(raw or "")
    if not m:
        return None
    return m.group(1).lower() == "yes"


# %%
# Run all 91 conditions sequentially
t_total_start = time.perf_counter()
for condition in CONDITIONS:
    out_path = LLM_EVAL_DIR / f"judgments_{condition}.jsonl"
    df_resp_cond = df_resp[df_resp["condition"] == condition]
    if len(df_resp_cond) == 0:
        print(f"  [{condition}] no responses to judge, skipping")
        continue
    process_condition_judge(condition, df_resp_cond, out_path)

print(f"\nALL CONDITIONS DONE in {(time.perf_counter() - t_total_start) / 60:.1f} min")
for condition in CONDITIONS:
    out_path = LLM_EVAL_DIR / f"judgments_{condition}.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in out_path.open("r", encoding="utf-8"))
        size_kb = out_path.stat().st_size / 1024
        print(f"  {out_path.name}: {n_lines:,} judgments, {size_kb:.1f} KB")

# %% [markdown]
# ## 7 · Load + parse all judgments into one DataFrame

# %%
judgments: list[dict] = []
for condition in CONDITIONS:
    path = LLM_EVAL_DIR / f"judgments_{condition}.jsonl"
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            judgments.append(json.loads(line))

df_judge = pd.DataFrame(judgments)
print(f"loaded judgments: {len(df_judge):,} across "
      f"{df_judge['condition'].nunique()} conditions")

# %%
# Tally any unparseable judgments (verdict_bool=None) — should be near-zero
# with Qwen3-8B + strict system prompt, but worth tracking.
n_unparsed = df_judge["verdict_bool"].isna().sum()
print(f"unparseable judgments: {n_unparsed:,} / {len(df_judge):,} "
      f"({n_unparsed / len(df_judge):.1%})")
if n_unparsed > 0:
    print("\nsample of unparseable judge outputs (first 5):")
    print(df_judge[df_judge["verdict_bool"].isna()]
          .head(5)[["condition", "query_id", "judge_raw"]]
          .to_string(index=False))

# %% [markdown]
# ## 8 · Aggregate per condition + plot

# %% [markdown]
# ### 8.1 · TRUE counts per condition

# %%
# Treat unparseable verdicts as FALSE (conservative: doesn't artificially
# inflate accuracy). Trace the count separately so it stays visible.
df_judge["verdict_safe"] = df_judge["verdict_bool"].fillna(False)

agg = (
    df_judge.groupby("condition")["verdict_safe"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "n_true", "count": "n_total", "mean": "accuracy"})
)
# Reorder conditions: same order as CONDITIONS (retrieval first, then
# alpha_X_dist{D}_thr{T} in nested dist→thr→alpha order)
agg = agg.reindex(CONDITIONS)
print("accuracy per condition:")
print(agg.to_string())

# %% [markdown]
# ### 8.2 · Three-panel plot — one subplot per distance, six curves per threshold
#
# Layout: 3 subplots side-by-side (dist=1, dist=2, dist=3). Each subplot
# shows 6 curves (one per threshold) × 5 α points, plus a horizontal
# dashed line at the retrieval baseline accuracy for reference. Reading
# the plot:
# - Vertical comparisons WITHIN a subplot → effect of threshold at fixed dist
# - Horizontal comparisons ACROSS subplots → effect of distance
# - Distance of any curve from the dashed line → KG-rerank gain vs retrieval

# %%
import matplotlib.pyplot as plt

retrieval_acc = agg.loc["retrieval", "accuracy"]
fig, axes = plt.subplots(1, len(CELL_DIST_VARIANTS), figsize=(15, 5), sharey=True)

# Markers and colors per threshold (consistent across subplots for legibility)
thr_markers = ["o", "s", "^", "D", "v", "P"]
thr_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(THR_VARIANTS)))

xs = ALPHAS_TO_TEST
for ax, dist in zip(axes, CELL_DIST_VARIANTS):
    # Retrieval baseline (same value in every subplot — α-independent)
    ax.axhline(retrieval_acc, color="black", linestyle="--", linewidth=1.5,
               label=f"retrieval = {retrieval_acc:.3f}")

    # 6 curves, one per threshold
    for (thr_value, thr_label), marker, color in zip(THR_VARIANTS, thr_markers, thr_colors):
        ys = [
            agg.loc[f"alpha_{int(round(a * 10))}_dist{dist}_{thr_label}", "accuracy"]
            for a in xs
        ]
        ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8, label=thr_label)

    ax.set_title(f"dist={dist}")
    ax.set_xlabel("α  (KG weight)")
    ax.grid(True, alpha=0.3)
    if dist == CELL_DIST_VARIANTS[0]:
        ax.set_ylabel("accuracy (judge says YES / total)")

# Single legend outside the right edge — avoids cluttering each subplot
axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
fig.suptitle("LLM-judge accuracy across the 18-cell KG-rerank grid", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9 · McNemar significance + sanity spot-check

# %% [markdown]
# ### 9.1 · McNemar test — retrieval vs each alpha condition
#
# Each query has a paired observation (was retrieval correct? was
# condition X correct?). McNemar tests if the **disagreement** between
# the two is symmetric (no real effect) or asymmetric (one beats the
# other beyond chance).
#
# 2×2 table: rows = retrieval verdict, cols = condition verdict.
# We only care about the off-diagonal:
# - `b` = retrieval=T, condition=F  (KG made it worse)
# - `c` = retrieval=F, condition=T  (KG made it better)
# Statistic: `(b - c)² / (b + c)`, χ² with 1 df.
# Significant p < 0.05 means the two methods disagree non-symmetrically.

# %%
from scipy.stats import chi2


def mcnemar_paired(verdict_a: pd.Series, verdict_b: pd.Series) -> dict:
    """Returns {'b': only_A, 'c': only_B, 'stat': chi2, 'p_value': float}.

    b = A correct AND B incorrect
    c = A incorrect AND B correct
    """
    b = int(((verdict_a == True) & (verdict_b == False)).sum())
    c = int(((verdict_a == False) & (verdict_b == True)).sum())
    if (b + c) == 0:
        return {"b": b, "c": c, "stat": 0.0, "p_value": 1.0}
    stat = (b - c) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(stat, df=1))
    return {"b": b, "c": c, "stat": float(stat), "p_value": p_value}


# Pivot to (query_id × condition) wide form for paired testing
pivot_verdict = df_judge.pivot(
    index="query_id", columns="condition", values="verdict_safe"
).reindex(columns=CONDITIONS)

mcnemar_rows = []
for cond in CONDITIONS:
    if cond == "retrieval":
        continue
    res = mcnemar_paired(pivot_verdict["retrieval"], pivot_verdict[cond])
    delta_acc = (
        pivot_verdict[cond].mean() - pivot_verdict["retrieval"].mean()
    )
    mcnemar_rows.append({
        "condition": cond,
        "retrieval_only_correct (b)": res["b"],
        f"{cond}_only_correct (c)": res["c"],
        "delta_acc": delta_acc,
        "chi2": res["stat"],
        "p_value": res["p_value"],
        "significant_05": res["p_value"] < 0.05,
    })

# Keep generic column names for the printed table
mcnemar_df = pd.DataFrame([
    {
        "condition": r["condition"],
        "retrieval_only (b)": r["retrieval_only_correct (b)"],
        "condition_only (c)": [v for k, v in r.items() if k.endswith("_only_correct (c)")][0],
        "delta_acc": r["delta_acc"],
        "chi2": r["chi2"],
        "p_value": r["p_value"],
        "sig@.05": r["significant_05"],
    }
    for r in mcnemar_rows
])
print("McNemar test: retrieval vs each rerank condition")
print(mcnemar_df.to_string(index=False))

# %% [markdown]
# ### 9.2 · Sanity spot-check: random TRUE + random FALSE judgments
#
# Eyeball check that Qwen3 is actually doing what we think.
# Print 10 random TRUE and 10 random FALSE rows with (question, gold, response,
# verdict). If verdicts look wrong → judge is unreliable for our task and
# we'd need to revisit the prompt or model.

# %%
N_SAMPLE = 10
rng = np.random.default_rng(SEED)

for verdict_label, verdict_value in [("TRUE", True), ("FALSE", False)]:
    pool = df_judge[df_judge["verdict_safe"] == verdict_value]
    if len(pool) == 0:
        print(f"\n=== {verdict_label} samples: none available ===")
        continue
    sample_idx = rng.choice(len(pool), size=min(N_SAMPLE, len(pool)), replace=False)
    sample = pool.iloc[sample_idx]
    print(f"\n{'=' * 78}")
    print(f"=== {N_SAMPLE} random {verdict_label} judgments ===")
    print(f"{'=' * 78}")
    for _, row in sample.iterrows():
        print(f"\n[{row['condition']}]  query_id={row['query_id']}")
        print(f"  question: {queries[int(row['query_id'])].get('question', '')}")
        print(f"  gold:     {row['gold_answers']}")
        print(f"  response: {row['response']!r}")
        print(f"  judge:    {row['judge_raw']!r} → {row['verdict_bool']}")

# %% [markdown]
# ### 9.3 · Save aggregated summary
#
# Combine 8.1 (accuracy per condition) + 9.1 (McNemar) into one parquet
# for downstream use / writeup.

# %%
summary = agg.copy()
summary["retrieval_baseline_acc"] = retrieval_acc
summary["delta_vs_retrieval"] = summary["accuracy"] - retrieval_acc

# Attach McNemar columns where available (retrieval row stays NaN)
mcnemar_idx = mcnemar_df.set_index("condition")
for col in ["chi2", "p_value", "sig@.05"]:
    summary[col] = mcnemar_idx[col]

summary.to_parquet(JUDGMENTS_SUMMARY_PATH, compression="snappy")
print(f"saved: {JUDGMENTS_SUMMARY_PATH.relative_to(REPO_ROOT)}  "
      f"({JUDGMENTS_SUMMARY_PATH.stat().st_size / 1024:.1f} KB)")
print()
print("final summary:")
print(summary.to_string())