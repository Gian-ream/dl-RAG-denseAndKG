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
# # 08 — LLM evaluation (Llama-2-7B base) over the α sweep
#
# ## Big picture
#
# Pipeline position:
# `01_corpus → 02_filter → 03_embed → 04_answer → 05_curate → 06_apply → 07_kg_rerank → **08_llm_eval**`
#
# **Goal**: per each query and for **91 retrieval/rerank conditions**
# (pure retrieval at α=0 + α ∈ {0.1, 0.3, 0.5, 0.7, 0.9} on **all 18 KG
# cells** — full grid: 3 distances {1, 2, 3} × 6 thresholds {500, 1000,
# 2000, 5000, 10000, ∞}), build the prompt with top-5 passages and
# **record Llama-2-7B's answer** to the question.
#
# Full grid coverage so we can map systematically where (if anywhere)
# KG-rerank beats the dense retrieval baseline. Previous experiments
# with only (dist=3, thr=10k) and (dist=3, thr=∞) showed retrieval
# winning everywhere; this run extends to the full {dist × thr} space.
#
# We do NOT score answers here; we only collect the raw responses for
# downstream comparison (LLM-as-judge in notebook 09).
#
# **Scale**: 91 conditions × 1000 queries = **91000 LLM generations**
# (~20h on RTX 5070 Ti at ~75 prompts/min — overnight run).
#
# ## I/O
#
# **Inputs** (`data/NQ_answer/`):
# - `queries_curated.jsonl`            — question text + question_qids
# - `passage_entities_curated.parquet` — passage id, **title, text**, qids
# - `top100_curated.parquet`           — (query_id, passage_id, score, rank)
# - `kg_pairs_raw.parquet`             — Phase A output: kg_score per cell
#
# **Outputs** (`data/NQ_answer/llm_eval/`):
# - `llm_eval_inputs.parquet` — (condition, query_id, passage_ids[5], prompt)
# - `llm_responses_{condition}.jsonl` — 1 file per condition, JSONL with
#   `{query_id, question, passage_ids, response}`. Append mode (resumable).
#
# **Model**: `meta-llama/Llama-2-7b-hf` (HuggingFace, base / non-instruct
# variant — naturally completion-style, no RLHF "Based on the documents…"
# preambles. Requires Meta license acceptance + HF_TOKEN env var).

# %% [markdown]
# ## 1 · Setup

# %%
import json
import os
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
LLM_OUT_DIR = NQ_DIR / "llm_eval"
LLM_OUT_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_PATH = NQ_DIR / "queries_curated.jsonl"
PASSAGES_PATH = NQ_DIR / "passage_entities_curated.parquet"
TOP100_PATH = NQ_DIR / "top100_curated.parquet"
PHASE_A_PATH = NQ_DIR / "kg_pairs_raw.parquet"

LLM_INPUTS_PATH = LLM_OUT_DIR / "llm_eval_inputs.parquet"

print(f"REPO_ROOT: {REPO_ROOT}")
for p in [QUERIES_PATH, PASSAGES_PATH, TOP100_PATH, PHASE_A_PATH]:
    flag = "OK" if p.exists() else "MISSING"
    print(f"  [{flag}] {p.relative_to(REPO_ROOT)}")
print(f"output dir: {LLM_OUT_DIR.relative_to(REPO_ROOT)}")

# %%
# Load HF token from .hf_token (gitignored) and expose it via env var so that
# AutoTokenizer / AutoModelForCausalLM in section 5 can authenticate against
# gated repos (Llama-2 requires Meta license acceptance + a personal token).
HF_TOKEN_PATH = REPO_ROOT / ".hf_token"
if HF_TOKEN_PATH.is_file():
    os.environ["HF_TOKEN"] = HF_TOKEN_PATH.read_text(encoding="utf-8").strip()
    print(f"HF_TOKEN loaded from {HF_TOKEN_PATH.name} (len={len(os.environ['HF_TOKEN'])})")
elif "HF_TOKEN" in os.environ:
    print(f"HF_TOKEN already in env (len={len(os.environ['HF_TOKEN'])})")
else:
    print(f"WARNING: no HF_TOKEN found. Section 5 will fail on gated models.")

# %% [markdown]
# ## 2 · Configuration

# %%
# Model + chosen rerank cell + alpha sweep granularity
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# Full {distance × threshold} grid: 3 distances × 6 thresholds = 18 cells.
# - Distances 1, 2, 3 = max hops to reach passage entity from query entity.
# - Thresholds: hub-degree cutoff. Low (500) = aggressive hub filtering;
#   high (10000) = mild filtering; 0 = no filter (codes ∞ in kg_pairs_raw).
# Naming: thr500 ... thr10k for finite values, thrinf for 0.
CELL_DIST_VARIANTS: list[int] = [1, 2, 3]
THR_VARIANTS: list[tuple[int, str]] = [
    (500, "thr500"),
    (1000, "thr1k"),
    (2000, "thr2k"),
    (5000, "thr5k"),
    (10000, "thr10k"),
    (0, "thrinf"),
]

# Conditions: retrieval baseline (α=0, cell-agnostic) + 5 alphas × 18 cells
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

# Inference knobs (tune to your GPU)
BATCH_SIZE = 4               # bump up if VRAM allows (8/16 on 24GB+)
MAX_NEW_TOKENS = 15          # Silvestri "Power of Noise": extract-style answer (≤5 tok target, 15 = safe ceiling)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Note: no PASSAGE_CHAR_LIMIT / no prompt truncation. Per design choice,
# prompts that exceed the model's native context window (4096 tokens for
# Llama-2) are SKIPPED and logged as overflow errors, never truncated.
# With MAX_NEW_TOKENS=15 the PROMPT_TOKEN_LIMIT becomes 4096-15 = 4081
# (more permissive than the previous 3968 → fewer overflows). See section 5.2.

# Sanity print
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: no GPU available, will be very slow on CPU")

# %% [markdown]
# ## 3 · Load inputs

# %% [markdown]
# ### 3.1 · Queries (jsonl)

# %%
queries: list[dict] = [json.loads(line) for line in QUERIES_PATH.open(encoding="utf-8")]
print(f"queries loaded: {len(queries):,}")

# %%
queries[0]

# %% [markdown]
# ### 3.2 · Passages (parquet) — id + title + text

# %%
# Load title + text here (unlike notebook 07 where we loaded only id + qids).
# The text is needed for the prompt.
passages_df = pq.read_table(
    PASSAGES_PATH, columns=["id", "title", "text"]
).to_pandas()
print(f"passages loaded: {len(passages_df):,}")

# %%
passages_df.head(1)

# %%
# Lookup passage_id (int) → (title, text) for prompt construction.
text_by_passage: dict[int, tuple[str, str]] = {
    int(row.id): (row.title or "", row.text or "")
    for row in passages_df.itertuples(index=False)
}
print(f"passage text lookup built: {len(text_by_passage):,} entries")

# %% [markdown]
# ### 3.3 · top100 + kg_pairs_raw

# %%
top100 = pq.read_table(TOP100_PATH).to_pandas()
print(f"top100 rows: {len(top100):,}")

# %%
top100.head(1)

# %%
df_phase_a = pd.read_parquet(PHASE_A_PATH)
print(f"df_phase_a rows: {len(df_phase_a):,}")
print(f"  unique queries: {df_phase_a['query_id'].nunique()}  (expected: 1000)")

# %%
df_phase_a.head(1)

# %% [markdown]
# ## 4 · Build top-5 sets per (query, condition)
#
# Replicates the logic of section 6/8 in notebook 07: per-query dense_norm
# via max-only scaling, kg_score from **each of the 18 cells** (3 dists ×
# 6 thresholds), and `final_α = (1-α)·dense_norm + α·kg_score` computed
# per cell. For each of the 91 conditions (retrieval + 5 alphas per cell
# × 18 cells), take the top-5 `passage_id` per query.
#
# Output: parquet `llm_eval_inputs.parquet` with schema
# `(condition, query_id, passage_ids[5], prompt)`.

# %%
# Per-query dense_norm via max-only scaling (same formula as section 6 of notebook 07)
def scale_by_max_per_group(s: pd.Series) -> pd.Series:
    """Max-only scaling: divide by per-group max, no shift."""
    hi = s.max()
    if hi < 1e-12:
        return pd.Series([0.0] * len(s), index=s.index)
    return s / hi


top100["query_id_s"] = top100["query_id"].astype(str)
top100["passage_id_s"] = top100["passage_id"].astype(str)
top100["dense_norm"] = top100.groupby("query_id")["score"].transform(scale_by_max_per_group)

# %%
top100.head(1)

# %%
# Build one kg_score column per cell (dist × thr). Each filtered df_cell
# has pair-level kg_score for that specific cell; rename so all 18 columns
# coexist in `base` without collisions.
def kg_for_cell(dist: int, thr_value: int, thr_label: str) -> pd.DataFrame:
    return (
        df_phase_a[
            (df_phase_a["distance"] == dist) & (df_phase_a["threshold"] == thr_value)
        ][["query_id", "passage_id", "kg_score"]]
        .rename(columns={
            "query_id": "query_id_s",
            "passage_id": "passage_id_s",
            "kg_score": f"kg_score_dist{dist}_{thr_label}",
        })
    )


# Build dict of 18 cells, keyed by combined label "dist{D}_{thr_label}"
df_cells: dict[str, pd.DataFrame] = {}
for dist in CELL_DIST_VARIANTS:
    for thr_value, thr_label in THR_VARIANTS:
        cell_label = f"dist{dist}_{thr_label}"
        df_cells[cell_label] = kg_for_cell(dist, thr_value, thr_label)
print(f"built {len(df_cells)} cells; example shapes:")
for cell_label, df in list(df_cells.items())[:3]:
    print(f"  {cell_label}: {df.shape}")

# %%
# Merge all 18 kg_score variants into `base` (left join → NaN where the pair
# has no kg_score, e.g. passage with no qids). Fill NaN with 0.0.
base = top100[["query_id_s", "passage_id_s", "score", "dense_norm"]].copy()
for cell_label, df in df_cells.items():
    base = base.merge(df, on=["query_id_s", "passage_id_s"], how="left")
    base[f"kg_score_{cell_label}"] = base[f"kg_score_{cell_label}"].fillna(0.0)
print(f"base shape: {base.shape}  (expected: ~100K rows × {4 + len(df_cells)} cols)")

# %%
base.head(1)

# %%
# Add final_alpha_X_dist{D}_thr{T} columns: one per (alpha, dist, threshold).
# Formula identical to section 8 of notebook 07:
#   final = (1 - α) · dense_norm + α · kg_score_{cell}
# Total: 18 cells × 5 alphas = 90 final_alpha columns.
for dist in CELL_DIST_VARIANTS:
    for thr_value, thr_label in THR_VARIANTS:
        kg_col = f"kg_score_dist{dist}_{thr_label}"
        for alpha in ALPHAS_TO_TEST:
            col = f"final_alpha_{int(round(alpha * 10))}_dist{dist}_{thr_label}"
            base[col] = (1 - alpha) * base["dense_norm"] + alpha * base[kg_col]

n_final_cols = len([c for c in base.columns if c.startswith('final_alpha')])
print(f"final_alpha columns: {n_final_cols}  (expected: {len(CELL_DIST_VARIANTS) * len(THR_VARIANTS) * len(ALPHAS_TO_TEST)})")

# %%
# For each (query, condition) extract top-5 passage_id sorted by score
# descending. groupby + nlargest = O(N log 5) per group, fast.
def top5_for_condition(base_df: pd.DataFrame, score_col: str) -> dict[str, list[str]]:
    """Returns {query_id_s → list of 5 passage_id_s in score-descending order}."""
    return {
        qid: g.nlargest(5, score_col)["passage_id_s"].tolist()
        for qid, g in base_df.groupby("query_id_s", sort=False)
    }


# Map condition → score column to sort by. retrieval uses the raw dense
# score; each alpha_X_dist{D}_thr{T} uses its matching final_alpha column.
condition_to_score_col: dict[str, str] = {"retrieval": "score"}
for dist in CELL_DIST_VARIANTS:
    for thr_value, thr_label in THR_VARIANTS:
        for alpha in ALPHAS_TO_TEST:
            cond = f"alpha_{int(round(alpha * 10))}_dist{dist}_{thr_label}"
            condition_to_score_col[cond] = (
                f"final_alpha_{int(round(alpha * 10))}_dist{dist}_{thr_label}"
            )

print(f"condition → score column ({len(condition_to_score_col)} entries):")
# Print first 3 and last 3 for compact preview
items = list(condition_to_score_col.items())
for cond, col in items[:3] + items[-3:]:
    print(f"  {cond:>32}  ←  {col}")

# %%
# For each condition, build the flat list of rows to write into the input
# parquet. One row per (condition, query_id) with the 5 passage_ids in order.
input_rows: list[dict] = []
for condition in CONDITIONS:
    score_col = condition_to_score_col[condition]
    top5_dict = top5_for_condition(base, score_col)
    for qid, passage_ids in top5_dict.items():
        # Find original (positional int) query_id to fetch question text from queries[]
        query_int = int(qid)
        question = queries[query_int].get("question", "")
        input_rows.append({
            "condition": condition,
            "query_id": qid,
            "question": question,
            "passage_ids": passage_ids,
        })
    print(f"  {condition}: {len(top5_dict)} queries built")

print(f"\ntotal input rows: {len(input_rows):,}  "
      f"(expected: {len(CONDITIONS)} × {len(queries)} = {len(CONDITIONS) * len(queries)})")

# %% [markdown]
# ### 4.1 · Build prompts (paper-derived + one-shot example + Q-first)
#
# Prompt structure derived from `florin-git/The-Power-of-Noise`
# (`src/prompt_dataset.py`) with two deliberate deviations:
# 1. **One-shot example prepended** to teach Llama-2 base the
#    short-extractive answer FORMAT (base model has no instruction tuning;
#    without an example it continues "Answer:" with Wikipedia-style prose).
# 2. **Question moved ABOVE Documents** (vs paper's Question-after).
#    Rationale: the model reads the query first and can attend back to it
#    while scanning the docs; example block and real task share the same
#    layout so the demonstration generalizes cleanly.
#
# Final structure:
# ```
# {EXTRACT_INSTRUCTION}
#
# Example:
# Question: who wrote hamlet
# Documents:
# Document [1](Title: Macbeth) ...
# Document [2](Title: Globe Theatre) ...
# Document [3](Title: Tragedy) ...
# Document [4](Title: Elizabethan theatre) ...
# Document [5](Title: Hamlet) ...written by William Shakespeare...
# Answer: William Shakespeare
#
# Question: {real_question}
# Documents:
# Document [1](Title: {title}) {text}
# ...
# Document [5](Title: {title}) {text}
# Answer:
# ```
#
# Design notes:
# - **Raw completion, NO chat template**. Plain text fed directly to the
#   tokenizer — no `apply_chat_template`, no `[INST]...[/INST]` wrappers,
#   no system role. The `Answer:` marker at the bottom is the generation
#   trigger: Llama continues autoregressively producing the extracted span.
# - **Why base (not -chat)**: the chat variant has RLHF fine-tuning that
#   biases it toward conversational preambles ("Based on the documents…")
#   even without `[INST]`. The base model has no such bias — it's a pure
#   next-token completion model and continues "Answer:" with the extracted
#   span directly.
# - **Document format**: `Document [i](Title: {title}) {text}` — title is
#   informative for disambiguation and matches the paper's code.
# - **`reversed(passage_ids)`**: passage_ids arrives sorted by score
#   descending (top-1 at index 0). We reverse so the top-1 ends up in
#   `Document [5]`, immediately above `Answer:`. Rationale: "Power of
#   Noise" + "Lost in the Middle" — causal LLMs have attention bias
#   toward the tail of the context (here = closest to the generation
#   trigger). Same convention used inside the one-shot example
#   (gold Hamlet doc at position [5]).
# - **Hard generation cap**: `MAX_NEW_TOKENS=15` (section 2) caps the
#   API output. The soft NL constraint in the instruction ("max 5 tokens")
#   is verbatim from the paper.
# - **No truncation**: passages enter in full; overflows are skipped in
#   section 6 (logged as error rows in the JSONL).

# %%
EXTRACT_INSTRUCTION = (
    "You are given a question and you MUST respond by EXTRACTING the answer "
    "(max 5 tokens) from one of the provided documents. "
    "If none of the documents contain the answer, respond with NO-RES."
)

# One-shot complete example: same exact structure as the real task block
# (5 documents + Question + Answer). Purpose: prime Llama-2 base on
# (a) the FORMAT — short single-line extractive answer ending in newline —
# and (b) the BEHAVIOR — answer must literally appear in one of the docs.
#
# Topic chosen off-distribution from NQ (common-knowledge Shakespeare fact)
# → zero contamination risk. Distractors deliberately mention "William
# Shakespeare" too (Macbeth, Globe Theatre) so the example teaches "ground
# in the GOLD doc, not just any mention of the entity", mirroring the
# challenge real NQ queries face. Gold doc placed at position [5]
# (closest to Question:) to match our reversed(passage_ids) convention
# where the top-1-scored real passage also lands at [5].
FEW_SHOT_EXAMPLE = """Example:
Question: who wrote hamlet
Documents:
Document [1] (Title: Macbeth) Macbeth is a tragedy by William Shakespeare, believed to have been first performed in 1606. It dramatises the damaging psychological effects of political ambition.
Document [2] (Title: Globe Theatre) The Globe Theatre was a playhouse in London associated with William Shakespeare. It was built in 1599 by Shakespeare's playing company, the Lord Chamberlain's Men.
Document [3] (Title: Tragedy) Tragedy is a genre of drama based on human suffering and, mainly, the terrible events that befall a main character.
Document [4] (Title: Elizabethan theatre) English Renaissance theatre, also known as Elizabethan theatre, refers to the theatre of England between 1562 and 1642.
Document [5] (Title: Hamlet) The Tragedy of Hamlet, Prince of Denmark, often shortened to Hamlet, is a tragedy written by William Shakespeare sometime between 1599 and 1601. It is Shakespeare's longest play.
Answer: William Shakespeare"""


def build_user_message(question: str, passage_ids: list[str]) -> str:
    """Build the raw completion prompt: instruction + 1-shot example + real task.

    Layout (matches FEW_SHOT_EXAMPLE exactly so the model sees the same
    structure in the demonstration and in the real task):
      1. EXTRACT_INSTRUCTION (task definition)
      2. FEW_SHOT_EXAMPLE (demonstration: Question → Documents → Answer)
      3. Real `Question: ... / Documents: ... / Answer:` block

    With Question BEFORE Documents, the model reads the query first and
    can attend back to it while scanning the docs — and the top-1-scored
    real passage still sits at Document [5], immediately above 'Answer:'
    (the generation trigger), preserving the "Power of Noise" tail-bias
    placement.

    Having seen the demonstration end with "Answer: William Shakespeare\\n",
    Llama base strongly predicts the real "Answer:" will be followed by a
    similarly short extractive span and a newline → our newline-stop fires
    cleanly instead of the 15-token cap chopping mid-word.
    """
    parts = [EXTRACT_INSTRUCTION, "", FEW_SHOT_EXAMPLE, ""]
    parts.append(f"Question: {question}")
    parts.append("Documents:")
    for i, pid in enumerate(reversed(passage_ids), start=1):
        title, text = text_by_passage.get(int(pid), ("", "[passage not found]"))
        parts.append(f"Document [{i}](Title: {title}) {text}")
    parts.append("Answer:")
    return "\n".join(parts)


# Demo: build one user message and print it
_demo_row = input_rows[0]
_demo_msg = build_user_message(_demo_row["question"], _demo_row["passage_ids"])
print("=== example user message ===")
print(_demo_msg[:1200])
print(f"\n[...truncated, total length: {len(_demo_msg)} chars]")

# %%
# Bulk: add a "user_message" column to every row
for r in input_rows:
    r["user_message"] = build_user_message(r["question"], r["passage_ids"])

df_inputs = pd.DataFrame(input_rows)
df_inputs.to_parquet(LLM_INPUTS_PATH, compression="snappy")
print(f"\nsaved: {LLM_INPUTS_PATH}  ({LLM_INPUTS_PATH.stat().st_size / 1024**2:.1f} MB)")
print(f"rows: {len(df_inputs):,}")

# %%
df_inputs.head(1)

# %% [markdown]
# ## 5 · Init Llama-2-7B base (4-bit NF4 via bitsandbytes)
#
# Loads model + tokenizer **quantized to 4-bit NF4** on the GPU.
# Aligned with the Silvestri "Power of Noise" setup (4-bit representation).
# **Base model** (not -chat): naturally completion-style for extractive QA.
# Requires:
# - HF account that has accepted the Meta license on the model card
# - `HF_TOKEN` in env (huggingface-cli login OR `os.environ["HF_TOKEN"]`)
# - `bitsandbytes >= 0.49.2` (handles Windows + CUDA 12.8 + Blackwell sm120)
#
# Estimated VRAM: ~4 GB (vs ~14 GB in fp16). Compute dtype = fp16 → fast
# matmuls, weights compressed to 4 bits with double-quant to save further.

# %%
print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Llama-2 tokenizer has no pad_token: alias it to eos_token for batching.
# padding_side="left" is MANDATORY for causal LM batched generation —
# otherwise right-side pad tokens end up inside the generation context.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"tokenizer loaded. pad_token={tokenizer.pad_token!r}, padding_side={tokenizer.padding_side}")

# %%
# 4-bit NF4 quantization config. Same 4 values as florin-git/The-Power-of-Noise
# (src/llm.py::_set_quantization): load_in_4bit, nf4, double_quant, bf16 compute
# — we just pass them as constructor kwargs instead of setting attributes one
# by one. nf4 = 4-bit NormalFloat, optimized for
# weights that follow a normal distribution (i.e. most LLM weights).
# double_quant also quantizes the quantization constants themselves,
# saving another ~0.4 bit/weight. compute_dtype=bfloat16 (paper choice):
# bf16 has the same exponent range as fp32 → fewer overflow issues than
# fp16 during dequantized matmul; Blackwell sm120 supports it natively.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model {MODEL_NAME} in 4-bit NF4 on {DEVICE}...")
t0 = time.perf_counter()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",  # auto-place on available GPU
)
model.eval()
print(f"model loaded in {time.perf_counter() - t0:.1f}s")
print(f"model dtype: {model.dtype}")
if torch.cuda.is_available():
    used_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM used: {used_gb:.2f} GB")

# %%
# Generation config: deterministic + paper-aligned.
# - do_sample=False: greedy decoding (paper)
# - repetition_penalty=1.1: paper (Silvestri llm.py::generate). Discourages
#   the model from repeating tokens it has just generated.
# - eos_token_id=[eos, "\n"]: stop on EITHER the natural eos OR the first
#   newline. Llama-2 in raw-completion mode tends to keep generating
#   "Question: ... Answer: ..." pairs forever — newline-stop catches the
#   end of the extractive answer cleanly. "\n" tokenizes to a single token
#   in the Llama tokenizer (id 13).
NEWLINE_TOKEN_ID = tokenizer.encode("\n", add_special_tokens=False)[-1]
gen_config = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=[tokenizer.eos_token_id, NEWLINE_TOKEN_ID],
)
print(
    f"generation config: max_new_tokens={MAX_NEW_TOKENS}, greedy, "
    f"repetition_penalty=1.1, eos_token_id=[{tokenizer.eos_token_id}, "
    f"{NEWLINE_TOKEN_ID} (\\n)]"
)

# %% [markdown]
# ### 5.1 · Sanity: 1 prompt, 1 generation
#
# Before launching 91000 inferences (overnight run), verify that the pipeline runs on a
# single (query, top-5) pair in retrieval-baseline style.

# %%
# Paper-style raw completion: feed user_message directly to the tokenizer.
# No chat template, no [INST] wrapping. The prompt ends with "Answer:" and
# Llama-2 continues autoregressively with the extracted answer.
_demo_prompt_text = df_inputs.iloc[0]["user_message"]
print("=== example raw-completion prompt (first 800 chars) ===")
print(_demo_prompt_text)

# %%
# Tokenize + generate for a single row
_demo_inputs = tokenizer(_demo_prompt_text, return_tensors="pt").to(DEVICE)
print(f"prompt tokens: {_demo_inputs.input_ids.shape[1]}")

t0 = time.perf_counter()
with torch.no_grad():
    _demo_output = model.generate(**_demo_inputs, generation_config=gen_config)
elapsed = time.perf_counter() - t0

# Decode only the newly generated tokens (skip the prompt)
_demo_input_len = _demo_inputs.input_ids.shape[1]
_demo_response = tokenizer.decode(
    _demo_output[0][_demo_input_len:], skip_special_tokens=True
)
print(f"\ngeneration: {elapsed:.1f}s ({(_demo_output.shape[1] - _demo_input_len) / elapsed:.0f} tok/s)")
print(f"\nquestion: {df_inputs.iloc[0]['question']}")
print(f"\nresponse: {_demo_response}")

# %% [markdown]
# ### 5.2 · Pre-flight: count tokens per prompt, identify overflow
#
# Llama-2 has a 4096-token context window. Reserving `MAX_NEW_TOKENS=15`
# for the answer, each prompt can use at most `4096 − 15 = 4081` tokens.
#
# Strategy (agreed with the user): **no truncation**. We tokenize each
# raw-completion prompt and count tokens. If the prompt exceeds the limit,
# it will be SKIPPED in section 6 (error row in the JSONL instead of a
# truncated answer that would bias the rerank).
#
# Cost: ~91000 tokenizations ≈ tens of seconds (local, no GPU).

# %%
# Natural model limit, NOT an arbitrary value invented by us.
PROMPT_TOKEN_LIMIT = tokenizer.model_max_length - MAX_NEW_TOKENS
print(f"tokenizer.model_max_length: {tokenizer.model_max_length}")
print(f"MAX_NEW_TOKENS:             {MAX_NEW_TOKENS}")
print(f"→ PROMPT_TOKEN_LIMIT:       {PROMPT_TOKEN_LIMIT}  (overflow ⇒ skip)")


# %%
def prompt_token_count(user_message: str) -> int:
    """Tokenize the raw completion prompt and return n_tokens.

    Identical to the pre-processing that section 6 will do — so the count
    is apples-to-apples.
    """
    return len(tokenizer(user_message, return_tensors="pt").input_ids[0])


print("Tokenizing all prompts to count tokens...")
t0 = time.perf_counter()
df_inputs["n_tokens"] = df_inputs["user_message"].apply(prompt_token_count)
print(f"done in {time.perf_counter() - t0:.1f}s")

# %%
# Distribution + overflow identification
print(f"n_tokens distribution (all {len(df_inputs):,} prompts):")
print(df_inputs["n_tokens"].describe())

overflow_mask = df_inputs["n_tokens"] > PROMPT_TOKEN_LIMIT
n_overflow = int(overflow_mask.sum())
print(f"\nprompts EXCEEDING {PROMPT_TOKEN_LIMIT} tokens: "
      f"{n_overflow:,} / {len(df_inputs):,} ({n_overflow / len(df_inputs):.1%})")

if n_overflow > 0:
    print(f"\nworst offenders (top 5 by token count):")
    print(
        df_inputs.nlargest(5, "n_tokens")[["condition", "query_id", "n_tokens"]]
        .to_string(index=False)
    )
    print(f"\nbreakdown per condition:")
    print(
        df_inputs.assign(overflow=overflow_mask)
        .groupby("condition")["overflow"].agg(["sum", "count"])
        .rename(columns={"sum": "overflow", "count": "total"})
        .to_string()
    )
else:
    print("✓ no overflow: all prompts fit within the model's context window")

# %%
# Save the updated parquet so the inference loop in section 6 has n_tokens available.
df_inputs.to_parquet(LLM_INPUTS_PATH, compression="snappy")
print(f"updated: {LLM_INPUTS_PATH}  (now includes n_tokens column)")

# %% [markdown]
# ## 6 · Batched inference + per-condition save (resumable)
#
# For each condition we write `llm_responses_{condition}.jsonl` in append
# mode. If the file exists, we read the already-processed `query_id`s and
# skip them → resume after crash.
#
# **Estimated cost**: with BATCH_SIZE=4, ~50-100 prompts/min on an RTX
# 3090. 91 conditions × 1000 queries / 75 ≈ **150 minutes total**.
# A beefier GPU or a larger batch_size brings it down.

# %%
def chunked(iterable: list, n: int) -> Iterable[list]:
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def process_condition(
    condition: str,
    df_inputs: pd.DataFrame,
    out_path: Path,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Run batched inference for ONE condition, append responses to out_path JSONL.

    Idempotent: existing query_ids in out_path are skipped (resume after crash).
    """
    # Load already-done query_ids (resume support)
    done_qids: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                done_qids.add(json.loads(line)["query_id"])
        print(f"  [{condition}] resume: {len(done_qids):,} already done, skipping")

    rows = df_inputs[df_inputs["condition"] == condition].to_dict("records")
    todo = [r for r in rows if r["query_id"] not in done_qids]
    if not todo:
        print(f"  [{condition}] all done, skipping")
        return

    # Split todo into runnable (within model's context window) vs overflow.
    runnable = [r for r in todo if r["n_tokens"] <= PROMPT_TOKEN_LIMIT]
    overflow = [r for r in todo if r["n_tokens"] > PROMPT_TOKEN_LIMIT]

    # Log overflow rows immediately (no generation, just an error marker)
    if overflow:
        print(
            f"  [{condition}] {len(overflow):,} prompts > {PROMPT_TOKEN_LIMIT} tok "
            f"→ writing error rows, skipping inference"
        )
        with out_path.open("a", encoding="utf-8") as fout:
            for r in overflow:
                fout.write(json.dumps({
                    "query_id": r["query_id"],
                    "question": r["question"],
                    "passage_ids": r["passage_ids"],
                    "response": None,
                    "error": f"prompt_overflow_n_tokens_{int(r['n_tokens'])}",
                }, ensure_ascii=False) + "\n")

    if not runnable:
        print(f"  [{condition}] no runnable prompts, skipping inference")
        return

    print(
        f"  [{condition}] processing {len(runnable):,} prompts in batches of {batch_size}"
    )

    t_cond_start = time.perf_counter()

    # Open in append mode (so partial progress survives crashes)
    with out_path.open("a", encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(chunked(runnable, batch_size)):
            # Paper-style raw completion: feed user_message directly.
            # No chat template, no [INST] wrapping (the prompt already ends
            # with "Answer:" as the generation trigger).
            prompts = [r["user_message"] for r in batch]

            # Tokenize batch (left-padded for causal LM). truncation=False
            # because we already filtered overflows above — by construction,
            # every prompt here fits within PROMPT_TOKEN_LIMIT.
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=gen_config)

            # Decode each generation, skipping the input prompt portion.
            # input_ids[i].shape[0] is the same for all i because we left-padded.
            input_len = inputs.input_ids.shape[1]
            for i, r in enumerate(batch):
                response_tokens = outputs[i][input_len:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
                fout.write(json.dumps({
                    "query_id": r["query_id"],
                    "question": r["question"],
                    "passage_ids": r["passage_ids"],
                    "response": response,
                }, ensure_ascii=False) + "\n")
            fout.flush()  # ensure progress is on disk after each batch

            # Progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) * batch_size >= len(runnable):
                done_count = min((batch_idx + 1) * batch_size, len(runnable))
                elapsed_min = (time.perf_counter() - t_cond_start) / 60
                rate = done_count / max(elapsed_min, 1e-6)
                eta_min = (len(runnable) - done_count) / max(rate, 1e-6)
                print(
                    f"    [{condition}] {done_count}/{len(runnable)}  "
                    f"({elapsed_min:.1f} min elapsed, {eta_min:.1f} min ETA, "
                    f"~{rate:.0f} prompts/min)",
                    flush=True,
                )

    print(f"  [{condition}] done in {(time.perf_counter() - t_cond_start) / 60:.1f} min")


# %%
# Run all 91 conditions sequentially. Each one saves to its own JSONL.
t_total_start = time.perf_counter()
for condition in CONDITIONS:
    out_path = LLM_OUT_DIR / f"llm_responses_{condition}.jsonl"
    process_condition(condition, df_inputs, out_path)

print(f"\nALL CONDITIONS DONE in {(time.perf_counter() - t_total_start) / 60:.1f} min")
for condition in CONDITIONS:
    out_path = LLM_OUT_DIR / f"llm_responses_{condition}.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in out_path.open("r", encoding="utf-8"))
        size_mb = out_path.stat().st_size / 1024**2
        print(f"  {out_path.name}: {n_lines:,} responses, {size_mb:.1f} MB")

# %% [markdown]
# ## 7 · Sanity inspection
#
# Load the 6 JSONL files, verify row counts, and show a few responses for
# the same query across conditions — at a glance you should see how the
# answer changes as the rerank changes.

# %%
all_responses: list[dict] = []
for condition in CONDITIONS:
    out_path = LLM_OUT_DIR / f"llm_responses_{condition}.jsonl"
    if not out_path.exists():
        continue
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["condition"] = condition
            all_responses.append(d)

df_responses = pd.DataFrame(all_responses)
print(f"loaded responses: {len(df_responses):,} total across {df_responses['condition'].nunique()} conditions")

# %%
df_responses.head(1)

# %%
# For query 0, show the response across all conditions
sample_qid = "0"
sample_view = df_responses[df_responses["query_id"] == sample_qid].sort_values("condition")
print(f"=== query_id={sample_qid} ===")
print(f"question: {queries[int(sample_qid)].get('question', '')}")
print()
for _, row in sample_view.iterrows():
    print(f"--- {row['condition']} ---")
    print(f"top-5 passages: {row['passage_ids']}")
    print(f"response: {row['response']}")
    print()

# %% [markdown]
# ## 8 · Where do dense (retrieval) and KG-rerank disagree?
#
# Build a wide pivot (query × condition) of normalized responses, then
# locate queries where the **retrieval baseline** differs from at least
# one **alpha_* condition** — those are the cases where the KG signal
# actually changed what Llama said.
#
# Two views:
# - **8.1** — aggregate: how many queries see disagreement, distribution
#   of distinct-response-count per query.
# - **8.2** — qualitative: print the top-K most disagreeing queries with
#   their per-condition top-5 passages and responses, plus the gold answer
#   from `queries_curated.jsonl` for a sanity baseline.

# %% [markdown]
# ### 8.1 · Aggregate disagreement stats

# %%
# Normalize responses for fair textual comparison: lowercase + strip.
# Without normalization "Frank Sinatra" vs "frank sinatra" reads as a diff.
def _norm(r: str | None) -> str:
    return (r or "").strip().lower()


# Pivot to wide: index=query_id, columns=condition, values=normalized response.
# Reorder columns so 'retrieval' sits first, then alphas in numeric order.
pivot = (
    df_responses
    .assign(resp_norm=df_responses["response"].apply(_norm))
    .pivot(index="query_id", columns="condition", values="resp_norm")
)
pivot = pivot[["retrieval"] + [c for c in CONDITIONS if c != "retrieval" and c in pivot.columns]]

# How many DISTINCT responses does each query produce across all conditions?
# n_distinct=1 → all conditions agree; >1 → at least one disagreement.
pivot["n_distinct"] = pivot.apply(lambda row: row.iloc[: len(CONDITIONS)].dropna().nunique(), axis=1)

# Separate disagreement type: retrieval vs ANY alpha (the interesting axis
# for "did KG change the answer?")
def _retrieval_vs_alpha(row) -> bool:
    base = row["retrieval"]
    alpha_resps = [row[c] for c in CONDITIONS if c.startswith("alpha_") and c in pivot.columns]
    return any(a != base for a in alpha_resps if pd.notna(a))


pivot["retrieval_vs_alpha"] = pivot.apply(_retrieval_vs_alpha, axis=1)

print(f"total queries: {len(pivot):,}")
print(f"  all {len(CONDITIONS)} conditions agree:       {(pivot['n_distinct'] == 1).sum():>4}")
print(f"  at least one disagreement:     {(pivot['n_distinct'] > 1).sum():>4}")
print(f"  retrieval differs from ≥1 α:   {pivot['retrieval_vs_alpha'].sum():>4}")
print()
print("distribution of n_distinct responses per query:")
print(pivot["n_distinct"].value_counts().sort_index().to_string())

# %% [markdown]
# ### 8.2 · Sample K disagreement cases (most contentious first)
#
# Sort queries by `n_distinct` desc → see first the cases where the
# conditions produce the widest spread of answers (likely the most
# informative cases for understanding what KG-rerank changes).

# %%
K = 6  # number of examples to print

samples = (
    pivot[pivot["retrieval_vs_alpha"]]
    .nlargest(K, "n_distinct")
    .index.tolist()
)
print(f"showing {len(samples)} most disagreeing queries (out of "
      f"{pivot['retrieval_vs_alpha'].sum():,} with retrieval≠α disagreement)\n")

# Lookup gold answers from curated queries
def _gold_for(qid_str: str) -> list[str]:
    return queries[int(qid_str)].get("answers", []) or []


for qid in samples:
    print(f"{'=' * 78}")
    print(f"query_id={qid}  |  n_distinct_responses={int(pivot.loc[qid, 'n_distinct'])}")
    print(f"question: {queries[int(qid)].get('question', '')}")
    gold = _gold_for(qid)
    if gold:
        print(f"gold:     {gold}")
    print()

    qview = (
        df_responses[df_responses["query_id"] == qid]
        .set_index("condition")
        .reindex(["retrieval"] + [c for c in CONDITIONS if c != "retrieval"])
    )
    for cond, row in qview.iterrows():
        if pd.isna(row.get("response")):
            continue
        print(f"  [{cond:>10}]  top-5: {row['passage_ids']}")
        print(f"  {' ' * 12}  resp:  {row['response']!r}")
    print()