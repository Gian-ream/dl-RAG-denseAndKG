"""One-shot check: Table 1 in relazione/bozza/main.tex vs accuracy outputs in 09_llm_judge.ipynb.

Parses the accuracy summary printed by notebook 09 (lines like
"alpha_3_dist2_thr5k      372     1000     0.372") and compares every cell of
the LaTeX table against it. The dist-1 row ("any") is checked against all six
thr variants, which must be identical (distance 1 has no bridge).
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- 1. Accuracy per condition from notebook 09 outputs -------------------
nb = json.loads((REPO / "09_llm_judge.ipynb").read_text(encoding="utf-8"))
acc: dict[str, str] = {}
row_re = re.compile(r"^\s*(retrieval|alpha_\d_dist\d_thr\w+)\s+(\d+)\s+1000\s+(0\.\d{3})\s*$")
for cell in nb["cells"]:
    for out in cell.get("outputs", []):
        for line in out.get("text", []):
            m = row_re.match(line.rstrip("\n"))
            if m:
                name, val = m.group(1), m.group(3)
                # Same condition printed in several cells: must always agree.
                if name in acc and acc[name] != val:
                    print(f"INTERNAL MISMATCH in notebook: {name} {acc[name]} vs {val}")
                acc[name] = val

print(f"conditions parsed from notebook: {len(acc)} (expected 91 = baseline + 90)")

# --- 2. Table 1 from main.tex, transcribed verbatim -----------------------
ALPHAS = ["1", "3", "5", "7", "9"]
TEX = {
    ("1", "any"):   [".353", ".344", ".343", ".316", ".322"],
    ("2", "500"):   [".362", ".358", ".361", ".348", ".354"],
    ("2", "1k"):    [".364", ".361", ".358", ".343", ".350"],
    ("2", "2k"):    [".360", ".371", ".357", ".337", ".341"],
    ("2", "5k"):    [".357", ".372", ".362", ".337", ".337"],
    ("2", "10k"):   [".364", ".365", ".357", ".334", ".329"],
    ("2", "inf"):   [".354", ".343", ".311", ".298", ".295"],
    ("3", "500"):   [".354", ".344", ".341", ".329", ".335"],
    ("3", "1k"):    [".346", ".355", ".342", ".322", ".310"],
    ("3", "2k"):    [".354", ".349", ".316", ".303", ".293"],
    ("3", "5k"):    [".366", ".351", ".310", ".292", ".290"],
    ("3", "10k"):   [".361", ".362", ".321", ".306", ".304"],
    ("3", "inf"):   [".347", ".333", ".314", ".314", ".318"],
}
THRS = ["500", "1k", "2k", "5k", "10k", "inf"]

# --- 3. Compare ------------------------------------------------------------
n_checked, mismatches = 0, []
for (dist, thr), values in TEX.items():
    for a, tex_val in zip(ALPHAS, values):
        # "any" row covers all six thresholds: every variant must match.
        thr_list = THRS if thr == "any" else [thr]
        for t in thr_list:
            cond = f"alpha_{a}_dist{dist}_thr{t}"
            nb_val = acc.get(cond)
            n_checked += 1
            if nb_val is None:
                mismatches.append(f"{cond}: NOT FOUND in notebook")
            elif "0" + tex_val != nb_val:
                mismatches.append(f"{cond}: tex={tex_val} nb={nb_val}")

print(f"baseline: tex=0.364 nb={acc.get('retrieval')}")
print(f"cells checked: {n_checked} (65 tex cells -> 90 conditions)")
if mismatches:
    print(f"\nMISMATCHES ({len(mismatches)}):")
    for m in mismatches:
        print(" ", m)
else:
    print("ALL MATCH: every Table 1 cell equals the notebook 09 output.")