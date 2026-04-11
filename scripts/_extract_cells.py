"""
Extract specific cell sources from answer_preparation.ipynb.
Writes each cell source to a separate file in scripts/_cells/ directory.

Run: python scripts/_extract_cells.py
  or: .venv/Scripts/python.exe scripts/_extract_cells.py
"""
import json, os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
nb_path = root / "answer_preparation.ipynb"
out_dir = root / "scripts" / "_cells"
out_dir.mkdir(exist_ok=True)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

target_ids = [
    "c1b2c3d4",    # imports
    "e1b2c3d4",    # config
    "i1b2c3d4",    # encode queries (section 4)
    "j1b2c3d4",    # mean_pool helper
    "k1b2c3d4",    # encode queries batch
    "o1b2c3d4",    # per-shard retrieval (section 6)
    "q1b2c3d4",    # merging cross-shard (section 7)
    "hzirm4qm2e",  # entity linking loop (section 9.4)
    "zg8zx2aikfb",  # merge entity chunks (section 9.5)
]

for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in target_ids:
        src_raw = cell.get("source", [])
        if isinstance(src_raw, list):
            src = "".join(src_raw)
        else:
            src = src_raw
        out_file = out_dir / f"{cid}.py.txt"
        out_file.write_text(src, encoding="utf-8")
        print(f"Wrote {out_file.name} ({len(src)} chars)")

print("Done.")
