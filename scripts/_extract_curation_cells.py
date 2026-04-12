"""
Extract specific cell sources from answer_curation.ipynb.
Writes each cell source to a separate file in scripts/_cells/ directory.

Run: .venv/Scripts/python.exe scripts/_extract_curation_cells.py
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
nb_path = root / "answer_curation.ipynb"
out_dir = root / "scripts" / "_cells"
out_dir.mkdir(exist_ok=True)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

target_ids = [
    "287ddd56",
    "e85c000d",
]

for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in target_ids:
        src_raw = cell.get("source", [])
        if isinstance(src_raw, list):
            src = "".join(src_raw)
        else:
            src = src_raw
        # Write to file
        out_file = out_dir / f"curation_{cid}.py.txt"
        out_file.write_text(src, encoding="utf-8")
        # Also print to stdout
        print(f"{'=' * 60}")
        print(f"Cell ID: {cid}  ({len(src)} chars)")
        print(f"{'=' * 60}")
        print(src)
        print()

print("Done.")