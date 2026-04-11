"""
Patch answer_preparation.ipynb: insert query subset selection (section 3b)
right after loading queries, BEFORE encoding and retrieval.

Sections 4-8 will then operate on the 1K subset natively.
Section 9.1 needs no change — top100_merged.parquet will already be subset-only.

Run: uv run python scripts/patch_answer_preparation.py
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("answer_preparation.ipynb")


def make_markdown_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "source": source,
        "metadata": {},
    }


def make_code_cell(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "source": source,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
    }


# --- New cells to insert after section 3 (cell g1b2c3d4 — Load Queries) ---

MARKDOWN_3B = make_markdown_cell(
    "subset_md_3b",
    (
        "## 3b. Query Subset Selection\n"
        "\n"
        "To make Wikidata 3-hop exploration tractable, we select **1,000 queries with the\n"
        "fewest total unique entities** (question + answer QIDs combined) and discard the rest\n"
        "**before** encoding and retrieval.\n"
        "\n"
        "**Why this filter:** every unique QID will require a SPARQL query to Wikidata for its\n"
        "3-hop neighborhood. Fewer QIDs per query = lighter load on the SPARQL endpoint\n"
        "and smaller local KG subgraphs. This is a computational-weight filter, not a\n"
        "complexity filter — we keep the lightest queries regardless of topic.\n"
        "\n"
        "**Minimum possible:** the filtering step (`nq_filtering.ipynb`) guarantees >= 1\n"
        "entity in the question AND >= 1 in every answer variant, so the minimum is **2\n"
        "unique QIDs** per query."
    ),
)

CODE_3B = make_code_cell(
    "subset_code_3b",
    (
        "SUBSET_SIZE = 1_000\n"
        "\n"
        "# --- 1. Count unique QIDs per query (question + all answer variants) ---\n"
        "entity_counts = []  # list of (original_index, n_unique_qids)\n"
        "for i, q in enumerate(queries):\n"
        "    qids = set(q[\"question_qids\"])\n"
        "    for variant in q[\"answer_variant_qids\"]:\n"
        "        qids.update(variant)\n"
        "    entity_counts.append((i, len(qids)))\n"
        "\n"
        "# --- 2. Distribution (full dataset, before filtering) ---\n"
        "from collections import Counter\n"
        "count_dist = Counter(c for _, c in entity_counts)\n"
        "print(\"Entity count distribution (unique QIDs per query):\")\n"
        "for k in sorted(count_dist):\n"
        "    print(f\"  {k} QIDs: {count_dist[k]:>6,} queries  ({count_dist[k]/len(entity_counts)*100:.1f}%)\")\n"
        "\n"
        "# --- 3. Select 1K with fewest entities ---\n"
        "# Stable sort: at equal count, original order is preserved\n"
        "entity_counts.sort(key=lambda x: x[1])\n"
        "subset_pairs = entity_counts[:SUBSET_SIZE]\n"
        "\n"
        "# Restore original order for reproducibility\n"
        "subset_pairs.sort(key=lambda x: x[0])\n"
        "subset_orig_ids = [idx for idx, _ in subset_pairs]\n"
        "max_entity_count = max(c for _, c in subset_pairs)\n"
        "\n"
        "print(f\"\\nSelected {len(subset_orig_ids):,} queries (entity count: 2 – {max_entity_count})\")\n"
        "\n"
        "# --- 4. Save subset queries with original index for traceability ---\n"
        "subset_jsonl_path = OUTPUT_DIR / \"queries_subset.jsonl\"\n"
        "with open(subset_jsonl_path, \"w\", encoding=\"utf-8\") as f:\n"
        "    for orig_idx in subset_orig_ids:\n"
        "        record = queries[orig_idx].copy()\n"
        "        record[\"original_query_id\"] = orig_idx\n"
        "        f.write(json.dumps(record, ensure_ascii=False) + \"\\n\")\n"
        "print(f\"Saved {subset_jsonl_path}\")\n"
        "\n"
        "# --- 5. Trim in-place: downstream cells see only the subset ---\n"
        "queries     = [queries[i] for i in subset_orig_ids]\n"
        "query_texts = [q[\"question\"] for q in queries]\n"
        "n_queries   = len(queries)\n"
        "\n"
        "print(f\"\\nqueries trimmed to {n_queries:,} — encoding and retrieval will use this subset\")\n"
        "print(f\"Example: '{query_texts[0]}'\")"
    ),
)


def patch_notebook():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- Find insertion point: after cell g1b2c3d4 (section 3 — Load Queries) ---
    insert_idx = None
    for i, cell in enumerate(cells):
        if cell.get("id") == "g1b2c3d4":
            insert_idx = i + 1
            break

    if insert_idx is None:
        raise RuntimeError("Cell g1b2c3d4 (section 3 — Load Queries) not found")

    # --- Check idempotency: don't insert if already patched ---
    if any(c.get("id") == "subset_code_3b" for c in cells):
        print("Already patched (cell subset_code_3b exists). Nothing to do.")
        return

    # --- Insert new cells ---
    cells.insert(insert_idx, MARKDOWN_3B)
    cells.insert(insert_idx + 1, CODE_3B)

    # --- Save ---
    NOTEBOOK_PATH.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Patched {NOTEBOOK_PATH}")
    print(f"  Inserted 2 cells (3b markdown + code) at position {insert_idx}")
    print()
    print("Next steps:")
    print("  1. Open the notebook in Jupyter/PyCharm")
    print("  2. Re-run from section 3 onwards (the subset trims queries/query_texts)")
    print("  3. Sections 4-8 will encode and retrieve only the 1K subset")
    print("  4. Section 9 works as-is on the resulting top100_merged.parquet")


if __name__ == "__main__":
    patch_notebook()