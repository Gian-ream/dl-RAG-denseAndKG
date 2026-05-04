"""Build SPARQL test queries with varying VALUES sizes.

Reads the predicates CSV and generates one .sparql file per batch size,
each containing a count-with-group-by query for a fixed test entity over
the first N predicates from the CSV.

Test entity is Q5 ("human") — the worst-case hub on Wikidata, intentionally
chosen to stress-test the endpoint rather than to confirm easy cases.
"""
import csv
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # scripts/legacy/file -> repo root
CSV_PATH     = PROJECT_ROOT / "data" / "Wikidata_service" / "queryPredicati.csv"
OUT_DIR      = PROJECT_ROOT / "data" / "Wikidata_service" / "test_queries"

# Test entity: Q5 (human). Worst-case hub: ~10M+ incoming P31 edges.
# We pick this on purpose to find where the batched VALUES query breaks.
ENTITY = "Q5"

# Batch sizes to probe.
SIZES = [100, 200, 500, 1000, 2000, 5000]

# Query template. The FILTER restricts to Q-entity neighbours (concepts),
# excluding literals (dates, strings, IDs, URLs). The UNION captures both
# outgoing and incoming edges (non-directional graph).
QUERY_TEMPLATE = """\
PREFIX wd:  <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

# Test query: count neighbours of {entity} per predicate.
# Predicates: first {n} from queryPredicati.csv.
# Direction: outgoing UNION incoming (non-directional graph).
# Target filter: only Wikidata items (Q...), no literals.

SELECT ?p (COUNT(?n) AS ?cnt) WHERE {{
  VALUES ?p {{ {values} }}
  {{ wd:{entity} ?p ?n . }} UNION {{ ?n ?p wd:{entity} . }}
  FILTER(STRSTARTS(STR(?n), "http://www.wikidata.org/entity/Q"))
}}
GROUP BY ?p
"""


def entity_uri_to_wdt(uri: str) -> str:
    """Convert 'http://www.wikidata.org/entity/Pxxx' to 'wdt:Pxxx'.

    The CSV stores property URIs in the entity namespace (the property's
    own item page), but in a triple pattern we need the 'truthy direct'
    form (wdt:) which is the simplified edge to the value.
    """
    pid = uri.rsplit("/", 1)[-1]  # take last path segment
    return f"wdt:{pid}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read all predicate URIs from the CSV (single column 'property')
    with CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        preds = [row["property"].strip() for row in reader if row.get("property")]

    print(f"Loaded {len(preds):,} predicates from {CSV_PATH.name}")

    # Convert to wdt: short form
    wdt_preds = [entity_uri_to_wdt(p) for p in preds]

    for n in SIZES:
        if n > len(wdt_preds):
            print(f"  skip {n}: only {len(wdt_preds)} predicates available")
            continue
        values = " ".join(wdt_preds[:n])
        query = QUERY_TEMPLATE.format(entity=ENTITY, n=n, values=values)
        out_path = OUT_DIR / f"query_{n:05d}.sparql"
        out_path.write_text(query, encoding="utf-8")
        print(f"  {out_path.name}: {n} predicates, {len(query):,} chars")

    print(f"\nOutput dir: {OUT_DIR}")


if __name__ == "__main__":
    main()