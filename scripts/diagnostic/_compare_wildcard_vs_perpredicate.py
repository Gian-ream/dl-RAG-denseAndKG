"""One-shot: per-predicate count diff, wildcard parquet vs canonical edges.parquet.

Corroborates PROJECT_NOTES 4.8 / hdt_export_per_predicate.py docstring: the
aborted wildcard export is missing rows scattered across predicates (e.g.
15,781 on P2860), not just a single unflushed buffer chunk.
"""
import duckdb

con = duckdb.connect()
df = con.execute("""
    WITH o AS (SELECT predicate, COUNT(*) c
               FROM read_parquet('data/db/edges_v1_wildcard_partial.parquet')
               GROUP BY 1),
         n AS (SELECT predicate, COUNT(*) c
               FROM read_parquet('data/db/edges.parquet')
               GROUP BY 1)
    SELECT COALESCE(n.predicate, o.predicate) AS pred,
           COALESCE(o.c, 0) AS old_c,
           COALESCE(n.c, 0) AS new_c,
           COALESCE(n.c, 0) - COALESCE(o.c, 0) AS delta
    FROM n FULL JOIN o ON n.predicate = o.predicate
""").df()

print("predicates total          :", len(df))
print("with deficit (new > old)  :", int((df.delta > 0).sum()))
print("with surplus (old > new)  :", int((df.delta < 0).sum()))
print("total deficit             :", int(df[df.delta > 0].delta.sum()))
print("total surplus             :", int(-df[df.delta < 0].delta.sum()))
print("\nP2860:")
print(df[df.pred.str.contains("P2860", regex=False)].to_string(index=False))
print("\ntop 10 deficits:")
print(df.nlargest(10, "delta").to_string(index=False))