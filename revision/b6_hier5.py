"""B6: Table 2 (GraphRAG hierarchy under maintenance) with 5 instead of 3 document orders, structural metrics only
(modelled report-token cost, Q_leaf, Q_top, coverage, reports).  Orders 0-2 were recomputed by graphrag_hier.py into
v2/hier5/ and are identical to the stored graphrag_hier*.json (checked below); orders 3-4 are new.
Writes v2/graphrag_hier_entity_5orders.json, v2/graphrag_hier_wikidata_5orders.json (same layout as graphrag_hier.json)
and v2/b6_hier5.json (95% t-intervals over orders for every Table 2 cell).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent
out = {}
for g, old_fn in (("entity", "graphrag_hier.json"), ("wikidata", "graphrag_hier_wikidata.json")):
    merged = {}
    for s in range(5):
        merged.update(json.loads((HERE / "hier5" / f"graphrag_hier_{g}_s{s}.json").read_text()))
    old = json.loads((A / old_fn).read_text())
    for s in old:                                       # reproduction check of the 3 stored orders
        for m in old[s]:
            assert old[s][m]["cost"] == merged[s][m]["cost"], (g, s, m)
            assert old[s][m]["Qleaf"] == merged[s][m]["Qleaf"] and old[s][m]["Qtop"] == merged[s][m]["Qtop"], (g, s, m)
    (HERE / f"graphrag_hier_{g}_5orders.json").write_text(json.dumps(merged, indent=1))
    cells = {}
    for m in merged["0"]:
        def ci(f):
            v = np.array([f(merged[s][m]) for s in merged], float)
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            return [float(v.mean()), float(h)]
        cells[m] = {**{f"cost_{t}": ci(lambda r, t=t: r["cost"][f"{t}|1"]) for t in ("0.9", "0.7", "0.5", "0.0")},
                    "Qleaf": ci(lambda r: r["Qleaf"]), "Qtop": ci(lambda r: r["Qtop"]),
                    "coverage_final": ci(lambda r: r["coverage_final"]["1"]), "reports_final": ci(lambda r: r["reports_final"]["1"])}
    out[g] = dict(n_orders=len(merged), cells=cells,
                  max_halfwidth=dict(cost=max(c[k][1] for c in cells.values() for k in c if k.startswith("cost")),
                                     Q=max(max(c["Qleaf"][1], c["Qtop"][1]) for c in cells.values()),
                                     coverage=max(c["coverage_final"][1] for c in cells.values())))
    print(g, out[g]["max_halfwidth"])
(HERE / "b6_hier5.json").write_text(json.dumps(out, indent=1))
