"""T2 (V13): which communities of the authors' k-core hierarchy carry reports?  Final graph (order 0), same input as
v2/b2_kcore_authors.py.  Per level: number of communities, how many are singletons, size range.  The "reports"
column of Table 3 counts every community on every level (as for GraphRAG, whose every community carries a report).
Output: analysis/v12/t2_kcore_levels.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A / "v2"), str(A), str(A / "wikidata")]
import b2_kcore_authors as ka  # noqa: E402
import entity_graph_eval as eg  # noqa: E402

out = {}
for g in ("entity", "wikidata"):
    if g == "wikidata":
        import stream_eval as se
        import wd_eval as wd
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, 0)
    else:
        d, node_t, edge_t, T = eg.stream(0)
    n, E = len(node_t), d["edges"]
    ok = (edge_t >= 0) & (edge_t < T)
    el = [(int(a), int(b)) for a, b in E[:, ok].T]
    out[g] = {}
    for ct in ("RkH", "M2hC", "MRC"):
        ref = ka.authors_kcore(el, n, ct)
        lv = sorted({k[0] for k in ref})
        per = []
        for l in lv:
            sz = [len(m) for k, m in ref.items() if k[0] == l]
            per.append(dict(level=int(l), communities=len(sz), singletons=int(sum(s == 1 for s in sz)),
                            min=int(min(sz)), median=float(np.median(sz)), max=int(max(sz))))
        out[g][ct] = dict(levels=per, reports=len(ref), singletons=int(sum(p["singletons"] for p in per)))
        print(g, ct, out[g][ct]["reports"], "singletons", out[g][ct]["singletons"], [(p["communities"], p["singletons"], p["min"]) for p in per], flush=True)
(HERE / "t2_kcore_levels.json").write_text(json.dumps(out, indent=1))
