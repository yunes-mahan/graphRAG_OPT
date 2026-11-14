"""A3 check: why Table 2 and Table 3 report different Q for GraphRAG's Leiden on the same graphs.

Table 2 (graphrag_hier.py): Q_leaf / Q_top averaged over ALL versions of a stream, then over 3 orders.
Table 3 (stability_variants.py): Q_leaf / Q_top of the FINAL graph only (order 0).
This script recomputes GraphRAG's hierarchical Leiden (fresh re-index) on every version of orders 0-2 with the
Table 3 quality function, and reports the per-version values, their mean, and the final-version value.
Output: analysis/v2/a3_q_per_version.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "wikidata"))
import entity_graph_eval as eg  # noqa: E402
import stability_variants as sv  # noqa: E402

out = {}
for graph in ("entity", "wikidata"):
    out[graph] = {}
    for seed in (0, 1, 2):
        if graph == "wikidata":
            import stream_eval as se
            import wd_eval as wd
            d = wd.build()[0]
            node_t, edge_t, T = se.schedule(d, seed)
        else:
            d, node_t, edge_t, T = eg.stream(seed)
        n, E = len(node_t), d["edges"]
        per = []
        for t in range(T):
            ok = (edge_t >= 0) & (edge_t <= t)
            el = [(int(a), int(b)) for a, b in E[:, ok].T]
            c = sv.leiden(el, n)
            q = sv.quality(c, el, n)
            per.append(dict(t=t, leaf=q["leaf"], top=q["top"], reports=q["reports"]))
        out[graph][seed] = dict(per_version=per, mean_leaf=float(np.nanmean([p["leaf"] for p in per])),
                                mean_top=float(np.nanmean([p["top"] for p in per])),
                                final_leaf=per[-1]["leaf"], final_top=per[-1]["top"], final_reports=per[-1]["reports"])
        print(graph, seed, {k: v for k, v in out[graph][seed].items() if k != "per_version"}, flush=True)
    out[graph]["mean_over_orders"] = dict(
        mean_leaf=float(np.mean([out[graph][s]["mean_leaf"] for s in (0, 1, 2)])),
        mean_top=float(np.mean([out[graph][s]["mean_top"] for s in (0, 1, 2)])))
    print(graph, out[graph]["mean_over_orders"], flush=True)
(HERE / "a3_q_per_version.json").write_text(json.dumps(out, indent=1, default=float))
