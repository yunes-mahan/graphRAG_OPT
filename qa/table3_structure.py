"""Hierarchy structure (levels, top-level communities, largest top-level community) for the Table 3 rows that
stability_variants.py / consensus_runs.py did not record: consensus (threshold, reattached) with 10 and 50 runs,
k-core (BFS split) and warm start, on the final graph of order 0.  Writes table3_structure.json."""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import consensus_lib as cl  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import stability_variants as sv  # noqa: E402


def struct(c):
    lv = Counter(k[0] for k in c)
    tops = [m for k, m in c.items() if k[0] == 0]
    return dict(levels=len(lv), n_top=len(tops), largest_top=max(len(m) for m in tops))


out = {}
for graph in ("entity", "wikidata"):
    if graph == "wikidata":
        import stream_eval as se
        import wd_eval as wd
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, 0)
    else:
        d, node_t, edge_t, T = eg.stream(0)
    n, E = len(node_t), d["edges"]
    ok = (edge_t >= 0) & (edge_t < T)
    el = [(int(a), int(b)) for a, b in E[:, ok].T]
    out[graph] = {"threshold10": struct(cl.consensus(el, n, 10, True)), "threshold50": struct(cl.consensus(el, n, 50, True)),
                  "kcore": struct(sv.kcore(el, n)), "warm": struct(sv.leiden(el, n))}
    print(graph, out[graph], flush=True)
(HERE / "table3_structure.json").write_text(json.dumps(out, indent=1))
