"""W1 (V16) summary: small-batch costs of the hierarchical frontier maintainer (ours) and the size-bound check
(leaves above max_cluster_size at the final version: ours, mean over 5 orders, vs GraphRAG's fresh clustering of the
same final graph).  Output: analysis/v16/w1_summary.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE), str(A / "wikidata")]
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import hier_frontier as hf  # noqa: E402
import noise_hier as nh  # noqa: E402

out = {"batch": {}}
for B in (1, 10, 100):
    v = [json.loads((HERE / "w1" / f"batch_b{B}_s{s}.json").read_text())[str(s)]["hier_frontier"]["cost"]["0.9|1"] for s in range(3)]
    out["batch"][str(B)] = float(np.mean(v))
for g in ("entity", "wikidata"):
    ours = [json.loads((HERE / "w1" / f"{g}_s{s}.json").read_text())[str(s)]["hier_frontier"]["validation"]["oversize_leaves"][-1] for s in range(5)]
    if g == "wikidata":
        import stream_eval as se
        import wd_eval as wd
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, 0)
    else:
        d, node_t, edge_t, T = eg.stream(0)
    E = d["edges"]
    ok = (edge_t >= 0) & (edge_t < T)
    el = [(int(a), int(b)) for a, b in E[:, ok].T]
    c = nh.cluster(el, len(node_t))
    _, keep = gh.lcc_edges(el, np.arange(len(node_t)))
    out[g] = dict(oversize_ours=float(np.mean(ours)), oversize_graphrag=hf.validate(c, keep, gh.MAXC)["oversize_leaves"])
    print(g, out[g], flush=True)
print(out["batch"])
(HERE / "w1_summary.json").write_text(json.dumps(out, indent=1))
