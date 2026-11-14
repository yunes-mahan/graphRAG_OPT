"""T1 (V13): why a full GraphRAG re-index covers < 100 % of the entities: GraphRAG clusters only the largest connected
component (use_lcc).  Final graph (same for every order) of the entity graph and Wikidata: number of present entities,
size of the largest connected component (as graphrag_hier.lcc_edges).  Output: analysis/v12/t1_lcc.json"""
import json
import sys
from pathlib import Path

import igraph as ig
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata")]
import entity_graph_eval as eg  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402

out = {}
for g in ("entity", "wikidata"):
    if g == "entity":
        d, node_t, edge_t, T = eg.stream(0)
    else:
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, 0)
    t = T - 1
    nodes = np.where((node_t >= 0) & (node_t <= t))[0]
    el = [(int(a), int(b)) for a, b in d["edges"][:, edge_t <= t].T]
    big = max(ig.Graph(n=len(node_t), edges=el).connected_components(), key=len)
    out[g] = dict(entities=int(len(nodes)), lcc=int(len(set(big) & set(nodes.tolist()))))
    out[g]["share"] = out[g]["lcc"] / out[g]["entities"]
    print(g, out[g])
(HERE / "t1_lcc.json").write_text(json.dumps(out, indent=1))
