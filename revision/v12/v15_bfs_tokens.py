"""V15/F2: Table 3's perturbation columns are relative to the method's OWN report tokens (noise_hier.rewrite), the
stream column to a GraphRAG re-index (graphrag_hier.py).  The BFS k-core stand-in writes many small reports, so its
total report tokens are a fraction of GraphRAG's.  Final graph (order 0): tokens of all reports of the BFS stand-in and of
GraphRAG's hierarchical Leiden, and their ratio.  Output: analysis/v12/v15_bfs_tokens.json"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata")]
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import stability_variants as sv  # noqa: E402

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
    tb = sum(gh.tok(len(m)) for m in sv.kcore(el, n).values())
    tl = sum(gh.tok(len(m)) for m in sv.leiden(el, n).values())
    out[g] = dict(bfs_tokens=tb, graphrag_tokens=tl, ratio=tb / tl)
    print(g, out[g])
(HERE / "v15_bfs_tokens.json").write_text(json.dumps(out, indent=1))
