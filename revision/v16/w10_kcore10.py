"""W10 (V16): size-bound confound. The authors' k-core code fixes MAX_CLUSTER_SIZE = 20 (kcore_cluster_graph.py, line
230), GraphRAG uses 10. Variant (ours): the same code with the constant set to 10 (no other change), RkH, final graph of
order 0; 0.1 % edge deletion (3 draws, same rng as v2/b2_kcore_authors.py) and the real last update, tau = 0.9.
Output: analysis/v16/w10_kcore10.json"""
import builtins
import json
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A / "v2"), str(A), str(A / "wikidata"), str(A / "v2" / "external" / "kcore_graphrag")]
import b2_kcore_authors as ka  # noqa: E402,F401  (sets up the authors' package import exactly as before)
import entity_graph_eval as eg  # noqa: E402
import noise_hier as nh  # noqa: E402

src_f = A / "v2" / "external" / "kcore_graphrag" / "graphrag" / "index" / "operations" / "kcore_cluster_graph.py"
src = src_f.read_text()
assert src.count("MAX_CLUSTER_SIZE, MIN_CLUSTER_SIZE = 20, 1") == 1
mod = types.ModuleType("kcore10")
mod.__package__ = "graphrag.index.operations"
exec(compile(src.replace("MAX_CLUSTER_SIZE, MIN_CLUSTER_SIZE = 20, 1", "MAX_CLUSTER_SIZE, MIN_CLUSTER_SIZE = 10, 1"),
             str(src_f), "exec"), mod.__dict__)


def kcore10(edges, n):
    g = nx.Graph()
    g.add_edges_from(edges)
    p = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        res = mod.kcore_cluster_graph(g, use_lcc=True, cluster_type="RkH")
    finally:
        builtins.print = p
    out = {}
    for level, cid, _pid, nodes in res:
        key = (int(level) - 1, cid)
        out[key] = frozenset(int(v) for v in nodes) | out.get(key, frozenset())
    return out


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
    before = [e for e, t in zip(el, edge_t[ok]) if t < T - 1]
    rng = np.random.default_rng(0)
    drops = [[e for e, k in zip(el, rng.random(len(el)) >= 0.001) if k] for _ in range(3)]
    ref = kcore10(el, n)
    out[graph] = dict(drop=[nh.rewrite(ref, kcore10(x, n), 0.9) for x in drops], last=nh.rewrite(kcore10(before, n), ref, 0.9),
                      reports=len(ref), max_size_leaf=max(len(m) for m in ref.values() if not any(c < m for c in ref.values())))
    print(graph, out[graph], flush=True)
(HERE / "w10_kcore10.json").write_text(json.dumps(out, indent=1))
