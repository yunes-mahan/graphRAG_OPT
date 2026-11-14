"""B2: the authors' own k-core hierarchy (Hossain & Sariyuce, KDD 2026; github.com/jakir-sust/Kcore-GraphRAG @ ae7b2ef,
fetched unmodified by fetch_external.sh) next to our BFS-split stand-in, with exactly the Table 3 protocol of
stability_variants.py: final graph of order 0 (entity graph, Wikidata), 3 random 0.1% edge deletions, and the real last
update; rewritten report-token share at tau = 0.9 after level-by-level matching; structure and quality of the hierarchy.

Their kcore_cluster_graph(graph, use_lcc=True, cluster_type) is called as in their GraphRAG pipeline (largest
connected component, MAX_CLUSTER_SIZE = 20 hard-coded by the authors) for the three heuristics RkH, M2hC, MRC.
Their levels start at 1 (coarsest); we map level L to our level L-1 (0 = top), as GraphRAG lists the coarsest level first.
Integer node ids are used, so the authors' hash()-based tie-breaking is deterministic.
Their module imports plotly.graph_objects without using it; an empty stub is registered for that import only.
Output: analysis/v2/b2_kcore_authors.json
"""
import json
import sys
import time
import types
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "wikidata"))
sys.path.insert(0, str(HERE / "external" / "kcore_graphrag"))
for m in ("plotly", "plotly.graph_objects"):            # unused import in the authors' module
    sys.modules.setdefault(m, types.ModuleType(m))
sys.modules["plotly"].graph_objects = sys.modules["plotly.graph_objects"]
import builtins  # noqa: E402
from graphrag.index.operations import kcore_cluster_graph as kc  # noqa: E402

import entity_graph_eval as eg  # noqa: E402
import noise_hier as nh  # noqa: E402
import stability_variants as sv  # noqa: E402

_print = builtins.print


def authors_kcore(edges, n, cluster_type):
    g = nx.Graph()
    g.add_edges_from(edges)
    builtins.print = lambda *a, **k: None      # the authors' function prints progress for every queue entry
    try:
        res = kc.kcore_cluster_graph(g, use_lcc=True, cluster_type=cluster_type)
    finally:
        builtins.print = _print
    out = {}
    for level, cid, _pid, nodes in res:
        key = (int(level) - 1, cid)
        out[key] = frozenset(int(v) for v in nodes) | out.get(key, frozenset())
    return out


def main():
    res = json.loads((HERE / "b2_kcore_authors.json").read_text()) if (HERE / "b2_kcore_authors.json").exists() else {}
    for graph in ("entity", "wikidata"):
        if all(ct in res.get(graph, {}) for ct in ("RkH", "M2hC", "MRC")):
            continue   # resume: graph complete
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
        rng = np.random.default_rng(0)                # identical draws to stability_variants.py
        drops = []
        for _ in range(3):
            keep = rng.random(len(el)) >= 0.001
            drops.append([e for e, k in zip(el, keep) if k])
        res.setdefault(graph, {})
        # environment check: our BFS stand-in must reproduce stability_variants.json (Table 3) in this venv
        stored = json.loads((HERE.parent / "stability_variants.json").read_text())[graph]["kcore"]
        ref_bfs = sv.kcore(el, n)
        chk = dict(drop=[nh.rewrite(ref_bfs, sv.kcore(x, n), 0.9) for x in drops], last=nh.rewrite(sv.kcore(before, n), ref_bfs, 0.9))
        res[graph]["bfs_standin_check"] = dict(recomputed=chk, stored=dict(drop=stored["drop"], last=stored["last"]),
                                               identical=bool(np.allclose(chk["drop"], stored["drop"]) and np.isclose(chk["last"], stored["last"])))
        print(graph, "BFS stand-in check", res[graph]["bfs_standin_check"]["identical"], flush=True)
        for ct in ("RkH", "M2hC", "MRC"):
            if ct in res[graph]:
                continue   # resume
            t0 = time.time()
            ref = authors_kcore(el, n, ct)
            secs = time.time() - t0
            again = authors_kcore(el, n, ct)
            q = sv.quality(ref, el, n)
            q["reproducible"] = again == ref
            q["seconds"] = secs
            q["drop"] = [nh.rewrite(ref, authors_kcore(x, n, ct), 0.9) for x in drops]
            q["last"] = nh.rewrite(authors_kcore(before, n, ct), ref, 0.9)
            lv = sorted({k[0] for k in ref})
            q["communities_per_level"] = [sum(1 for k in ref if k[0] == l) for l in lv]
            q["second_level"] = dict(n=q["communities_per_level"][1] if len(lv) > 1 else None)
            if len(lv) > 1:          # quality of the first level below the single-community top
                sub = {k: m for k, m in ref.items() if k[0] <= 1}
                q["second_level"]["Q"] = sv.quality({(0 if k[0] == 1 else -1, k[1]): m for k, m in sub.items() if k[0] == 1}, el, n)["top"]
            res[graph][ct] = q
            print(graph, ct, {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in q.items()
                              if k != "communities_per_level"}, flush=True)
            (HERE / "b2_kcore_authors.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
