"""Can instability be engineered away?  Reproducibility (same input -> same output) vs stability (similar input ->
similar output), on the final graph of the entity and Wikidata streams (order 0), largest connected component.

Hierarchical clusterings, every community on every level carries a report (as in Table 2):
  leiden    : GraphRAG's hierarchical Leiden (max cluster size 10, GraphRAG's fixed seed)          [deterministic]
  consensus : consensus clustering (Lancichinetti & Fortunato 2012): 10 hierarchical Leiden runs (seeds 0..9);
              edges weighted by the share of runs that put both endpoints in the same leaf, edges below 0.5 dropped,
              then hierarchical Leiden (GraphRAG settings) on the weighted consensus graph            [deterministic]
  kcore     : k-core hierarchy (Hossain & Sariyuce 2026): connected components of every k-core level; components
              above 10 entities split into chunks of 10 along a BFS order from the smallest node id      [deterministic]
  warm      : hierarchical Leiden warm-started from the previous clustering's leaves (stability-aware)
Perturbations: delete a random 0.1% of the edges (3 draws), and the real last update (graph before vs after).
Each perturbed clustering is matched level by level to the reference (one-to-one overlap) and the tau = 0.9 rule is
applied; we report the share of report tokens rewritten, plus reports, leaf/top modularity and coverage.
Writes stability_variants.json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import igraph as ig
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import noise_hier as nh  # noqa: E402


def leaves(comms):
    leaf = {}
    for (l, c), m in comms.items():
        for v in m:
            if l >= leaf.get(v, (-1, None))[0]:
                leaf[v] = (l, c)
    return leaf


def leiden(edges, n, seed=gh.SEED, weights=None, start=None):
    if weights is not None:            # consensus graph: edges already restricted to the original LCC; keep all components
        e = [(str(a), str(b), float(weights.get((a, b), 1.0))) for a, b in edges]
        gh.SEED = seed
        res = gh.gn.hierarchical_leiden(e, max_cluster_size=gh.MAXC, seed=seed, use_modularity=True)
        gh.SEED = 0xDEADBEEF
        lev = defaultdict(dict)
        for h in res:
            lev[h.level][int(h.node)] = h.cluster
        return gh.communities(gh.complete_levels(dict(lev)))
    lcc, _ = gh.lcc_edges(edges, np.arange(n))
    if start is not None:
        return gh.communities(gh.complete_levels(gh.hier(lcc, list(range(n)), start)[0]))
    return nh.cluster(edges, n, seed)


def consensus(edges, n, runs=10):
    lcc, _ = gh.lcc_edges(edges, np.arange(n))
    agree = defaultdict(int)
    for s in range(runs):
        lf = leaves(nh.cluster(edges, n, s))
        for a, b in lcc:
            agree[(a, b)] += lf.get(a) == lf.get(b)
    keep = [(a, b) for a, b in lcc if agree[(a, b)] / runs >= 0.5]
    return leiden(keep, n, weights={e: agree[e] / runs for e in keep})


def kcore(edges, n, cap=10):
    lcc, _ = gh.lcc_edges(edges, np.arange(n))
    nodes = sorted({x for e in lcc for x in e})
    idx = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in lcc])
    core = g.coreness()
    out, lvl = {}, 0
    for k in range(1, max(core) + 1):
        vs = [v for v in range(g.vcount()) if core[v] >= k]
        sub = g.induced_subgraph(vs)
        for ci, comp in enumerate(sub.connected_components()):
            members = [nodes[vs[i]] for i in comp]
            if len(members) <= cap:
                out[(lvl, ("k", k, ci))] = frozenset(members)
                continue
            sg = sub.induced_subgraph(comp)
            sn = [nodes[vs[comp[i]]] for i in range(sg.vcount())]
            order = sg.bfs(min(range(sg.vcount()), key=lambda i: sn[i]))[0]
            for s in range(0, len(order), cap):
                out[(lvl, ("k", k, ci, s))] = frozenset(sn[i] for i in order[s:s + cap] if i >= 0)
        lvl += 1
    return out


def quality(comms, edges, n):
    g = ig.Graph(n=n, edges=edges)
    lf, top = leaves(comms), {}
    for (l, c), m in comms.items():
        if l == 0:
            for v in m:
                top[v] = c
    out = {}
    for name, part in (("leaf", {v: str(x) for v, x in lf.items()}), ("top", {v: str(x) for v, x in top.items()})):
        vs = sorted(part)
        if not vs:
            out[name] = float("nan")
            continue
        sub = g.induced_subgraph(vs)
        _, lab = np.unique([part[v] for v in vs], return_inverse=True)
        out[name] = sub.modularity(lab.tolist())
    from collections import Counter
    lv = Counter(k[0] for k in comms)
    tops = [m for k, m in comms.items() if k[0] == 0]
    out["levels"], out["n_top"] = len(lv), len(tops)
    out["largest_top"] = max((len(m) for m in tops), default=0)
    covered = set().union(*comms.values()) if comms else set()
    out["coverage"] = len(covered) / max(len({x for e in edges for x in e}), 1)
    out["reports"] = len(comms)
    return out


def main():
    res = {}
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
        drops = []
        for _ in range(3):
            keep = rng.random(len(el)) >= 0.001
            drops.append([e for e, k in zip(el, keep) if k])
        variants = {"leiden": lambda e: leiden(e, n), "consensus": lambda e: consensus(e, n), "kcore": lambda e: kcore(e, n)}
        out = {}
        for name, fn in variants.items():
            ref = fn(el)
            q = quality(ref, el, n)
            q["drop"] = [nh.rewrite(ref, fn(x), 0.9) for x in drops]
            q["last"] = nh.rewrite(fn(before), ref, 0.9)
            out[name] = q
            print(graph, name, {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in q.items()}, flush=True)
        # stability-aware: warm-start from the reference clustering's leaves (deletion) / from the pre-update clustering
        ref = out_ref = leiden(el, n)

        def warm_from(prev, edges):
            lf = leaves(prev)
            ids, start = {}, {}
            for v, lc in lf.items():
                start[str(v)] = ids.setdefault(lc, len(ids))
            lcc, _ = gh.lcc_edges(edges, np.arange(n))
            nxt = len(ids)
            for a, b in lcc:
                for v in (a, b):
                    if str(v) not in start:
                        start[str(v)] = nxt
                        nxt += 1
            return leiden(edges, n, start=start)
        q = quality(out_ref, el, n)
        q["drop"] = [nh.rewrite(ref, warm_from(ref, x), 0.9) for x in drops]
        prev = leiden(before, n)
        q["last"] = nh.rewrite(prev, warm_from(prev, el), 0.9)
        out["warm"] = q
        print(graph, "warm", {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in q.items()}, flush=True)
        res[graph] = out
        (HERE / "stability_variants.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
