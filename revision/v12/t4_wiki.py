"""T4 (V13): every flat maintainer of Table 7 on the Wikipedia stream (yearly snapshots 2017-2025; pages and links
can also disappear).

The maintainers in programs/ (and the lazy baseline) only accept insertions. Snapshot adapter, applied before every
update: pages and links that disappeared are removed from the maintainer's stored graph (edge lists / adjacency) and
its label table; then the maintainer receives the new pages and new links as an ordinary insertion update. Removed
links therefore do NOT put their endpoints into a frontier (a limitation stated in the paper).
Same evaluation as wiki/wiki_priced.py (graph = links between pages of the snapshot; token cost = reports of
communities >= 5 that are new or have Jaccard < 0.9 to the same identifier, relative to reporting every community of
a from-scratch Leiden run with the same seed; lambda 0.31; seeds 0-4 of Leiden's RNG as replicates). The seed is set
for every leidenalg call of a maintainer (some programs fix seed=0 internally), so all maintainers use the same five
seeds. Check: recompute and warm start must reproduce wiki/priced_results.json.
Output: analysis/v12/t4_wiki.json
"""
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import leidenalg as la
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wiki"), str(A / "programs")]
import run_suite as rs  # noqa: E402
import wiki_priced as wp  # noqa: E402

SEED = [0]
_fp = la.find_partition
la.find_partition = lambda *a, **k: _fp(*a, **{**k, "seed": SEED[0]})
_srs = la.Optimiser.set_rng_seed
la.Optimiser.set_rng_seed = lambda self, s: _srs(self, SEED[0])

METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid", "lazy", "graphrag_delta",
           "priced_warm", "priced_frontier", "local_repair"]


def load(method):
    path, env = rs.SPECS[method]
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(f"m_{method}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_maintainer()


def reset(m, kept, keep_nodes):
    """Remove deleted pages/links from the maintainer's stored graph."""
    if hasattr(m, "E"):
        m.E = [kept]
    if hasattr(m, "edges") and isinstance(m.edges, list):
        m.edges = [kept]
    if hasattr(m, "adj"):
        if isinstance(m.adj, defaultdict):
            adj = defaultdict(list)
            for a, b in kept.T.tolist():
                adj[a].append(b)
                adj[b].append(a)
        else:
            adj = {}
            for a, b in kept.T.tolist():
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        m.adj = adj
    if hasattr(m, "edge_count"):
        m.edge_count = kept.shape[1]
    m.label = {v: c for v, c in m.label.items() if v in keep_nodes}


def snap_edges(snap, nodes):
    pos = set(nodes)
    return {(min(int(a), b), max(int(a), b)) for a, s in snap.items() for b in s["links"] if b in pos and b != int(a)}


def arr(es):
    return np.array(sorted(es), dtype=np.int64).T if es else np.zeros((2, 0), np.int64)


def run(stream, method, seed):
    SEED[0] = seed
    m = load(method)
    years = [str(y) for y in stream["years"]]
    prev, pe, pn, Q, C = {}, set(), set(), [], []
    for t, y in enumerate(years):
        snap = stream["snapshots"][y]
        nodes = sorted(int(k) for k in snap)
        es = snap_edges(snap, nodes)
        g, _ = wp.graph(snap, nodes)
        ref = _fp(g, la.ModularityVertexPartition, seed=seed).membership
        nn = np.array(nodes, dtype=np.int64)
        f0 = np.zeros((len(nodes), 1), np.float32)
        if t == 0:
            lab = np.asarray(m.initialize(nn, arr(es), f0, np.zeros(len(nodes), np.int64)))
        else:
            reset(m, arr(es & pe), set(nodes) & pn)
            new = np.array(sorted(set(nodes) - pn), dtype=np.int64)
            lab = np.asarray(m.update(new, arr(es - pe), np.zeros((len(new), 1), np.float32),
                                      np.zeros(len(new), np.int64), nn))
        Q.append(g.modularity(np.unique(lab, return_inverse=True)[1].tolist()))
        if t > 0:
            C.append(wp.regen_tokens(prev, lab, nodes) / max(wp.summ_tokens(np.array(ref)), 1))
        prev = {v: int(c) for v, c in zip(nodes, lab.tolist())}
        pe, pn = es, set(nodes)
    return float(np.mean(Q)), float(np.mean(C))


def main():
    stream = json.loads((A / "wiki" / "stream.json").read_text())
    out_f = HERE / "t4_wiki.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for method in METHODS:
        if method in res:
            continue
        runs = [run(stream, method, s) for s in range(5)]
        sc = [q - wp.LAM * c for q, c in runs]
        res[method] = dict(Q=float(np.mean([r[0] for r in runs])), C=float(np.mean([r[1] for r in runs])),
                           score=float(np.mean(sc)), score_sd=float(np.std(sc)), runs=runs)
        print(f"{method:18s} Q={res[method]['Q']:.4f} C={res[method]['C']:.4f} score={res[method]['score']:.4f}", flush=True)
        out_f.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
