"""W8 (V16): synthetic deletion streams. arXiv-15, collab-12 and the flat LLM entity graph, orders 0-2: the usual
insertion schedule, and after every update 1 % of the present nodes (uniform, seeded) are deleted for good together
with their edges.  Every flat maintainer of Table 7 that runs through the snapshot adapter of v12/t4_wiki.py (deleted
nodes and edges are removed from the maintainer's stored graph before the update; the update then inserts the new
nodes and edges) is evaluated as on the Wikipedia stream: Q = modularity of the snapshot, C = report tokens rewritten
(new id or Jaccard < 0.9, communities >= 5) relative to reporting every community of a from-scratch Leiden run;
score = Q - 0.31 C.  Leiden seed 0 (the programs' default); the orders are the replicates.
DF Leiden (authors' code): our driver passes insertions only (n/a).
usage: python v16/w8_deletions.py GRAPH MAINTAINER   -> analysis/v16/w8/GRAPH_MAINTAINER.json"""
import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
TASK = A.parent / "tasks" / "kgversion2"
sys.path[:0] = [str(A), str(A / "wiki"), str(A / "programs"), str(A / "v12"), str(TASK)]
import os  # noqa: E402
os.environ.setdefault("TASK_SOURCE_DIR", str(TASK))
import t4_wiki as tw  # noqa: E402   (snapshot adapter; its seed override stays at seed 0)
import wiki_priced as wp  # noqa: E402

LAM, DEL = 0.31, 0.01
GRAPHS = {"arxiv15": ("arxiv", 2015, 2020), "collab12": ("collab", 2012, 2017), "entity": None}


def stream(graph, order):
    if GRAPHS[graph] is None:
        import entity_graph_eval as eg
        d, node_t, edge_t, T = eg.stream(order)
    else:
        import evaluate as ev
        name, base, last = GRAPHS[graph]
        d = ev.load(name)
        node_t, edge_t, T = ev.schedule(d, base, last, order)
    return d["edges"], np.asarray(node_t), np.asarray(edge_t), int(T)


def snapshots(graph, order):
    E, node_t, edge_t, T = stream(graph, order)
    rng = np.random.default_rng(1000 + order)
    deleted = np.zeros(len(node_t), bool)
    for t in range(T):
        if t > 0:
            pres = np.where((node_t >= 0) & (node_t < t) & ~deleted)[0]      # present before this update's insertions
            deleted[rng.choice(pres, int(round(DEL * len(pres))), replace=False)] = True
        alive = (node_t >= 0) & (node_t <= t) & ~deleted
        ok = (edge_t <= t) & alive[E[0]] & alive[E[1]] & (E[0] != E[1])
        es = {(int(min(a, b)), int(max(a, b))) for a, b in E[:, ok].T}
        yield sorted(np.where(alive)[0].tolist()), es


def run(graph, method, order):
    tw.SEED[0] = 0
    m = tw.load(method)
    prev, pe, pn, Q, C = {}, set(), set(), [], []
    for t, (nodes, es) in enumerate(snapshots(graph, order)):
        pos = {v: i for i, v in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(pos[a], pos[b]) for a, b in es])
        ref = tw._fp(g, la.ModularityVertexPartition, seed=0).membership
        nn = np.array(nodes, dtype=np.int64)
        if t == 0:
            lab = np.asarray(m.initialize(nn, tw.arr(es), np.zeros((len(nodes), 1), np.float32), np.zeros(len(nodes), np.int64)))
        else:
            tw.reset(m, tw.arr(es & pe), set(nodes) & pn)
            new = np.array(sorted(set(nodes) - pn), dtype=np.int64)
            lab = np.asarray(m.update(new, tw.arr(es - pe), np.zeros((len(new), 1), np.float32), np.zeros(len(new), np.int64), nn))
        Q.append(g.modularity(np.unique(lab, return_inverse=True)[1].tolist()))
        if t > 0:
            C.append(wp.regen_tokens(prev, lab, nodes) / max(wp.summ_tokens(np.array(ref)), 1))
        prev = {v: int(c) for v, c in zip(nodes, lab.tolist())}
        pe, pn = es, set(nodes)
    return float(np.mean(Q)), float(np.mean(C))


if __name__ == "__main__":
    graph, method = sys.argv[1], sys.argv[2]
    out_f = HERE / "w8" / f"{graph}_{method}.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for order in range(3):
        if str(order) in res:
            continue
        q, c = run(graph, method, order)
        res[str(order)] = dict(Q=q, C=c, score=q - LAM * c)
        out_f.write_text(json.dumps(res, indent=1))
        print(graph, method, order, res[str(order)], flush=True)
