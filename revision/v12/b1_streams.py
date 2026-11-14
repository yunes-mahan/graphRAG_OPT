"""B1 (V12) small-batch streams for the batch-size sweep.

entity graph (documents arrive; an entity arrives with its first document, a relationship with the first document that
states it, as in entity_graph_eval.stream):
    version 0 = all documents up to 2018; then N updates of B documents each, drawn without replacement in a random
    order (rng seed) from the 2019-2020 documents.
arXiv (ogbn-arxiv; a paper arrives with its references, as in tasks/kgversion2/evaluate.schedule):
    version 0 = all papers up to 2019; then N updates of B papers each, drawn in a random order from the 2020 papers.
Nodes/edges that would arrive after the last update are left out (node_t = edge_t = -1 / large), as in the main streams.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
TASK = Path(os.environ.get("GCM_TASK", A.parent / "tasks" / "kgversion2"))
os.environ.setdefault("TASK_SOURCE_DIR", str(TASK))
sys.path[:0] = [str(TASK), str(A)]
import entity_graph_eval as eg  # noqa: E402

FAR = 10 ** 6


def entity_stream(B, N, seed=0, base_year=2018):
    g = json.loads((eg.G / "graph.json").read_text())
    n, E = len(g["entities"]), np.array(g["edges"], dtype=np.int64).T
    rng = np.random.default_rng(seed)
    doc_t = {d: 0 for d, v in g["docs"].items() if v["year"] <= base_year}
    pool = rng.permutation(sorted(d for d, v in g["docs"].items() if v["year"] > base_year))
    for k in range(N):
        for d in pool[k * B:(k + 1) * B]:
            doc_t[d] = k + 1
    node_t, edge_t = np.full(n, FAR, dtype=np.int64), np.full(E.shape[1], FAR, dtype=np.int64)
    batch_rels = [set() for _ in range(N + 1)]
    for d, t in doc_t.items():
        v = g["docs"][d]
        for e in v["ents"]:
            node_t[e] = min(node_t[e], t)
        for r in v["rels"]:
            edge_t[r] = min(edge_t[r], t)
        batch_rels[t].update(v["rels"])
    node_t[node_t == FAR] = -1
    edge_t[edge_t == FAR] = FAR                       # never arrives (filtered by edge_t <= t)
    d = dict(edges=E, node_year=np.zeros(n, dtype=np.int64), edge_year=np.zeros(E.shape[1], dtype=np.int64),
             feat=np.zeros((n, 1), np.float32), label=None, base=base_year, last=None, edge_mode="nodes")
    return d, node_t, edge_t, N + 1, batch_rels


def arxiv_stream(B, N, seed=0, base_year=2019):
    import evaluate as ev
    d = ev.load("arxiv")
    year = d["node_year"]
    rng = np.random.default_rng(seed)
    node_t = np.full(len(year), -1, dtype=np.int64)
    node_t[year <= base_year] = 0
    pool = rng.permutation(np.where(year == base_year + 1)[0])
    for k in range(N):
        node_t[pool[k * B:(k + 1) * B]] = k + 1
    edges = d["edges"]
    edge_t = np.where((node_t[edges[0]] >= 0) & (node_t[edges[1]] >= 0),
                      np.maximum(node_t[edges[0]], node_t[edges[1]]), FAR)
    return d, node_t, edge_t, N + 1
