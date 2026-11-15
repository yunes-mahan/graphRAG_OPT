"""V15/D4: why GraphRAG's update covers 90 / 16 / 41 % of the new entities at batches of 1 / 10 / 100 documents (entity
graph, b1_streams.entity_stream, seeds 0-2, 10 updates): GraphRAG clusters only the largest connected component of the
batch's delta graph (all relationships stated in the batch's documents, as graphrag_hier.py).  Per update: number of
connected components of the delta graph, size of its largest one, share of the update's new entities inside it.
Output: analysis/v12/v15_delta_lcc.json"""
import json
import sys
from pathlib import Path

import igraph as ig
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import b1_streams as bs  # noqa: E402

out = {}
for B in (1, 10, 100):
    comp, lccn, share = [], [], []
    for seed in (0, 1, 2):
        d, node_t, edge_t, T, batch_rels = bs.entity_stream(B, 10, seed)
        E = d["edges"]
        for t in range(1, T):
            el = sorted({(int(E[0, j]), int(E[1, j])) for j in batch_rels[t]})
            vs = sorted({v for e in el for v in e})
            idx = {v: i for i, v in enumerate(vs)}
            g = ig.Graph(n=len(vs), edges=[(idx[a], idx[b]) for a, b in el])
            cc = g.connected_components()
            big = {vs[i] for i in max(cc, key=len)} if len(vs) else set()
            new = set(np.where(node_t == t)[0].tolist())
            comp.append(len(cc))
            lccn.append(len(big))
            share.append(len(new & big) / len(new) if new else float("nan"))
    out[str(B)] = dict(components=float(np.mean(comp)), lcc_size=float(np.mean(lccn)), new_in_lcc=float(np.nanmean(share)))
    print(B, out[str(B)])
(HERE / "v15_delta_lcc.json").write_text(json.dumps(out, indent=1))
