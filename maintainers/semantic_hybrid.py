"""Hybrid maintainer with SEMANTIC stress (idea A).

Identical to hybrid_refresh.py (local repair + drift-triggered refresh, delta = 0.15) except for the
stress score that decides WHICH touched nodes are considered for a move:
    stress = structural stress (share of edges leaving the community, as before)
           + semantic stress   (1 - cosine(node embedding, centroid of its community))
with weight 1:1 fixed a priori (no tuning). Moves are still accepted only on an exact modularity
gain, so embeddings change the candidate set, not the objective. Node embeddings are the features
passed by the evaluator (128-d text embeddings in ogbn-arxiv / ogbl-collab). Causal.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_refresh import HybridRefresh  # noqa: E402

SEM_WEIGHT = float(os.environ.get("SEM_WEIGHT", "1.0"))
SEM_CENTER = os.environ.get("SEM_CENTER", "0") == "1"  # subtract running mean of all seen embeddings


class SemanticHybrid(HybridRefresh):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.emb = {}
        self.raw = {}
        self.centroid = {}
        self.sum = None
        self.count = 0

    def _store(self, nodes, feats):
        f = np.asarray(feats, dtype=np.float32)
        if SEM_CENTER:
            self.sum = f.sum(0) if self.sum is None else self.sum + f.sum(0)
            self.count += len(f)
            for v, x in zip(np.asarray(nodes).tolist(), f):
                self.raw[int(v)] = x
            mu = self.sum / max(self.count, 1)   # causal: mean over nodes seen so far
            for v, x in self.raw.items():
                y = x - mu
                self.emb[v] = y / max(np.linalg.norm(y), 1e-9)
            return
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
        for v, x in zip(np.asarray(nodes).tolist(), f):
            self.emb[int(v)] = x

    def _recompute_centroids(self, node_list):
        sums, cnt = {}, {}
        for v in node_list:
            c = self.label[v]
            x = self.emb.get(v)
            if x is None:
                continue
            if c in sums:
                sums[c] += x
                cnt[c] += 1
            else:
                sums[c] = x.copy()
                cnt[c] = 1
        self.centroid = {c: s / max(np.linalg.norm(s), 1e-9) for c, s in sums.items()}

    def _node_stress(self, v):
        stress, support = super()._node_stress(v)
        x, cen = self.emb.get(v), self.centroid.get(self.label.get(v))
        if x is not None and cen is not None:
            stress += SEM_WEIGHT * (1.0 - float(np.dot(x, cen)))
        return stress, support

    def initialize(self, nodes, edges, features, years):
        self._store(nodes, features)
        return super().initialize(nodes, edges, features, years)

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self._store(new_nodes, new_features)
        # centroids from the current assignment (before this update's moves); new nodes are added by
        # placement inside super().update, so recompute lazily right before stress ranking
        orig_rank = self._ranked_frontier

        def ranked(active, old_nodes, node_count):
            self._recompute_centroids([int(v) for v in all_nodes.tolist()])
            return orig_rank(active, old_nodes, node_count)

        self._ranked_frontier = ranked
        try:
            return super().update(new_nodes, new_edges, new_features, new_years, all_nodes)
        finally:
            self._ranked_frontier = orig_rank


def make_maintainer():
    return SemanticHybrid(seed=0, delta=float(os.environ.get("HYBRID_DELTA", "0.15")),
                          base=os.environ.get("HYBRID_BASE", "local"),
                          trigger=os.environ.get("HYBRID_TRIGGER", "drift"),
                          k=int(os.environ.get("HYBRID_K", "5")))
