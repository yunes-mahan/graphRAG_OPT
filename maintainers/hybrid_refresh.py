"""Hybrid maintainer family: a per-update base step + an ID-preserving warm-start refresh.

Base step (HYBRID_BASE):
  local  - core-anchored local repair (local_repair_winner.py)                 [default]
  place  - only place new nodes (modularity-gain placement), no local moves      [ablation]
Refresh trigger (HYBRID_TRIGGER):
  drift    - refresh when modularity fell by more than HYBRID_DELTA since the last refresh [default]
  periodic - refresh every HYBRID_K updates                                       [baseline]
  never    - no refresh
A refresh runs one warm-started Leiden pass seeded with the current assignment; refreshed
communities are mapped one-to-one to existing identifiers by majority overlap.
Causal: only current and past inputs are used.
"""

import os
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from local_repair_winner import AnchoredBudgetedRepair  # noqa: E402


def map_to_existing_ids(node_list, new_membership, label, next_id):
    """One-to-one majority-overlap mapping of new communities to existing IDs."""
    votes = {}
    for v, c in zip(node_list, new_membership):
        if v in label:
            votes.setdefault(int(c), []).append(label[v])
    mapping, used = {}, set()
    for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        vals, counts = np.unique(ids, return_counts=True)
        for cand in vals[np.argsort(-counts, kind="stable")]:
            if int(cand) not in used:
                mapping[c] = int(cand)
                used.add(int(cand))
                break
    for c in sorted(set(int(x) for x in new_membership)):
        if c not in mapping:
            mapping[c] = next_id
            next_id += 1
    return mapping, next_id


class HybridRefresh(AnchoredBudgetedRepair):
    def __init__(self, seed=0, delta=0.05, base="local", trigger="drift", k=5):
        super().__init__(seed=seed)
        self.delta, self.base, self.trigger, self.k = delta, base, trigger, k
        self.edges = []
        self.q_ref = None
        self.refreshes = 0
        self.t = 0

    def _graph(self, node_list):
        idx = {v: i for i, v in enumerate(node_list)}
        e = np.concatenate(self.edges, 1) if self.edges else np.zeros((2, 0), dtype=np.int64)
        return ig.Graph(n=len(node_list), edges=[(idx[int(a)], idx[int(b)]) for a, b in e.T], directed=False)

    def initialize(self, nodes, edges, features, years):
        self.edges.append(np.asarray(edges))
        out = super().initialize(nodes, edges, features, years)
        g = self._graph([int(v) for v in nodes])
        _, compact = np.unique(out, return_inverse=True)
        self.q_ref = g.modularity(compact.tolist())
        return out

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self.t += 1
        self.edges.append(np.asarray(new_edges))
        node_list = [int(v) for v in all_nodes.tolist()]
        if self.base == "local":
            out = super().update(new_nodes, new_edges, new_features, new_years, all_nodes)
        else:  # placement only
            self._add_edges(new_edges)
            self._place_new_nodes([int(v) for v in new_nodes.tolist()], node_list)
            out = np.asarray([self.label[v] for v in node_list], dtype=np.int64)
        if self.trigger == "never":
            return out
        g = self._graph(node_list)
        _, compact = np.unique(out, return_inverse=True)
        if self.trigger == "drift":
            fire = g.modularity(compact.tolist()) < self.q_ref - self.delta
        else:
            fire = self.t % self.k == 0
        if fire:
            part = la.find_partition(g, la.ModularityVertexPartition, seed=self.seed,
                                     initial_membership=compact.tolist())
            mapping, self.next_id = map_to_existing_ids(node_list, part.membership, self.label, self.next_id)
            for v, c in zip(node_list, part.membership):
                self.label[v] = mapping[int(c)]
            out = np.asarray([self.label[v] for v in node_list], dtype=np.int64)
            self.q_ref = part.modularity
            self.refreshes += 1
        return out


def make_maintainer():
    return HybridRefresh(seed=0, delta=float(os.environ.get("HYBRID_DELTA", "0.05")),
                         base=os.environ.get("HYBRID_BASE", "local"),
                         trigger=os.environ.get("HYBRID_TRIGGER", "drift"),
                         k=int(os.environ.get("HYBRID_K", "5")))
