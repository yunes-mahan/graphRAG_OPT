"""Dynamic-frontier Leiden baseline (in the spirit of DF-Leiden, Sahu 2024).

On every update the previous assignment is kept for all unaffected nodes, which are held FIXED;
only the frontier - new nodes, endpoints of new edges and their direct neighbours - is
re-optimised by Leiden (leidenalg Optimiser with is_membership_fixed). Communities are then mapped
one-to-one to existing identifiers by majority overlap. Causal.
"""

import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hybrid_refresh import map_to_existing_ids  # noqa: E402


class FrontierLeiden:
    def __init__(self, seed=0):
        self.seed = seed
        self.edges = []
        self.adj = {}
        self.label = {}
        self.next_id = 0

    def _add(self, edges):
        self.edges.append(np.asarray(edges))
        for a, b in np.asarray(edges).T.tolist():
            self.adj.setdefault(int(a), set()).add(int(b))
            self.adj.setdefault(int(b), set()).add(int(a))

    def _graph(self, node_list):
        idx = {v: i for i, v in enumerate(node_list)}
        e = np.concatenate(self.edges, 1)
        return ig.Graph(n=len(node_list), edges=[(idx[int(a)], idx[int(b)]) for a, b in e.T], directed=False)

    def initialize(self, nodes, edges, features, years):
        self._add(edges)
        node_list = [int(v) for v in nodes.tolist()]
        part = la.find_partition(self._graph(node_list), la.ModularityVertexPartition, seed=self.seed)
        for v, c in zip(node_list, part.membership):
            self.label[v] = int(c)
        self.next_id = max(part.membership) + 1
        return np.asarray([self.label[v] for v in node_list], dtype=np.int64)

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self._add(new_edges)
        node_list = [int(v) for v in all_nodes.tolist()]
        frontier = {int(v) for v in new_nodes.tolist()}
        for a, b in np.asarray(new_edges).T.tolist():
            frontier.update((int(a), int(b)))
        frontier |= {u for v in list(frontier) for u in self.adj.get(v, ())}
        init, nxt = [], self.next_id
        for v in node_list:
            if v in self.label:
                init.append(self.label[v])
            else:
                init.append(nxt)
                nxt += 1
        _, compact = np.unique(init, return_inverse=True)
        g = self._graph(node_list)
        part = la.ModularityVertexPartition(g, initial_membership=compact.tolist())
        opt = la.Optimiser()
        opt.set_rng_seed(self.seed)
        opt.optimise_partition(part, n_iterations=2, is_membership_fixed=[v not in frontier for v in node_list])
        mapping, self.next_id = map_to_existing_ids(node_list, part.membership, self.label, self.next_id)
        for v, c in zip(node_list, part.membership):
            self.label[v] = mapping[int(c)]
        return np.asarray([self.label[v] for v in node_list], dtype=np.int64)


def make_maintainer():
    return FrontierLeiden(seed=0)
