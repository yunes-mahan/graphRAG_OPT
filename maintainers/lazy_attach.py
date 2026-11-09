"""Baseline community maintainer: Leiden once on V0, then lazy incremental updates.

Interface (called by evaluate.py, one object for the whole version stream):
    make_maintainer() -> object with
      initialize(nodes, edges, features, years) -> labels for `nodes`
      update(new_nodes, new_edges, new_features, new_years, all_nodes) -> labels for `all_nodes`
  nodes / new_nodes / all_nodes : int64 arrays of global node IDs (all_nodes is sorted)
  edges / new_edges             : int64 array of shape (2, E) with global node IDs (undirected,
                                  each edge given once, only the edges ADDED in this version)
  features / new_features       : float32 (n, 128) text embeddings of the papers
  years / new_years             : int64 publication years
Return: int array with one persistent community ID per node, in the order of the node array.
Keeping an ID for a community means its GraphRAG summary can be reused.
"""

from collections import Counter, defaultdict

import igraph as ig
import leidenalg as la
import numpy as np


class LazyIncremental:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.adj = defaultdict(list)
        self.label = {}
        self.next_id = 0

    def _add_edges(self, edges):
        for a, b in edges.T.tolist():
            self.adj[a].append(b)
            self.adj[b].append(a)

    def initialize(self, nodes, edges, features, years):
        self._add_edges(edges)
        idx = {int(v): i for i, v in enumerate(nodes)}
        g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in edges.T.tolist()])
        part = la.find_partition(g, la.ModularityVertexPartition, seed=self.seed)
        for v, c in zip(nodes.tolist(), part.membership):
            self.label[v] = c
        self.next_id = max(part.membership) + 1
        return np.array([self.label[int(v)] for v in nodes], dtype=np.int64)

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self._add_edges(new_edges)
        pending = [int(v) for v in new_nodes]
        for _ in range(3):  # a few propagation rounds so chains of new nodes get labelled
            for v in pending:
                if v in self.label:
                    continue
                nb = [self.label[u] for u in self.adj[v] if u in self.label]
                if nb:
                    self.label[v] = Counter(nb).most_common(1)[0][0]
        for v in pending:
            if v not in self.label:
                self.label[v] = self.next_id
                self.next_id += 1
        return np.array([self.label[int(v)] for v in all_nodes], dtype=np.int64)


def make_maintainer():
    return LazyIncremental(seed=0)
