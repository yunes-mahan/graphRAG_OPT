"""Microsoft GraphRAG incremental update ('graphrag update'), emulated as a maintainer.

Faithful to the official update workflow (microsoft/graphrag, index/update/communities.py and the
create_communities / update_communities workflows): Leiden is run on the DELTA only, i.e. on the relationships
added by the update; the resulting communities receive fresh identifiers (offset by the previous maximum id) and
are appended; existing communities and their reports are carried over unchanged. Existing nodes therefore keep
their community (in GraphRAG an existing entity that re-appears in the delta is additionally listed in a new
delta community; a single-label partition keeps its old community). New nodes without delta edges become
singleton communities. Online and causal.
"""

import igraph as ig
import leidenalg as la
import numpy as np


class GraphRAGDelta:
    def __init__(self, seed=0):
        self.seed = seed
        self.label = {}
        self.next_id = 0

    def initialize(self, nodes, edges, features, years):
        node_list = [int(v) for v in np.asarray(nodes).tolist()]
        idx = {v: i for i, v in enumerate(node_list)}
        g = ig.Graph(n=len(node_list), edges=[(idx[int(a)], idx[int(b)]) for a, b in np.asarray(edges).T])
        memb = la.find_partition(g, la.ModularityVertexPartition, seed=self.seed).membership
        for v, c in zip(node_list, memb):
            self.label[v] = int(c)
        self.next_id = max(memb) + 1
        return np.asarray([self.label[v] for v in node_list], dtype=np.int64)

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        e = np.asarray(new_edges)
        delta_nodes = sorted({int(v) for v in np.asarray(new_nodes).tolist()} | set(e.ravel().tolist()))
        if delta_nodes:
            idx = {v: i for i, v in enumerate(delta_nodes)}
            g = ig.Graph(n=len(delta_nodes), edges=[(idx[int(a)], idx[int(b)]) for a, b in e.T])
            memb = la.find_partition(g, la.ModularityVertexPartition, seed=self.seed).membership
            for v, c in zip(delta_nodes, memb):
                if v not in self.label:            # existing nodes keep their community (old rows untouched)
                    self.label[v] = self.next_id + int(c)
            self.next_id += max(memb) + 1
        return np.asarray([self.label[int(v)] for v in np.asarray(all_nodes).tolist()], dtype=np.int64)


def make_maintainer():
    return GraphRAGDelta(seed=0)
