"""Reference: recompute Leiden from scratch every version; IDs matched greedily by max overlap."""
import igraph as ig, leidenalg as la, numpy as np
class Recompute:
    def __init__(self): self.E = []; self.label = {}; self.next = 0
    def _part(self, nodes):
        idx = {int(v): i for i, v in enumerate(nodes)}
        e = np.concatenate(self.E, 1)
        g = ig.Graph(n=len(nodes), edges=[(idx[int(a)], idx[int(b)]) for a, b in e.T])
        return np.array(la.find_partition(g, la.ModularityVertexPartition, seed=0).membership)
    def _assign(self, nodes, m):
        if not self.label:
            self.label = {int(v): int(c) for v, c in zip(nodes, m)}; self.next = int(m.max()) + 1
        else:
            votes = {}
            for v, c in zip(nodes, m):
                if int(v) in self.label: votes.setdefault(int(c), []).append(self.label[int(v)])
            mp, used = {}, set()
            for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
                b = max(set(ids), key=ids.count)
                if b not in used: mp[c] = b; used.add(b)
            for c in set(m.tolist()):
                if c not in mp: mp[c] = self.next; self.next += 1
            self.label = {int(v): mp[int(c)] for v, c in zip(nodes, m)}
        return np.array([self.label[int(v)] for v in nodes])
    def initialize(self, nodes, edges, features, years):
        self.E.append(edges); return self._assign(nodes, self._part(nodes))
    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self.E.append(new_edges); return self._assign(all_nodes, self._part(all_nodes))
def make_maintainer(): return Recompute()
