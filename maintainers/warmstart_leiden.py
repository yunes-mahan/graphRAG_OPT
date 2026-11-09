import igraph as ig, leidenalg as la, numpy as np
class WS:
    def __init__(self): self.E = []; self.label = {}; self.next = 0
    def _run(self, nodes, init=None):
        idx = {int(v): i for i, v in enumerate(nodes)}
        e = np.concatenate(self.E, 1)
        g = ig.Graph(n=len(nodes), edges=np.stack([np.vectorize(idx.get)(e[0]), np.vectorize(idx.get)(e[1])], 1).tolist())
        return np.array(la.find_partition(g, la.ModularityVertexPartition, seed=0, initial_membership=init).membership)
    def initialize(self, nodes, edges, features, years):
        self.E.append(edges); m = self._run(nodes)
        self.label = {int(v): int(c) for v, c in zip(nodes, m)}; self.next = m.max() + 1
        return np.array([self.label[int(v)] for v in nodes])
    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self.E.append(new_edges)
        init = []; nxt = self.next
        for v in all_nodes:
            v = int(v)
            if v in self.label: init.append(self.label[v])
            else: init.append(nxt); nxt += 1
        _, ic = np.unique(init, return_inverse=True)
        m = self._run(all_nodes, ic.tolist())
        votes = {}
        for v, c in zip(all_nodes, m):
            if int(v) in self.label: votes.setdefault(int(c), []).append(self.label[int(v)])
        mp, used = {}, set()
        for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
            b = max(set(ids), key=ids.count)
            if b not in used: mp[c] = b; used.add(b)
        for c in set(m.tolist()):
            if c not in mp: mp[c] = self.next; self.next += 1
        self.label = {int(v): mp[int(c)] for v, c in zip(all_nodes, m)}
        return np.array([self.label[int(v)] for v in all_nodes])
def make_maintainer(): return WS()
