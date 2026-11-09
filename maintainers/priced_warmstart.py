"""Summary-priced rollback on top of warm-started Leiden (idea I4).

Each update: (1) warm-started Leiden proposes a partition and maps it to existing IDs (as warmstart_leiden.py);
(2) for every previous community k with >= 5 members whose summary the proposal would invalidate
(Jaccard(old members, new members) < 0.9), try to roll back the moves of EXISTING nodes into and out of k
(new nodes keep their proposed community); (3) keep the rollback if it makes k's summary reusable and the
modularity it costs is smaller than the summary's price:
      Q(proposal) - Q(rollback)  <  PRICE * tokens(k) / tokens(all summaries of the proposal)
with tokens(S) = 1500 + min(8000, 250 |S|) and PRICE = lambda * 16/15 (lambda = 0.31, the token-level weight;
16 versions in the quality mean, 15 updates in the cost mean). Communities are tried in order of decreasing price.
Online and causal: uses only the current and previous versions.
"""

import os

import igraph as ig
import leidenalg as la
import numpy as np

LAMBDA = float(os.environ.get("PRICE_LAMBDA", "0.31"))
PRICE = LAMBDA * 16 / 15
MIN_SIZE, JAC = 5, 0.9


def tokens(size):
    return 1500 + min(8000, 250 * size)


class PricedWarm:
    def __init__(self):
        self.E = []
        self.label = {}
        self.next = 0

    def _graph(self, nodes):
        e = np.concatenate(self.E, 1)
        pos = np.searchsorted(nodes, e)
        return pos, ig.Graph(n=len(nodes), edges=np.stack([pos[0], pos[1]], 1).tolist())

    def initialize(self, nodes, edges, features, years):
        self.E.append(edges)
        nodes = np.sort(nodes)
        _, g = self._graph(nodes)
        m = np.array(la.find_partition(g, la.ModularityVertexPartition, seed=0).membership)
        self.label = {int(v): int(c) for v, c in zip(nodes, m)}
        self.next = int(m.max()) + 1
        return np.array([self.label[int(v)] for v in nodes])

    @staticmethod
    def _Q(y, u, v, deg, two_m):
        same = y[u] == y[v]
        k = int(y.max()) + 1
        e_c = np.bincount(y[u][same], minlength=k)
        d_c = np.bincount(y, weights=deg, minlength=k)
        return float(e_c.sum() / (two_m / 2) - ((d_c / two_m) ** 2).sum())

    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        self.E.append(new_edges)
        nodes = np.asarray(all_nodes)
        pos, g = self._graph(nodes)
        init, nxt = [], self.next
        for v in nodes.tolist():
            if v in self.label:
                init.append(self.label[v])
            else:
                init.append(nxt)
                nxt += 1
        _, ic = np.unique(init, return_inverse=True)
        m = np.array(la.find_partition(g, la.ModularityVertexPartition, seed=0, initial_membership=ic.tolist()).membership)
        votes = {}
        for v, c in zip(nodes.tolist(), m.tolist()):
            if v in self.label:
                votes.setdefault(c, []).append(self.label[v])
        mp, used = {}, set()
        for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
            b = max(set(ids), key=ids.count)
            if b not in used:
                mp[c] = b
                used.add(b)
        for c in sorted(set(m.tolist())):
            if c not in mp:
                mp[c] = self.next
                self.next += 1
        prop = np.array([mp[c] for c in m.tolist()])
        # --- summary-priced rollback -------------------------------------------------------------------------
        prev = np.array([self.label.get(v, -1) for v in nodes.tolist()])      # -1 for new nodes
        ids, y = np.unique(prop, return_inverse=True)                           # compact labels for Q
        all_ids = np.unique(np.concatenate([prop, prev[prev >= 0]]))
        y = np.searchsorted(all_ids, prop)
        u, w = pos
        deg = np.bincount(u, minlength=len(nodes)) + np.bincount(w, minlength=len(nodes))
        two_m = 2.0 * u.shape[0]
        q_cur = self._Q(y, u, w, deg, two_m)
        sizes = np.bincount(y)
        total_tok = sum(tokens(int(s)) for s in sizes if s >= MIN_SIZE)
        old_ids, old_sizes = np.unique(prev[prev >= 0], return_counts=True)
        cand = []
        for k, s_old in zip(old_ids.tolist(), old_sizes.tolist()):
            if s_old < MIN_SIZE:
                continue
            old_mem = prev == k
            new_mem = prop == k
            inter = int((old_mem & new_mem).sum())
            if inter / max(int((old_mem | new_mem).sum()), 1) < JAC:
                cand.append((tokens(int(new_mem.sum() or s_old)), k))
        for price_tok, k in sorted(cand, reverse=True):
            old_mem = prev == k
            cur_k = (prop == k)
            back_in = old_mem & ~cur_k                     # old members that left k
            back_out = cur_k & (prev >= 0) & (prev != k)   # existing nodes that moved into k
            if not back_in.any() and not back_out.any():
                continue
            trial = prop.copy()
            trial[back_in] = k
            trial[back_out] = prev[back_out]
            new_k = trial == k
            if int((old_mem & new_k).sum()) / max(int((old_mem | new_k).sum()), 1) < JAC:
                continue                                    # growth by new nodes alone already invalidates it
            yt = np.searchsorted(all_ids, trial)            # vectorised relabelling (labels are in all_ids)
            q_new = self._Q(yt, u, w, deg, two_m)
            if q_cur - q_new < PRICE * price_tok / max(total_tok, 1):
                prop, q_cur = trial, q_new
        self.label = {v: int(c) for v, c in zip(nodes.tolist(), prop.tolist())}
        return prop


def make_maintainer():
    return PricedWarm()
