"""Summary-priced rollback on top of frontier-restricted Leiden (idea I4b).

Identical rollback rule to priced_warmstart.py, applied to the proposal of frontier_leiden.py:
for every previous community (>= 5 members) whose summary the proposal invalidates, roll back the moves of
existing nodes into/out of it if that makes the summary reusable and costs less modularity than
PRICE * tokens(k) / tokens(all summaries), PRICE = 0.31 * 16/15. Online and causal.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from frontier_leiden import FrontierLeiden  # noqa: E402

LAMBDA = float(os.environ.get("PRICE_LAMBDA", "0.31"))
PRICE = LAMBDA * 16 / 15
MIN_SIZE, JAC = 5, 0.9


def tokens(size):
    return 1500 + min(8000, 250 * size)


def _Q(y, u, w, deg, two_m):
    same = y[u] == y[w]
    k = int(y.max()) + 1
    e_c = np.bincount(y[u][same], minlength=k)
    d_c = np.bincount(y, weights=deg, minlength=k)
    return float(e_c.sum() / (two_m / 2) - ((d_c / two_m) ** 2).sum())


def rollback(nodes, prop, prev, edges):
    """nodes sorted int array; prop/prev label arrays aligned with nodes (prev = -1 for new nodes)."""
    pos = np.searchsorted(nodes, edges)
    u, w = pos
    deg = np.bincount(u, minlength=len(nodes)) + np.bincount(w, minlength=len(nodes))
    two_m = 2.0 * u.shape[0]
    ids = np.unique(np.concatenate([prop, prev[prev >= 0]]))
    to_y = lambda x: np.searchsorted(ids, x)  # noqa: E731  (labels are always in ids; vectorised)
    q_cur = _Q(to_y(prop), u, w, deg, two_m)
    sizes = np.bincount(to_y(prop))
    total_tok = sum(tokens(int(s)) for s in sizes if s >= MIN_SIZE)
    old_ids, old_sizes = np.unique(prev[prev >= 0], return_counts=True)
    cand = []
    for k, s_old in zip(old_ids.tolist(), old_sizes.tolist()):
        if s_old < MIN_SIZE:
            continue
        old_mem, new_mem = prev == k, prop == k
        if int((old_mem & new_mem).sum()) / max(int((old_mem | new_mem).sum()), 1) < JAC:
            cand.append((tokens(int(new_mem.sum() or s_old)), k))
    for price_tok, k in sorted(cand, reverse=True):
        old_mem, cur_k = prev == k, prop == k
        back_in = old_mem & ~cur_k
        back_out = cur_k & (prev >= 0) & (prev != k)
        if not back_in.any() and not back_out.any():
            continue
        trial = prop.copy()
        trial[back_in] = k
        trial[back_out] = prev[back_out]
        new_k = trial == k
        if int((old_mem & new_k).sum()) / max(int((old_mem | new_k).sum()), 1) < JAC:
            continue
        q_new = _Q(to_y(trial), u, w, deg, two_m)
        if q_cur - q_new < PRICE * price_tok / max(total_tok, 1):
            prop, q_cur = trial, q_new
    return prop


class PricedFrontier(FrontierLeiden):
    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes):
        nodes = np.asarray(all_nodes)
        prev = np.array([self.label.get(int(v), -1) for v in nodes.tolist()])
        prop = super().update(new_nodes, new_edges, new_features, new_years, all_nodes)
        out = rollback(nodes, np.asarray(prop), prev, np.concatenate(self.edges, 1))
        for v, c in zip(nodes.tolist(), out.tolist()):
            self.label[int(v)] = int(c)
        return out


def make_maintainer():
    return PricedFrontier(seed=0)
