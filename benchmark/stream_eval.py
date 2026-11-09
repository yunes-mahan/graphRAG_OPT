"""General version-stream replay + scoring for community maintainers (no LLM cost).

Datasets
  arxiv  : ogbn-arxiv citation graph; nodes arrive by publication year, an edge arrives with its
           later endpoint (identical to tasks/kgversion/evaluate.py).
  collab : ogbl-collab co-authorship graph; a node arrives with its first collaboration, an edge
           arrives in its own (first) collaboration year, so new edges also connect EXISTING nodes.
Protocol (both): V0 = everything up to BASE year, then every later year in 3 random batches
(schedule seed selects the batch order). Score = mean_t Q_t - lam * mean_t R_t, where Q_t is the
modularity of version t and R_t the fraction of nodes in communities whose summary must be
regenerated (new ID, or Jaccard(old members, new members) < 0.9).
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402



import importlib.util
import sys
import time
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd

ARXIV_NPZ = config.ARXIV_NPZ
COLLAB_NPZ = config.DATA / "collab_versions.npz"
JACCARD_KEEP = 0.9
TAUS = (0.8, 0.9, 0.95)
BATCHES = 3


def load(dataset: str):
    """Return dict with edges (2,E), edge_year (E,), node_year (N,), feat (N,d), label or None."""
    if dataset == "arxiv":
        z = np.load(ARXIV_NPZ)
        edges = z["edges"].astype(np.int64)
        year = z["year"].astype(np.int64)
        return dict(edges=edges, edge_year=np.maximum(year[edges[0]], year[edges[1]]),
                    node_year=year, feat=z["feat"], label=z["label"].astype(np.int64),
                    base=2015, last=2020, edge_mode="nodes")
    if dataset == "collab":
        if not COLLAB_NPZ.exists():
            raw = config.OGB_ROOT / "ogbl_collab" / "raw"
            e = pd.read_csv(raw / "edge.csv.gz", header=None).values.T
            y = pd.read_csv(raw / "edge_year.csv.gz", header=None).values.ravel()
            lo, hi = np.minimum(e[0], e[1]), np.maximum(e[0], e[1])
            df = pd.DataFrame({"a": lo, "b": hi, "y": y}).groupby(["a", "b"], as_index=False)["y"].min()
            feat = pd.read_csv(raw / "node-feat.csv.gz", header=None).values.astype(np.float16)
            n = feat.shape[0]
            node_year = np.full(n, 10**4, dtype=np.int64)
            np.minimum.at(node_year, df.a.values, df.y.values)
            np.minimum.at(node_year, df.b.values, df.y.values)
            COLLAB_NPZ.parent.mkdir(exist_ok=True)
            np.savez_compressed(COLLAB_NPZ, edges=np.stack([df.a.values, df.b.values]).astype(np.int32),
                                edge_year=df.y.values.astype(np.int16), node_year=node_year.astype(np.int16),
                                feat=feat)
        z = np.load(COLLAB_NPZ)
        return dict(edges=z["edges"].astype(np.int64), edge_year=z["edge_year"].astype(np.int64),
                    node_year=z["node_year"].astype(np.int64), feat=z["feat"], label=None,
                    base=2012, last=2017, edge_mode="own-year")
    raise ValueError(dataset)


def schedule(d: dict, seed: int, base: int | None = None):
    """Assign every node and edge to a version index (0 = base). `base` overrides the base year."""
    rng = np.random.default_rng(seed)
    year, last = d["node_year"], d["last"]
    base = d["base"] if base is None else base
    node_t = np.full(len(year), -1, dtype=np.int64)
    node_t[year <= base] = 0
    t = 1
    year_start = {}
    for yr in range(base + 1, last + 1):
        year_start[yr] = t
        ids = rng.permutation(np.where(year == yr)[0])
        for part in np.array_split(ids, BATCHES):
            node_t[part] = t
            t += 1
    edges = d["edges"]
    if d["edge_mode"] == "nodes":
        edge_t = np.maximum(node_t[edges[0]], node_t[edges[1]])
    else:
        ey = d["edge_year"]
        own = np.zeros(len(ey), dtype=np.int64)
        late = ey > base
        own[late] = np.array([year_start[y] for y in ey[late]]) + rng.integers(0, BATCHES, late.sum())
        edge_t = np.maximum(np.maximum(node_t[edges[0]], node_t[edges[1]]), own)
    return node_t, edge_t, t


def load_program(path: str):
    spec = importlib.util.spec_from_file_location(f"prog_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(mod)
    return mod.make_maintainer


def replay(program_path: str, d: dict, node_t, edge_t, n_versions: int):
    maint = load_program(program_path)()
    edges, feat, year = d["edges"], d["feat"], d["node_year"]
    labels, times = [], []
    for t in range(n_versions):
        new_nodes = np.where(node_t == t)[0]
        new_edges = edges[:, edge_t == t]
        all_nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        t0 = time.time()
        if t == 0:
            lab = maint.initialize(nodes=new_nodes.copy(), edges=new_edges.copy(),
                                   features=feat[new_nodes].astype(np.float32), years=year[new_nodes].copy())
        else:
            lab = maint.update(new_nodes=new_nodes.copy(), new_edges=new_edges.copy(),
                               new_features=feat[new_nodes].astype(np.float32),
                               new_years=year[new_nodes].copy(), all_nodes=all_nodes.copy())
        times.append(time.time() - t0)
        lab = np.asarray(lab, dtype=np.int64)
        assert lab.shape == (len(all_nodes),), (t, lab.shape, len(all_nodes))
        labels.append(lab)
    return labels, times


def score(labels, d, node_t, edge_t):
    edges, n_total = d["edges"], len(d["node_year"])
    SENT = np.iinfo(np.int64).min
    Q, R, moved, ncomm = [], [], [], []
    R_tau, R_comm = {}, []
    prev_full = None
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        E = edges[:, edge_t <= t]
        remap = -np.ones(n_total, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        G = ig.Graph(n=len(nodes), edges=np.stack([remap[E[0]], remap[E[1]]], 1).tolist())
        _, compact = np.unique(lab, return_inverse=True)
        Q.append(float(G.modularity(compact.tolist())))
        ncomm.append(int(compact.max()) + 1)
        full = np.full(n_total, SENT, dtype=np.int64)
        full[nodes] = lab
        if prev_full is not None:
            ids_new = full[nodes]
            old_mask = prev_full != SENT
            ids_old = prev_full[old_mask]
            uniq, inv_new = np.unique(ids_new, return_inverse=True)
            size_new = np.bincount(inv_new)
            pos = np.clip(np.searchsorted(uniq, ids_old), 0, len(uniq) - 1)
            hit = uniq[pos] == ids_old
            size_old = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(size_old, pos[hit], 1)
            both = old_mask & (prev_full == full)
            inter = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(inter, np.searchsorted(uniq, full[both]), 1)
            jac = inter / np.maximum(size_old + size_new - inter, 1)
            regen = (size_old == 0) | (jac < JACCARD_KEEP)
            R.append(float(size_new[regen].sum() / len(nodes)))
            for tau in TAUS:
                rg = (size_old == 0) | (jac < tau)
                R_tau.setdefault(tau, []).append(float(size_new[rg].sum() / len(nodes)))
            big = size_new >= 2  # communities that would carry a summary (singletons excluded)
            R_comm.append(float((regen & big).sum() / max(big.sum(), 1)))
            moved.append(float((prev_full[old_mask] != full[old_mask]).mean()))
        prev_full = full
    return dict(Q=float(np.mean(Q)), R=float(np.mean(R)), final_Q=Q[-1], moved=float(np.mean(moved)),
                n_comm_final=ncomm[-1], Q_curve=Q, R_curve=R,
                R_tau={str(t): float(np.mean(v)) for t, v in R_tau.items()},
                R_comm=float(np.mean(R_comm)))
