"""B1 (V12) batch-size sweep on the flat arXiv stream (b1_streams.arxiv_stream: version 0 = papers up to 2019, then N=10
updates of B in {10, 100, 1000} papers from 2020; seeds 0-2).
Maintainers (analysis/run_suite.SPECS, unchanged): recompute_leiden (re-index + identifier matching), warmstart_leiden,
frontier_leiden (ours), graphrag_delta.  Sahu's DF Leiden is added by b1_df.py (container).
Per update: report-token cost relative to the tokens of all reports of the recomputed partition (tok(S) =
1500 + min(8000, 250|S|), communities >= 5 members) for tau = 0.9 (rewrite if new id or Jaccard < 0.9, as
tasks/kgversion2/evaluate.score_stream) and "never" (rewrite only communities that newly carry a report);
mean modularity; coverage (share of present nodes in a community with >= 5 members).
Output: analysis/v12/b1/flat_arxiv.json        usage: python v12/b1_flat_sweep.py
"""
import json
import os
import sys
from pathlib import Path

import igraph as ig
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE)]
import b1_streams as bs  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

MAINT = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "graphrag_delta"]
ENVK = ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "TOK_OVERHEAD", "TOK_PER", "TOK_CAP", "PRICE_LAMBDA")


def tok(sizes):
    s = np.asarray(sizes, float)
    return float((1500 + np.minimum(8000, 250 * s)).sum())


def score(labels, d, node_t, edge_t):
    """sizes of reported communities per version and of rewritten ones per update, for tau = 0.9 and never."""
    edges, n_total = d["edges"], len(d["node_year"])
    SENT = np.iinfo(np.int64).min
    Q, full, rg9, rg0, cov = [], [], [], [], []
    prev = None
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        E = edges[:, edge_t <= t]
        remap = -np.ones(n_total, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        G = ig.Graph(n=len(nodes), edges=np.stack([remap[E[0]], remap[E[1]]], 1).tolist())
        uniq, inv = np.unique(lab, return_inverse=True)
        Q.append(float(G.modularity(inv.tolist())))
        size = np.bincount(inv)
        big = size >= 5
        full.append(size[big].tolist())
        cov.append(float(size[big].sum() / len(nodes)))
        fl = np.full(n_total, SENT, dtype=np.int64)
        fl[nodes] = lab
        if prev is not None:
            om = prev != SENT
            pos = np.clip(np.searchsorted(uniq, prev[om]), 0, len(uniq) - 1)
            hit = uniq[pos] == prev[om]
            so = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(so, pos[hit], 1)
            both = om & (prev == fl)
            inter = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(inter, np.searchsorted(uniq, fl[both]), 1)
            jac = inter / np.maximum(so + size - inter, 1)
            rg9.append(size[big & ((so < 5) | (jac < 0.9))].tolist())
            rg0.append(size[big & (so < 5)].tolist())
        prev = fl
    return dict(Q=Q, full=full, rg9=rg9, rg0=rg0, cov=cov)


if __name__ == "__main__":
    out_f = HERE / "b1" / "flat_arxiv.json"
    res = json.loads(out_f.read_text()) if out_f.exists() else {}
    for B in (1000, 100, 10):
        for seed in (0, 1, 2):
            d, node_t, edge_t, T = bs.arxiv_stream(B, 10, seed)
            sz = {}
            for m in MAINT:
                key = f"{B}|{seed}|{m}"
                path, env = rs.SPECS[m]
                for k in ENVK:
                    os.environ.pop(k, None)
                os.environ.update(env)
                labels, _ = se.replay(str(path), d, node_t, edge_t, T)
                sz[m] = score(labels, d, node_t, edge_t)
            ref = sz["recompute_leiden"]["full"]
            for m in MAINT:
                r = sz[m]
                res[f"{B}|{seed}|{m}"] = dict(
                    C09=float(np.mean([tok(x) / tok(f) for x, f in zip(r["rg9"], ref[1:])])),
                    Cnever=float(np.mean([tok(x) / tok(f) for x, f in zip(r["rg0"], ref[1:])])),
                    Q=float(np.mean(r["Q"])), coverage=float(np.mean(r["cov"])),
                    new_nodes=float(np.mean([int((node_t == t).sum()) for t in range(1, T)])))
                print(B, seed, m, {k: round(v, 4) for k, v in res[f"{B}|{seed}|{m}"].items()}, flush=True)
            out_f.write_text(json.dumps(res, indent=1))
