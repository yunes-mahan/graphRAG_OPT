"""Typed Wikidata film KG stream: maintainers under the token-level score (transfer test, lambda = 0.31 unchanged).

Graph: films (2000-2023) and the people, companies and genres connected by director, cast member, screenwriter,
genre and production company relations (undirected). A film arrives in its release year, an entity with its
first film. Stream: V0 = everything up to 2015, then 2016-2023 in 3 random batches per year (schedule seeds 0-4).
Score per stream: mean modularity - 0.31 * mean relative token cost (tokens as in tasks/kgversion2; reference =
summarising all communities of Leiden recomputed on that version). Writes wikidata/eval.json.
"""

import json
import os
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

LAM, BASE, LAST = 0.31, 2015, 2023
METHODS = ["recompute_leiden", "lazy", "warmstart_leiden", "frontier_leiden", "hybrid", "priced_warm", "priced_frontier", "graphrag_delta"]


def tokens(size):
    return 1500 + np.minimum(8000, 250 * size)


def build():
    d = json.loads((HERE / "films.json").read_text())
    films = d["films"]
    idx, year, typ = {}, [], []

    def node(q, y, t):
        if q not in idx:
            idx[q] = len(year)
            year.append(y)
            typ.append(t)
        else:
            year[idx[q]] = min(year[idx[q]], y)
        return idx[q]
    for f, y in films.items():
        node(f, y, "film")
    E, R = [], []
    for f, rel, o in d["edges"]:
        if f in films and o != f:
            a = idx[f]
            b = node(o, films[f], {"director": "person", "cast": "person", "screenwriter": "person"}.get(rel, rel))
            E.append((min(a, b), max(a, b)))
            R.append(rel)
    E = np.array(sorted(set(E))).T
    year = np.array(year)
    return dict(edges=E, node_year=year, edge_year=np.maximum(year[E[0]], year[E[1]]),
                feat=np.zeros((len(year), 8), np.float32), label=None, base=BASE, last=LAST, edge_mode="nodes"), typ, R


def score(labels, d, node_t, edge_t, ref=None):
    edges, n = d["edges"], len(d["node_year"])
    SENT = np.iinfo(np.int64).min
    Q, regen, full, prev = [], [], [], None
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        Et = edges[:, edge_t <= t]
        remap = -np.ones(n, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        G = ig.Graph(n=len(nodes), edges=np.stack([remap[Et[0]], remap[Et[1]]], 1).tolist())
        uniq, inv = np.unique(lab, return_inverse=True)
        Q.append(G.modularity(inv.tolist()))
        size = np.bincount(inv)
        big = size >= 5
        full.append(float(tokens(size[big]).sum()))
        fl = np.full(n, SENT, dtype=np.int64)
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
            rg = big & ((so < 5) | (jac < 0.9))
            regen.append(float(tokens(size[rg]).sum()))
        prev = fl
    out = dict(Q=float(np.mean(Q)), full=full)
    if ref is not None:
        out["C"] = float(np.mean([r / f for r, f in zip(regen, ref[1:])]))
    return out


def main():
    d, typ, R = build()
    print(f"nodes {len(typ)} ({sum(t == 'film' for t in typ)} films), edges {d['edges'].shape[1]}; "
          f"relation counts { {r: R.count(r) for r in set(R)} }", flush=True)
    f = HERE / "eval.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for seed in range(5):
        node_t, edge_t, T = se.schedule(d, seed)
        ref = None
        for m in METHODS:
            key = f"{m}|{seed}"
            if key in res and m != "recompute_leiden":
                continue
            path, env = rs.SPECS[m]
            for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "PRICE_LAMBDA"):
                os.environ.pop(k, None)
            os.environ.update(env)
            labels, times = se.replay(str(path), d, node_t, edge_t, T)
            if m == "recompute_leiden":
                ref = score(labels, d, node_t, edge_t)["full"]
            r = score(labels, d, node_t, edge_t, ref)
            res[key] = dict(Q=r["Q"], C=r["C"], score=r["Q"] - LAM * r["C"], seconds=float(sum(times)))
            f.write_text(json.dumps(res, indent=1))
            print(seed, m, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
    print("\nmeans over 5 orders:")
    for m in METHODS:
        v = [res[f"{m}|{s}"] for s in range(5)]
        sc = [x["score"] for x in v]
        print(f"{m:18s} score={np.mean(sc):.4f} +- {np.std(sc):.4f}  Q={np.mean([x['Q'] for x in v]):.4f} "
              f"C={np.mean([x['C'] for x in v]):.4f}")
    (HERE / "meta.json").write_text(json.dumps(dict(nodes=len(typ), films=sum(t == "film" for t in typ),
                                                    edges=int(d["edges"].shape[1]),
                                                    relations={r: R.count(r) for r in set(R)},
                                                    node_types={t: typ.count(t) for t in set(typ)})))


if __name__ == "__main__":
    main()
