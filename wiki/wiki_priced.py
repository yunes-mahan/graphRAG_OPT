"""Priced rollback on the Wikipedia stream (links and pages can disappear).

Per yearly snapshot: warm-started Leiden from the previous labels (new pages singletons), IDs mapped by
one-to-one majority overlap; 'priced' additionally applies the summary-priced rollback of priced_frontier.rollback
(lambda 0.31). Recompute = Leiden from scratch with overlap ID mapping. Token-level cost exactly as in the
kgversion2 evaluator (summaries for communities >= 5, 1500 + min(8000, 250|S|) tokens, relative to summarising all
communities of the recomputed partition). Seeds 0-4 of Leiden's RNG give 5 runs per method.
"""

import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
HERE = config.RESULTS / "wiki"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "maintainers"))
from priced_frontier import rollback, tokens  # noqa: E402

LAM = 0.31


def graph(snap, nodes):
    pos = {v: i for i, v in enumerate(nodes)}
    e = {(min(pos[int(a)], pos[b]), max(pos[int(a)], pos[b])) for a, s in snap.items() for b in s["links"]
         if b in pos and b != int(a)}
    e = np.array(sorted(e)).T if e else np.zeros((2, 0), int)
    return ig.Graph(n=len(nodes), edges=e.T.tolist()), e


def map_ids(nodes, memb, prev, nxt):
    votes = {}
    for v, c in zip(nodes, memb):
        if v in prev:
            votes.setdefault(c, []).append(prev[v])
    mp, used = {}, set()
    for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        b = max(set(ids), key=ids.count)
        if b not in used:
            mp[c] = b
            used.add(b)
    for c in sorted(set(memb)):
        if c not in mp:
            mp[c] = nxt
            nxt += 1
    return np.array([mp[c] for c in memb]), nxt


def summ_tokens(lab):
    _, cnt = np.unique(lab, return_counts=True)
    return sum(tokens(int(s)) for s in cnt if s >= 5)


def regen_tokens(prev_lab, lab, nodes):
    prev_mem, cur = {}, {}
    for v, c in prev_lab.items():
        prev_mem.setdefault(c, set()).add(v)
    for v, c in zip(nodes, lab.tolist()):
        cur.setdefault(c, set()).add(v)
    tot = 0
    for c, mem in cur.items():
        if len(mem) < 5:
            continue
        old = prev_mem.get(c, set())
        if len(old) < 5 or len(mem & old) / len(mem | old) < 0.9:
            tot += tokens(len(mem))
    return tot


def run(stream, method, seed):
    years = [str(y) for y in stream["years"]]
    prev, nxt, Q, C = {}, 0, [], []
    for t, y in enumerate(years):
        snap = stream["snapshots"][y]
        nodes = sorted(int(k) for k in snap)
        g, e = graph(snap, nodes)
        ref = la.find_partition(g, la.ModularityVertexPartition, seed=seed).membership
        if t == 0 or method == "recompute":
            memb = ref
            init = None
        if t > 0 and method != "recompute":
            init_l, k = [], nxt
            for v in nodes:
                if v in prev:
                    init_l.append(prev[v])
                else:
                    init_l.append(k)
                    k += 1
            _, ic = np.unique(init_l, return_inverse=True)
            memb = la.find_partition(g, la.ModularityVertexPartition, seed=seed, initial_membership=ic.tolist()).membership
        lab, nxt = map_ids(nodes, list(memb), prev, nxt) if t > 0 else (np.array(memb), max(memb) + 1)
        if t > 0 and method == "priced":
            prevarr = np.array([prev.get(v, -1) for v in nodes])
            ga = np.array(nodes)
            lab = rollback(ga, lab, prevarr, ga[e] if e.size else np.zeros((2, 0), int))
        Q.append(g.modularity(np.unique(lab, return_inverse=True)[1].tolist()))
        if t > 0:
            C.append(regen_tokens(prev, lab, nodes) / max(summ_tokens(np.array(ref)), 1))
        prev = {v: int(c) for v, c in zip(nodes, lab.tolist())}
        nxt = max(nxt, int(lab.max()) + 1)
    return float(np.mean(Q)), float(np.mean(C))


def main():
    stream = json.loads((HERE / "stream.json").read_text())
    res = {}
    for method in ("recompute", "warm", "priced"):
        runs = [run(stream, method, s) for s in range(5)]
        q, c = np.mean([r[0] for r in runs]), np.mean([r[1] for r in runs])
        sc = [r[0] - LAM * r[1] for r in runs]
        res[method] = dict(Q=q, C=c, score=float(np.mean(sc)), score_sd=float(np.std(sc)), runs=runs)
        print(f"{method:10s} Q={q:.4f} C={c:.4f} score={np.mean(sc):.4f} +- {np.std(sc):.4f}", flush=True)
    (HERE / "priced_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
