"""W5 (V16): is the instability of GraphRAG's hierarchical clustering caused by Leiden itself or by its configuration?
Final graph (order 0) of the entity graph and Wikidata, GraphRAG settings (graspologic hierarchical Leiden, max cluster
size 10, LCC, seed 0xDEADBEEF), report rule tau = 0.9, rewrite share of report tokens as noise_hier.rewrite.
  base   : GraphRAG default (iterations = 1): 0.1 % edge deletion (3 draws, same rng as noise_hier) and the real last
           update, same-level one-to-one matching  (must reproduce noise_hier.json)
  (a) perm : same graph, node identifiers and edge order permuted (3 permutations), GraphRAG seed -> rewrite share
  (b) conv : Leiden iterated (iterations = ITER) instead of 1, for the reference AND the perturbed clustering;
             convergence is checked by comparing ITER with 2 x ITER on the reference; modularity leaf/top
  (c) xlevel : matching a community to a community of ANY level of the other clustering (one-to-one, greedy by
             Jaccard), for the base clusterings
Output: analysis/v16/w5_instability.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import graspologic_native as gn
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata")]
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import noise_hier as nh  # noqa: E402
import stability_variants as sv  # noqa: E402

ITER = 50
_orig = gn.hierarchical_leiden
_IT = [1]


def _hl(*a, **k):
    if _IT[0] != 1:
        k["iterations"] = _IT[0]
    return _orig(*a, **k)


gn.hierarchical_leiden = _hl
gh.gn.hierarchical_leiden = _hl


def rewrite_xlevel(ref, new, tau):
    owner = {}
    for k, m in ref.items():
        for v in m:
            owner.setdefault(v, []).append(k)
    pairs = []
    for k, m in new.items():
        cnt = Counter(p for v in m for p in owner.get(v, ()))
        for p, c in cnt.items():
            pairs.append((c / (len(m) + len(ref[p]) - c), k, p))
    mp, used = {}, set()
    for j, k, p in sorted(pairs, key=lambda x: (-x[0], str(x[1]), str(x[2]))):
        if k not in mp and p not in used:
            mp[k] = p
            used.add(p)
    tot = cost = 0
    for k, m in new.items():
        tot += gh.tok(len(m))
        old = ref.get(mp.get(k))
        if old is None or len(m & old) / len(m | old) < tau:
            cost += gh.tok(len(m))
    return cost / tot


def permuted(el, n, rng):
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    pe = [(int(perm[a]), int(perm[b])) for a, b in el]
    order = rng.permutation(len(pe))
    pe = [pe[i] for i in order]
    c = nh.cluster(pe, n)
    return {k: frozenset(int(inv[v]) for v in m) for k, m in c.items()}


def main():
    out = {}
    for graph in ("entity", "wikidata"):
        if graph == "wikidata":
            import stream_eval as se
            import wd_eval as wd
            d = wd.build()[0]
            node_t, edge_t, T = se.schedule(d, 0)
        else:
            d, node_t, edge_t, T = eg.stream(0)
        n, E = len(node_t), d["edges"]
        ok = (edge_t >= 0) & (edge_t < T)
        el = [(int(a), int(b)) for a, b in E[:, ok].T]
        before = [e for e, t in zip(el, edge_t[ok]) if t < T - 1]
        rng = np.random.default_rng(0)
        drops = [[e for e, k in zip(el, rng.random(len(el)) >= 0.001) if k] for _ in range(3)]
        r = {}
        _IT[0] = 1
        ref, prev = nh.cluster(el, n), nh.cluster(before, n)
        dc = [nh.cluster(x, n) for x in drops]
        r["base"] = dict(drop=[nh.rewrite(ref, c, 0.9) for c in dc], last=nh.rewrite(prev, ref, 0.9),
                         Q=sv.quality(ref, el, n))
        r["xlevel"] = dict(drop=[rewrite_xlevel(ref, c, 0.9) for c in dc], last=rewrite_xlevel(prev, ref, 0.9))
        prng = np.random.default_rng(1)
        r["perm"] = dict(same_graph=[nh.rewrite(ref, permuted(el, n, prng), 0.9) for _ in range(3)])
        print(graph, "base/xlevel/perm", json.dumps({k: r[k] for k in ("base", "xlevel", "perm")}, default=str)[:600], flush=True)
        _IT[0] = ITER
        cref, cprev = nh.cluster(el, n), nh.cluster(before, n)
        _IT[0] = 2 * ITER
        cref2 = nh.cluster(el, n)
        _IT[0] = ITER
        r["conv"] = dict(iterations=ITER, drop=[nh.rewrite(cref, nh.cluster(x, n), 0.9) for x in drops],
                         last=nh.rewrite(cprev, cref, 0.9), Q=sv.quality(cref, el, n),
                         converged_vs_2x=nh.rewrite(cref, cref2, 0.9))
        _IT[0] = 1
        out[graph] = r
        print(graph, "conv", json.dumps(r["conv"], default=str)[:400], flush=True)
        (HERE / "w5_instability.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
