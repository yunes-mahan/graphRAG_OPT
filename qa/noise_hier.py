"""Instability of GraphRAG's hierarchical clustering (graspologic hierarchical Leiden, max cluster size 10, LCC).

On the final graph of the entity and Wikidata streams (order 0), every community on every level carries a report.
A perturbed clustering is matched level by level to the reference clustering (GraphRAG's fixed seed 0xDEADBEEF) by
one-to-one overlap, exactly as the re-index + matching maintainer does, and we report the share of report tokens
(tok = 1500 + min(8000, 250|S|)) whose community is new or has Jaccard < tau (tau = 0.9 and 0.5):
  drop p : GraphRAG seed, a random share p of the edges deleted, p in {0.001, 0.01} (3 repetitions)
  seed   : seeds 1..3 on the unchanged graph (operators with a fixed seed never see this; for reference)
  last   : GraphRAG seed, graph before the last update vs after it
Writes noise_hier.json.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402


def cluster(edges, n, seed=gh.SEED):
    gh.SEED = seed
    lcc, _ = gh.lcc_edges(edges, np.arange(n))
    out = gh.communities(gh.complete_levels(gh.hier(lcc, list(range(n)))[0]))
    gh.SEED = 0xDEADBEEF
    return out


def rewrite(ref, new, tau):
    mp = gh.match_ids(ref, new)
    tot = cost = 0
    for k, m in new.items():
        tot += gh.tok(len(m))
        old = ref.get(mp.get(k))
        if old is None or len(m & old) / len(m | old) < tau:
            cost += gh.tok(len(m))
    return cost / tot


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
        ref = cluster(el, n)
        r = {}
        rng = np.random.default_rng(0)
        for p in (0.001, 0.01):
            vals = []
            for _ in range(3):
                keep = rng.random(len(el)) >= p
                vals.append({tau: rewrite(ref, cluster([e for e, k in zip(el, keep) if k], n), tau) for tau in (0.9, 0.5)})
            r[f"drop{p}"] = {str(t): [v[t] for v in vals] for t in (0.9, 0.5)}
        seeds = [cluster(el, n, s) for s in (1, 2, 3)]
        r["seed"] = {str(t): [rewrite(ref, s, t) for s in seeds] for t in (0.9, 0.5)}
        before = [e for e, t in zip(el, edge_t[ok]) if t < T - 1]
        prev = cluster(before, n)
        r["last"] = {str(t): rewrite(prev, ref, t) for t in (0.9, 0.5)}
        out[graph] = r
        print(graph, json.dumps(r), flush=True)
        (HERE / "noise_hier.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
