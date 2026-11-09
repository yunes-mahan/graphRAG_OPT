"""Sanity checks for full recomputation (no LLM calls).

On the final graph of each stream (schedule order 0):
  seed     : Leiden with seeds 1..4 versus seed 0 on the identical graph
  drop p   : Leiden (seed 0) after deleting a random share p of the edges, p in {0.001, 0.01}
  last     : Leiden (seed 0) on the graph after the last update versus before it (what recompute pays)
Each new partition is matched to the seed-0 partition exactly as recompute does (greedy majority overlap) and we
report the share of summary tokens (communities >= 5 members, tok = 1500 + min(8000, 250|S|)) that the identifier
rule with tau = 0.9 would rewrite, plus NMI. Also: new nodes and edges per version (batch sizes).
Writes noise_floor.json.
"""

import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import entity_graph_eval as eg  # noqa: E402
import stream_eval as se  # noqa: E402


def tok(n):
    return 1500 + min(8000, 250 * n)


def leiden(n, edges, seed):
    g = ig.Graph(n=n, edges=edges.T.tolist())
    return np.array(la.find_partition(g, la.ModularityVertexPartition, seed=seed).membership)


def rewrite_share(ref, new, tau=0.9, ms=5):
    """share of summary tokens of `new` rewritten after matching its ids to `ref` (greedy majority overlap)."""
    refc, newc = {}, {}
    for v, c in enumerate(ref):
        refc.setdefault(int(c), set()).add(v)
    for v, c in enumerate(new):
        newc.setdefault(int(c), set()).add(v)
    votes = {c: [int(ref[v]) for v in m] for c, m in newc.items()}
    mp, used = {}, set()
    for c, ids in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        b = max(set(ids), key=ids.count)
        if b not in used:
            mp[c] = b
            used.add(b)
    tot = cost = 0
    for c, m in newc.items():
        if len(m) < ms:
            continue
        tot += tok(len(m))
        old = refc.get(mp.get(c))
        if old is None or len(m & old) / len(m | old) < tau:
            cost += tok(len(m))
    return cost / max(tot, 1)


def final_graph(name):
    if name == "entity":
        d, node_t, edge_t, T = eg.stream(0)
    else:
        d = se.load(name)
        node_t, edge_t, T = se.schedule(d, 0)
    nodes = np.where(node_t >= 0)[0]
    idx = -np.ones(len(node_t), dtype=np.int64)
    idx[nodes] = np.arange(len(nodes))
    E = d["edges"]
    ok = (edge_t >= 0) & (edge_t < T) & (node_t[E[0]] >= 0) & (node_t[E[1]] >= 0)
    E, et = E[:, ok], edge_t[ok]
    batches = [dict(t=t, new_nodes=int((node_t == t).sum()), new_edges=int((et == t).sum())) for t in range(T)]
    return len(nodes), idx[E], et, T, batches


def main():
    out = {}
    for name in ("entity", "collab", "arxiv"):
        n, E, et, T, batches = final_graph(name)
        rng = np.random.default_rng(0)
        p0 = leiden(n, E, 0)
        r = dict(n=n, m=int(E.shape[1]), versions=T, batches=batches)
        r["seed"] = [dict(seed=s, rewrite=rewrite_share(p0, p := leiden(n, E, s)), nmi=float(nmi(p0, p))) for s in range(1, 5)]
        for p in (0.001, 0.01):
            keep = rng.random(E.shape[1]) >= p
            q = leiden(n, E[:, keep], 0)
            r[f"drop{p}"] = dict(rewrite=rewrite_share(p0, q), nmi=float(nmi(p0, q)))
        before = E[:, et < T - 1]
        q = leiden(n, before, 0)
        r["last"] = dict(rewrite=rewrite_share(q, p0), nmi=float(nmi(q, p0)), new_edges=batches[-1]["new_edges"])
        out[name] = r
        print(name, "seed:", [round(x["rewrite"], 3) for x in r["seed"]], "drop:",
              {k: round(r[k]["rewrite"], 3) for k in ("drop0.001", "drop0.01")}, "last:", round(r["last"]["rewrite"], 3),
              "| smallest batches:", sorted(b["new_edges"] for b in batches[1:])[:4], flush=True)
        (HERE / "noise_floor.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
