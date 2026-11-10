"""Hierarchical consensus clustering (Lancichinetti & Fortunato 2012) with GraphRAG's settings.

consensus(edges, n, runs, reattach): `runs` hierarchical Leiden runs (seeds 0..runs-1, GraphRAG settings) on the
largest connected component; every LCC edge is weighted by the share of runs that put its endpoints in the same leaf;
edges below 0.5 are dropped and hierarchical Leiden (GraphRAG's seed) clusters the weighted consensus graph (all its
components). With reattach=True, LCC entities that lost all their edges to the threshold are reattached, repeatedly,
to every community (on every level) of their covered neighbour with the highest agreement (ties: smaller id), so
coverage equals that of the input clustering.
"""

from collections import defaultdict

import numpy as np


def leaves(comms):
    leaf = {}
    for (l, c), m in comms.items():
        for v in m:
            if l >= leaf.get(v, (-1, None))[0]:
                leaf[v] = (l, c)
    return leaf


def consensus(edges, n, runs=10, reattach=False):
    import graphrag_hier as gh
    import noise_hier as nh
    lcc, _ = gh.lcc_edges(edges, np.arange(n))
    agree = defaultdict(int)
    for s in range(runs):
        lf = leaves(nh.cluster(edges, n, s))
        for a, b in lcc:
            agree[(a, b)] += lf.get(a) == lf.get(b)
    keep = [(a, b) for a, b in lcc if agree[(a, b)] / runs >= 0.5]
    e = [(str(a), str(b), float(agree[(a, b)] / runs)) for a, b in keep]
    res = gh.gn.hierarchical_leiden(e, max_cluster_size=gh.MAXC, seed=0xDEADBEEF, use_modularity=True)
    lev = defaultdict(dict)
    for h in res:
        lev[h.level][int(h.node)] = h.cluster
    comms = gh.communities(gh.complete_levels(dict(lev)))
    if not reattach:
        return comms
    comms = {k: set(m) for k, m in comms.items()}
    member_of = defaultdict(list)
    for k, m in comms.items():
        for v in m:
            member_of[v].append(k)
    adj = defaultdict(list)
    for a, b in lcc:
        w = agree[(a, b)] / runs
        adj[a].append((w, b))
        adj[b].append((w, a))
    todo = {v for v in adj if v not in member_of}
    while todo:
        done = []
        for v in sorted(todo):
            cand = [(-w, u) for w, u in adj[v] if u in member_of]
            if not cand:
                continue
            u = min(cand)[1]
            for k in member_of[u]:
                comms[k].add(v)
            done.append((v, list(member_of[u])))
        if not done:
            break
        for v, ks in done:
            member_of[v] = ks
            todo.discard(v)
    return {k: frozenset(m) for k, m in comms.items()}
