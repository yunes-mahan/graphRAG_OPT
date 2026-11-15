"""W1 (V16): hierarchical frontier-restricted maintainer (ours).

Keeps GraphRAG's hierarchy construction (top level = Leiden on the largest connected component; every community with
more than max_cluster_size members is clustered again one level down) but re-optimises only what an update touches:
  frontier F = new nodes, endpoints of new edges, and their neighbours (as our flat frontier-restricted Leiden);
  level 0: Leiden (leidenalg, modularity, seed fixed) on the LCC, starting from the previous top-level communities
           (new nodes as singletons), with every node outside F held fixed;
  level l+1: for each level-l community C with more than max_cluster_size members:
           - if C has exactly the members of a previous level-l community, its previous subtree is kept unchanged;
           - otherwise C's subgraph is re-optimised the same way (start: previous level-(l+1) communities restricted
             to C, nodes outside F fixed); if that cannot split C, GraphRAG's hierarchical Leiden (graspologic, same
             seed and size bound) clusters C's subgraph from scratch.
Identifiers are matched to the previous version per level by graphrag_hier.match_ids (unchanged communities keep
theirs).  This is NOT HIT-Leiden (Lin et al.); it is our own maintainer of the same kind.
validate(): every present LCC node in exactly one community per level, children inside their parents, leaves within
the size bound (or unsplittable).
"""
from collections import defaultdict

import igraph as ig
import leidenalg as la


def _leiden_fixed(nodes, edges, init, fixed, seed):
    idx = {v: i for i, v in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(idx[a], idx[b]) for a, b in edges])
    labs = {}
    memb = [labs.setdefault(init[v], len(labs)) for v in nodes]
    part = la.ModularityVertexPartition(g, initial_membership=memb)
    opt = la.Optimiser()
    opt.set_rng_seed(seed)
    opt.optimise_partition(part, n_iterations=2, is_membership_fixed=[v in fixed for v in nodes])
    groups = defaultdict(set)
    for v, c in zip(nodes, part.membership):
        groups[c].add(v)
    return list(groups.values())


def update(prev_comms, lcc_edges, lcc_nodes, frontier, maxc, seed, hier_fn):
    """prev_comms: {(level, key): frozenset}; returns {(level, local id): frozenset} (ids matched later)."""
    prev_lv = defaultdict(dict)                     # level -> node -> previous key
    prev_sets = defaultdict(set)                    # level -> set of previous member sets
    for k, m in prev_comms.items():
        for v in m:
            prev_lv[k[0]][v] = k
        prev_sets[k[0]].add(m)
    adj = defaultdict(set)
    for a, b in lcc_edges:
        adj[a].add(b)
        adj[b].add(a)
    out, nid, stats = {}, [0], dict(kept_subtrees=0, reoptimised=0, fallback=0)

    def new_id(level, members):
        nid[0] += 1
        out[(level, ("hf", nid[0]))] = frozenset(members)

    def copy_subtree(members, level):
        done = set()
        for l in sorted(prev_lv):
            if l <= level:
                continue
            for v in members:
                k = prev_lv[l].get(v)
                if k is not None and k not in done:
                    done.add(k)
                    new_id(l, prev_comms[k])

    def cluster(members, level):
        nodes = sorted(members)
        edges = [(a, b) for a in nodes for b in adj[a] if a < b and b in members]
        init = {v: prev_lv[level].get(v, ("new", v)) for v in nodes}
        groups = _leiden_fixed(nodes, edges, init, members - frontier, seed) if edges else [{v} for v in nodes]
        if len(groups) == 1 and len(members) > maxc and level > 0:
            stats["fallback"] += 1
            return "fallback"
        stats["reoptimised"] += 1
        for grp in groups:
            new_id(level, grp)
            if len(grp) > maxc:
                if frozenset(grp) in prev_sets[level]:
                    stats["kept_subtrees"] += 1
                    copy_subtree(frozenset(grp), level)
                elif cluster(set(grp), level + 1) == "fallback":
                    sub = hier_fn([(a, b) for a in grp for b in adj[a] if a < b and b in grp], grp)
                    for (l, c), m in sub.items():
                        new_id(level + 1 + l, m)
        return "ok"

    cluster(set(lcc_nodes), 0)
    return out, stats


def validate(comms, lcc_nodes, maxc):
    """returns counts of violations (all must be 0 except unsplittable oversize leaves, reported separately)."""
    by_level = defaultdict(list)
    for (l, _), m in comms.items():
        by_level[l].append(m)
    bad_partition = 0
    for l, ms in by_level.items():
        seen = defaultdict(int)
        for m in ms:
            for v in m:
                seen[v] += 1
        bad_partition += sum(1 for v, c in seen.items() if c > 1)
    top = set().union(*by_level.get(0, [set()]))
    missing_top = len(set(lcc_nodes) - top)
    bad_nesting = 0
    for l in sorted(by_level):
        if l == 0:
            continue
        owner = {}
        for m in by_level[l - 1]:
            for v in m:
                owner[v] = m
        for m in by_level[l]:
            parents = {owner.get(v) for v in m}
            bad_nesting += int(len(parents) != 1 or None in parents)
    leaves = [m for l, ms in by_level.items() for m in ms
              if not any(c < m for c in by_level.get(l + 1, []))]
    oversize = sum(1 for m in leaves if len(m) > maxc)
    return dict(bad_partition=bad_partition, missing_top=missing_top, bad_nesting=bad_nesting, oversize_leaves=oversize)


def complete(comms):
    """GraphRAG's complete_levels on community sets: a community that is not split further is repeated on every
    deeper level (it carries a report there too), so all maintainers are counted the same way."""
    by_level = defaultdict(list)
    for (l, _), m in comms.items():
        by_level[l].append(m)
    if not by_level:
        return {}
    L = max(by_level)
    out, n = {}, 0
    for l in range(L + 1):
        cur = list(by_level.get(l, []))
        if l > 0:
            covered = set().union(*cur) if cur else set()
            cur += [m for m in prev if not (m & covered)]
        for m in cur:
            n += 1
            out[(l, ("hf", n))] = m
        prev = cur
    return out
