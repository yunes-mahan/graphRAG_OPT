"""GraphRAG's real community structure on the LLM-extracted entity graph (structural, no LLM calls).

Clustering exactly as Microsoft GraphRAG 3.x (cluster_graph): graspologic hierarchical_leiden with
max_cluster_size = 10, use_lcc = True (only the largest connected component is clustered), seed 0xDEADBEEF.
Every community on every level carries a report.
Maintainers (per update):
  reindex        : full GraphRAG re-index; communities get fresh ids, every report is rewritten
  reindex+match  : full re-clustering, ids matched to the previous version by one-to-one majority overlap per level
  graphrag_update: GraphRAG's incremental update: hierarchical Leiden on the delta graph only (LCC of the delta),
                   new communities appended with fresh ids, old communities and reports kept.  The delta graph holds
                   every relationship extracted from the batch's documents, including relationships restated
                   between existing entities (entity graph); on Wikidata every edge of the new films
  warm           : hierarchical Leiden warm-started from the previous final clusters (new nodes singletons),
                   ids matched by overlap per level
  zep_attach     : Zep's dynamic extension: a new entity joins the leaf community of the plurality of its
                   neighbours (and that leaf's ancestors); no re-clustering
Report rule: a report is rewritten if its community is new or Jaccard(members at last report, members now) < tau,
tau in {0.9, 0.7, 0.5, never}; only communities with >= min_size members carry reports, min_size in {1, 5, 10}.
Cost: tokens(S) = 1500 + min(8000, 250 |S|) per rewritten report, relative to a full re-index at that version.
Quality: modularity of the leaf partition and of the top level (on the clustered nodes).
Writes graphrag_hier.json.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import graspologic_native as gn
import igraph as ig
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import os  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
GRAPH = os.environ.get("HIER_GRAPH", "entity")   # entity | wikidata

SEED, MAXC = 0xDEADBEEF, int(os.environ.get("HIER_MAXC", "10"))
TAUS = [0.9, 0.7, 0.5, 0.0]
ONLY = [x for x in os.environ.get("HIER_ONLY", "").split(",") if x]           # e.g. "consensus"
CONS_RUNS = int(os.environ.get("HIER_CONS_RUNS", "100"))
CONS_REATTACH = os.environ.get("HIER_CONS_REATTACH", "1") == "1"
CONS_MODE = os.environ.get("HIER_CONS_MODE", "threshold")
MINS = [1, 5, 10]


def entity_batch_rels(seed, T):
    """relationship indices extracted from the documents of each batch (same document order as eg.stream)."""
    g = json.loads((eg.G / "graph.json").read_text())
    rng = np.random.default_rng(seed)
    years = {d: v["year"] for d, v in g["docs"].items()}
    doc_t, t = {}, 1
    for d, y in years.items():
        if y <= eg.BASE:
            doc_t[d] = 0
    for y in range(eg.BASE + 1, eg.LAST + 1):
        ids = rng.permutation(sorted(d for d, yy in years.items() if yy == y))
        for part in np.array_split(ids, eg.BATCHES):
            for d in part:
                doc_t[d] = t
            t += 1
    out = [set() for _ in range(T)]
    for d, v in g["docs"].items():
        out[doc_t[d]].update(v["rels"])
    return out


def tok(n):
    return 1500 + min(8000, 250 * n)


# VLDB revision (B3/B6): HIER_TOKVARIANTS=1 also records the cost of the SAME rewrite decisions under the cost-model
# variants below (each constant -50%/+50%); HIER_OUT redirects the output file and disables the store dump.
# Without these variables the behaviour and outputs are unchanged.
TOKVARIANTS = {"a-50": (750, 250, 8000), "a+50": (2250, 250, 8000), "b-50": (1500, 125, 8000),
               "b+50": (1500, 375, 8000), "c-50": (1500, 250, 4000), "c+50": (1500, 250, 12000)} \
    if os.environ.get("HIER_TOKVARIANTS") else {}


def tokv(n, v):
    return v[0] + min(v[2], v[1] * n)


def lcc_edges(edges, nodes):
    g = ig.Graph(n=len(nodes), edges=edges)
    comp = g.connected_components()
    big = max(comp, key=len)
    keep = set(big)
    return [(a, b) for a, b in edges if a in keep and b in keep], keep


def hier(edge_list, node_ids, start=None):
    """edge_list over local indices; returns {level: {node: cluster}}, parent map {(level,cluster): parent}."""
    if not edge_list:
        return {}, {}
    e = [(str(node_ids[a]), str(node_ids[b]), 1.0) for a, b in edge_list]
    kw = dict(max_cluster_size=MAXC, seed=SEED, use_modularity=True)
    if os.environ.get("HIER_ITER"):                     # V16/W5: Leiden iterations (GraphRAG default 1)
        kw["iterations"] = int(os.environ["HIER_ITER"])
    if start is not None:
        present = {x for a, b, _ in e for x in (a, b)}
        kw["starting_communities"] = {k: v for k, v in start.items() if k in present}
    res = gn.hierarchical_leiden(e, **kw)
    lev = defaultdict(dict)
    parent = {}
    for h in res:
        lev[h.level][int(h.node)] = h.cluster
        if h.parent_cluster is not None:
            parent[(h.level, h.cluster)] = h.parent_cluster
    return dict(lev), parent


def complete_levels(lev):
    """GraphRAG reports per level: a node not split further keeps its cluster on deeper levels."""
    if not lev:
        return {}
    L = max(lev)
    out, cur = {}, {}
    for l in range(L + 1):
        cur = {**cur, **lev.get(l, {})}
        out[l] = dict(cur)
    return out


def communities(levels):
    """{(level, cid): frozenset(members)}"""
    out = defaultdict(set)
    for l, m in levels.items():
        for v, c in m.items():
            out[(l, c)].add(v)
    return {k: frozenset(v) for k, v in out.items()}


def match_ids_hungarian(prev, cur):
    """V12/B2: optimal one-to-one matching per level maximising the total Jaccard similarity (Hungarian algorithm,
    solved per connected component of the overlap graph, which is exact because components share no members)."""
    from scipy.optimize import linear_sum_assignment
    owner = {}
    for k, m in prev.items():
        for v in m:
            owner[(k[0], v)] = k
    pairs = {}
    for k, m in cur.items():
        cnt = Counter(owner.get((k[0], v)) for v in m)
        cnt.pop(None, None)
        for p, n in cnt.items():
            pairs[(k, p)] = n / (len(m) + len(prev[p]) - n)
    adj = defaultdict(set)
    for (k, p) in pairs:
        adj[("c", k)].add(("p", p))
        adj[("p", p)].add(("c", k))
    mp, seen = {}, set()
    for start in adj:
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        cs = [x[1] for x in comp if x[0] == "c"]
        ps = [x[1] for x in comp if x[0] == "p"]
        W = np.zeros((len(cs), len(ps)))
        for i, k in enumerate(cs):
            for j, p in enumerate(ps):
                W[i, j] = pairs.get((k, p), 0.0)
        ri, ci = linear_sum_assignment(-W)
        for i, j in zip(ri, ci):
            if W[i, j] > 0:
                mp[cs[i]] = ps[j]
    return mp


def match_ids(prev, cur):
    """one-to-one majority-overlap matching per level: cur key -> prev key (or new key)."""
    if os.environ.get("HIER_MATCH") == "hungarian":
        return match_ids_hungarian(prev, cur)
    mp, used = {}, set()
    pairs = []
    by_level = defaultdict(list)
    for k in prev:
        by_level[k[0]].append(k)
    owner = {}
    for k, m in prev.items():
        for v in m:
            owner[(k[0], v)] = k
    for k, m in cur.items():
        cnt = Counter(owner.get((k[0], v)) for v in m)
        cnt.pop(None, None)
        for p, n in cnt.items():
            pairs.append((n, k, p))
    for n, k, p in sorted(pairs, key=lambda x: (-x[0], str(x[1]), str(x[2]))):
        if k not in mp and p not in used:
            mp[k] = p
            used.add(p)
    return mp


def modularity(g_all, nodes_idx, part):
    sub = [v for v in nodes_idx if v in part]
    if not sub:
        return float("nan")
    g = g_all.induced_subgraph(sub)
    _, comp = np.unique([part[v] for v in sub], return_inverse=True)
    return g.modularity(comp.tolist())


def run(seed):
    if GRAPH == "wikidata":
        import stream_eval as se
        import wd_eval as wd
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, seed)
    elif os.environ.get("HIER_SMALLBATCH"):                 # V12/B1: small-batch sweep stream "B,N" (entity graph)
        sys.path.insert(0, str(HERE / "v12"))                 # revision/v12 copied into $GCM_RESULTS/v12 (revision/README.md)
        import b1_streams
        bb, nn = (int(x) for x in os.environ["HIER_SMALLBATCH"].split(","))
        d, node_t, edge_t, T, batch_rels = b1_streams.entity_stream(bb, nn, seed)
    else:
        d, node_t, edge_t, T = eg.stream(seed)
        batch_rels = entity_batch_rels(seed, T)
    E = d["edges"]
    res = {}
    state = {}
    for t in range(T):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        emask = edge_t <= t
        el = [(int(a), int(b)) for a, b in E[:, emask].T]
        g_all = ig.Graph(n=len(node_t), edges=el)
        lcc, keep = lcc_edges(el, np.arange(len(node_t)))
        idx = {v: v for v in range(len(node_t))}
        fresh_lv = complete_levels(hier(lcc, list(range(len(node_t))))[0])
        fresh = communities(fresh_lv)
        full_tok = {ms: sum(tok(len(m)) for m in fresh.values() if len(m) >= ms) for ms in MINS}
        full_tokv = {(vn, ms): sum(tokv(len(m), v) for m in fresh.values() if len(m) >= ms)
                     for vn, v in TOKVARIANTS.items() for ms in MINS}
        new_nodes = set(np.where(node_t == t)[0].tolist())
        if os.environ.get("HIER_EXACT"):                  # V16/W2+W3: GraphRAG-format context tokens (v16/exact_ctx.py)
            XR = np.load(HERE / "v16" / "w3_rows.npz")
            xadj = defaultdict(list)
            for j in np.where(emask)[0]:
                a_, b_ = int(E[0, j]), int(E[1, j])
                xadj[a_].append((b_, int(j)))
                xadj[b_].append((a_, int(j)))
            _xc = {}

            def xctx(m, _xc=_xc, xadj=xadj, XR=XR):
                if m not in _xc:
                    _xc[m] = int(XR["hdr"]) + int(XR["erow"][list(m)].sum()) + sum(
                        int(XR["edrow"][j]) for v in m for u, j in xadj.get(v, ()) if v < u and u in m)
                return _xc[m]

            def xtok(m, XR=XR):
                return float(XR["prompt"]) + min(xctx(m), int(XR["limit"])) + float(XR["out"])
            xfull = sum(xtok(m) for m in fresh.values())
        new_el = [(int(a), int(b)) for a, b in E[:, edge_t == t].T]
        delta_el = sorted({(int(E[0, j]), int(E[1, j])) for j in batch_rels[t]}) if GRAPH == "entity" else new_el
        for name in (ONLY or ("reindex", "reindex+match", "graphrag_update", "warm", "zep_attach")):
            st = state.setdefault(name, dict(comms=None, store={tau: {} for tau in TAUS}, nextid=0, cost=defaultdict(list),
                                             Qleaf=[], Qtop=[], nrep=[]))
            if name == "kcore":                                       # k-core hierarchy (BFS split) + id matching (V13/T2)
                import stability_variants as sv
                cc = sv.kcore(el, len(node_t))
                if st["comms"] is None:
                    comms = {(k[0], ("s", k[1])): m for k, m in cc.items()}
                else:
                    mp = match_ids(st["comms"], cc)
                    comms = {(mp.get(k, (k[0], ("s", t, k[1])))): m for k, m in cc.items()}
                lv = None
            elif name == "kcore_authors":                             # VLDB revision B2: authors' k-core code + id matching
                sys.path.insert(0, str(HERE / "v2"))
                import b2_kcore_authors as ka
                cc = ka.authors_kcore(el, len(node_t), os.environ.get("KCORE_TYPE", "RkH"))
                if st["comms"] is None:
                    comms = {(k[0], ("s", k[1])): m for k, m in cc.items()}
                else:
                    mp = match_ids(st["comms"], cc)
                    comms = {(mp.get(k, (k[0], ("s", t, k[1])))): m for k, m in cc.items()}
                lv = None
            elif name == "consensus":                                 # consensus clustering + id matching
                import consensus_lib as cl
                cc = cl.consensus(el, len(node_t), CONS_RUNS, CONS_REATTACH, CONS_MODE)
                if st["comms"] is None:
                    comms = {(k[0], ("s", k[1])): m for k, m in cc.items()}
                else:
                    mp = match_ids(st["comms"], cc)
                    comms = {(mp.get(k, (k[0], ("s", t, k[1])))): m for k, m in cc.items()}
                lv = None
            elif t == 0 or st["comms"] is None:
                comms = {(k[0], ("c", k[1])): m for k, m in fresh.items()}
                lv = fresh_lv
            elif name == "hier_frontier":                             # V16/W1: hierarchical frontier maintainer (ours)
                sys.path.insert(0, str(HERE / "v16"))                   # revision/v16 copied into $GCM_RESULTS/v16
                import hier_frontier as hf
                ladj = defaultdict(set)
                for a_, b_ in lcc:
                    ladj[a_].add(b_)
                    ladj[b_].add(a_)
                fr = set(new_nodes) | {v for e_ in new_el for v in e_}
                fr |= {u for v in list(fr) for u in ladj.get(v, ())}

                def _hfn(sub_edges, members):
                    sub = complete_levels(hier(sub_edges, list(range(len(node_t))))[0]) if sub_edges else {0: {v: 0 for v in members}}
                    return communities(sub)
                raw, hstats = hf.update(st["comms"], lcc, keep, fr, MAXC, SEED % (2 ** 31), _hfn)
                cc = hf.complete(raw)
                val = hf.validate(cc, keep, MAXC)
                for kk_, vv_ in {**val, **hstats}.items():
                    st.setdefault("hf_val", defaultdict(list))[kk_].append(vv_)
                mp = match_ids(st["comms"], cc)
                comms = {(mp.get(k, (k[0], ("hf", t, k[1])))): m for k, m in cc.items()}
                lv = None
            elif name == "reindex":
                comms = {(k[0], ("r", t, k[1])): m for k, m in fresh.items()}
                lv = fresh_lv
            elif name == "reindex+match":
                mp = match_ids(st["comms"], {k: m for k, m in fresh.items()})
                comms = {(mp.get(k, (k[0], ("m", t, k[1])))): m for k, m in fresh.items()}
                lv = fresh_lv
            elif name == "warm":
                leaf = {}
                for (l, c), m in st["comms"].items():
                    for v in m:
                        if l >= leaf.get(v, (-1, None))[0]:
                            leaf[v] = (l, c)
                ids = {}
                start = {}
                for v, lc in leaf.items():
                    start[str(v)] = ids.setdefault(lc, len(ids))
                nxt = len(ids)
                for a, b in lcc:
                    for v in (a, b):
                        if str(v) not in start:
                            start[str(v)] = nxt
                            nxt += 1
                lv0, _ = hier(lcc, list(range(len(node_t))), start)
                w = communities(complete_levels(lv0))
                mp = match_ids(st["comms"], w)
                comms = {(mp.get(k, (k[0], ("w", t, k[1])))): m for k, m in w.items()}
                lv = None
            elif name == "graphrag_update":
                # GraphRAG clusters only the largest connected component of the delta (cluster_graph.use_lcc, default
                # True; graphrag/index/operations/cluster_graph.py line 69 in 3.2.0/3.3.0).  V12/B7: HIER_GU_LCC=0 = use_lcc off.
                if os.environ.get("HIER_GU_LCC", "1") == "0":
                    lccd = sorted({(min(a, b), max(a, b)) for a, b in delta_el if a != b})
                else:
                    lccd, _ = lcc_edges(delta_el, np.arange(len(node_t))) if delta_el else ([], set())
                dl = communities(complete_levels(hier(lccd, list(range(len(node_t))))[0])) if lccd else {}
                comms = dict(st["comms"])
                for k, m in dl.items():
                    comms[(k[0], ("d", t, k[1]))] = m      # appended with fresh ids; old untouched
            else:  # zep_attach
                comms = {k: set(m) for k, m in st["comms"].items()}
                leaf = {}
                for (l, c), m in comms.items():
                    for v in m:
                        if l >= leaf.get(v, (-1, None))[0]:
                            leaf[v] = (l, c)
                anc = defaultdict(set)
                for (l, c), m in comms.items():
                    for v in m:
                        anc[v].add((l, c))
                adj = defaultdict(list)
                for a, b in new_el:
                    adj[a].append(b)
                    adj[b].append(a)
                for v in sorted(new_nodes):
                    votes = Counter(leaf[u] for u in adj.get(v, []) if u in leaf)
                    if votes:
                        target = votes.most_common(1)[0][0]
                        for k in anc[next(u for u in adj[v] if u in leaf and leaf[u] == target)]:
                            comms[k].add(v)
                        leaf[v] = target
                comms = {k: frozenset(m) for k, m in comms.items()}
            st["_prev_comms"] = st["comms"] or comms
            st["comms"] = comms
            # quality: leaf partition and top level (level 0)
            leafp, topp = {}, {}
            for (l, c), m in comms.items():
                for v in m:
                    if l >= leafp.get(v, (-1, None))[0]:
                        leafp[v] = (l, c)
                    if l == 0:
                        topp[v] = c
            cl = sorted(leafp)
            st["Qleaf"].append(modularity(g_all, cl, {v: str(leafp[v]) for v in cl}))
            st["Qtop"].append(modularity(g_all, sorted(topp), {v: str(topp[v]) for v in topp}))
            if os.environ.get("HIER_W4"):                 # V16/W4: Q on the LCC of G_t, unclustered nodes as singletons
                kl = sorted(keep)
                st.setdefault("Qleaf_lcc", []).append(modularity(g_all, kl, {v: str(leafp.get(v, ("s", v))) for v in kl}))
                st.setdefault("Qtop_lcc", []).append(modularity(g_all, kl, {v: str(topp[v]) if v in topp else "s" + str(v) for v in kl}))
            st["nrep"].append({ms: sum(1 for m in comms.values() if len(m) >= ms) for ms in MINS})
            # coverage: share of current entities (and of current LCC entities) inside some reported community
            for ms in MINS:
                cov = set()
                for m in comms.values():
                    if len(m) >= ms:
                        cov |= m
                st.setdefault("cov", defaultdict(list))[ms].append(len(cov & set(nodes.tolist())) / len(nodes))
                st.setdefault("cov_lcc", defaultdict(list))[ms].append(len(cov & keep) / max(len(keep), 1))
                newn = set(np.where(node_t == t)[0].tolist())       # V13/T3: coverage of this update's new entities
                if t > 0 and newn:
                    st.setdefault("cov_new", defaultdict(list))[ms].append(len(cov & newn) / len(newn))
            if t > 0:
                for tau in TAUS:
                    store = st["store"][tau]
                    for ms in MINS:
                        cost = 0
                        costv = dict.fromkeys(TOKVARIANTS, 0)
                        for k, m in comms.items():
                            if len(m) < ms:
                                continue
                            old = store.get((ms, k))
                            rewrite = old is None or (tau > 0 and len(m & old) / len(m | old) < tau)
                            if os.environ.get("HIER_BREAKDOWN") and tau == 0.9 and ms == 1:   # V12/B2
                                bk = (int(k[0]), "<=10" if len(m) <= 10 else "11-50" if len(m) <= 50 else
                                      "51-200" if len(m) <= 200 else ">200")
                                rec = st.setdefault("bd", {}).setdefault(bk, [0.0, 0.0, 0, 0])
                                rec[1] += tok(len(m))
                                rec[3] += 1
                                if rewrite:
                                    rec[0] += tok(len(m))
                                    rec[2] += 1
                            if rewrite and ms == 1 and os.environ.get("HIER_EXACT"):
                                st.setdefault("_rew", set()).add(k)
                            if rewrite:
                                cost += tok(len(m))
                                for vn, v in TOKVARIANTS.items():
                                    costv[vn] += tokv(len(m), v)
                                store[(ms, k)] = m
                        for vn in TOKVARIANTS:
                            st.setdefault("costv", defaultdict(list))[(vn, tau, ms)].append(costv[vn] / max(full_tokv[(vn, ms)], 1))
                        st["cost"][(tau, ms)].append(cost / max(full_tok[ms], 1))
                        if os.environ.get("HIER_EXACT") and ms == 1:
                            rew = st.pop("_rew", set())
                            xc = sum(xtok(comms[k]) for k in rew)
                            st.setdefault("xcost", defaultdict(list))[tau].append(xc / xfull)
                            st.setdefault("xabs", defaultdict(list))[tau].append(xc)
                            # cascade: a kept community whose context exceeds the limit reads its children's reports
                            # (build_mixed_context); it must be rewritten if a child report was rewritten. Deepest level first.
                            stc = st.setdefault("store_c", {}).setdefault(tau, {})
                            if not stc:
                                stc.update({kk: mm for kk, mm in st["_prev_comms"].items()})
                            lv_of = defaultdict(dict)
                            for kk, mm in comms.items():
                                for v in mm:
                                    lv_of[kk[0]][v] = kk
                            rewc, ncasc, nexceed = set(), 0, 0
                            for kk in sorted(comms, key=lambda x: -x[0]):
                                mm = comms[kk]
                                old = stc.get(kk)
                                base_r = old is None or (tau > 0 and len(mm & old) / len(mm | old) < tau)
                                casc = False
                                if not base_r and xctx(mm) > int(XR["limit"]):
                                    nexceed += 1
                                    kids = {lv_of[kk[0] + 1].get(v) for v in mm} - {None, kk}
                                    casc = any(c in rewc for c in kids)
                                if base_r or casc:
                                    rewc.add(kk)
                                    stc[kk] = mm
                                    ncasc += int(casc)
                            for kk in [kk for kk in stc if kk not in comms]:
                                del stc[kk]
                            st.setdefault("xcost_c", defaultdict(list))[tau].append(sum(xtok(comms[kk]) for kk in rewc) / xfull)
                            st.setdefault("ncasc", defaultdict(list))[tau].append(ncasc)
                            st.setdefault("nexceed", defaultdict(list))[tau].append(nexceed)
                        if os.environ.get("HIER_W4") and ms == 1:     # V16/W4: content coverage
                            pres, cc = set(nodes.tolist()), set()
                            for k, m in comms.items():
                                w = store.get((1, k))
                                if w:
                                    cc |= (m & w)
                            st.setdefault("ccov", defaultdict(list))[tau].append(len(cc & pres) / len(pres))
                        own = sum(tok(len(m)) for m in comms.values() if len(m) >= ms)
                        st.setdefault("cost_own", defaultdict(list))[(tau, ms)].append(cost / max(own, 1))
                    for k in [k for k in store if k[1] not in comms]:
                        del store[k]
            else:
                for tau in TAUS:
                    for ms in MINS:
                        for k, m in comms.items():
                            if len(m) >= ms:
                                st["store"][tau][(ms, k)] = m
        print(f"seed {seed} version {t}/{T - 1}: fresh communities {len(fresh)}", flush=True)
    if (seed == 0 and not ONLY and not os.environ.get("HIER_OUT")) or os.environ.get("HIER_DUMP"):   # V17/M3: HIER_DUMP=path
        dump = {}
        for name, st in state.items():
            store = st["store"][0.9]
            dump[name] = [dict(level=int(k[1][0]), cur=sorted(int(x) for x in st["comms"][k[1]]),
                               written=sorted(int(x) for x in m)) for k, m in store.items() if k[0] == 1]
        (Path(os.environ["HIER_DUMP"]) if os.environ.get("HIER_DUMP") else HERE / (f"graphrag_hier_stores_{GRAPH}.json" if MAXC == 10 else f"graphrag_hier_stores_{GRAPH}_max{MAXC}.json")).write_text(json.dumps(dump))
    for name, st in state.items():
        res[name] = dict(Qleaf=float(np.nanmean(st["Qleaf"])), Qtop=float(np.nanmean(st["Qtop"])),
                         reports_final={str(ms): st["nrep"][-1][ms] for ms in MINS},
                         coverage={str(ms): float(np.mean(st["cov"][ms])) for ms in MINS},
                         coverage_final={str(ms): float(st["cov"][ms][-1]) for ms in MINS},
                         coverage_lcc={str(ms): float(np.mean(st["cov_lcc"][ms])) for ms in MINS},
                         cost_series={f"{tau}|{ms}": [float(x) for x in v] for (tau, ms), v in st["cost"].items()},
                         cost={f"{tau}|{ms}": float(np.mean(v)) for (tau, ms), v in st["cost"].items()},
                         cost_own={f"{tau}|{ms}": float(np.mean(v)) for (tau, ms), v in st.get("cost_own", {}).items()})
        top_ = [m for k, m in st["comms"].items() if k[0] == 0]     # V16/P3: final top level (count, modelled tokens)
        res[name]["top_final"], res[name]["top_tok_final"] = len(top_), float(sum(tok(len(m)) for m in top_))
        if "Qleaf_lcc" in st:
            res[name]["Qleaf_lcc"], res[name]["Qtop_lcc"] = float(np.nanmean(st["Qleaf_lcc"])), float(np.nanmean(st["Qtop_lcc"]))
        if "ccov" in st:
            res[name]["content_coverage"] = {str(tau): float(np.mean(v)) for tau, v in st["ccov"].items()}
            res[name]["content_coverage_final"] = {str(tau): float(v[-1]) for tau, v in st["ccov"].items()}
        if "xcost" in st:
            res[name]["exact_cost"] = {str(t_): float(np.mean(v)) for t_, v in st["xcost"].items()}
            res[name]["exact_abs_per_update"] = {str(t_): float(np.mean(v)) for t_, v in st["xabs"].items()}
            res[name]["exact_cost_cascade"] = {str(t_): float(np.mean(v)) for t_, v in st["xcost_c"].items()}
            res[name]["cascade_rewrites_per_update"] = {str(t_): float(np.mean(v)) for t_, v in st["ncasc"].items()}
            res[name]["kept_exceeding_per_update"] = {str(t_): float(np.mean(v)) for t_, v in st["nexceed"].items()}
        if "hf_val" in st:
            res[name]["validation"] = {k: [int(x) for x in v] for k, v in st["hf_val"].items()}
        if "cov_new" in st:
            res[name]["coverage_new"] = {str(ms): float(np.mean(st["cov_new"][ms])) for ms in MINS}
        if "bd" in st:
            res[name]["breakdown_tau09"] = {f"{lv}|{bk}": v for (lv, bk), v in st["bd"].items()}
        if TOKVARIANTS:
            res[name]["cost_variants"] = {f"{vn}|{tau}|{ms}": float(np.mean(v)) for (vn, tau, ms), v in st["costv"].items()}
    return res


def main():
    out = {}
    seeds = [int(x) for x in os.environ.get("HIER_SEEDS", "0,1,2").split(",")]
    for seed in seeds:
        out[str(seed)] = run(seed)
        (Path(os.environ["HIER_OUT"]) if os.environ.get("HIER_OUT") else HERE / (f"graphrag_hier_{GRAPH}_{'_'.join(ONLY)}{os.environ.get('HIER_SUFFIX', '')}.json" if ONLY else (("graphrag_hier.json" if GRAPH == "entity" else f"graphrag_hier_{GRAPH}.json") if MAXC == 10 else f"graphrag_hier_{GRAPH}_max{MAXC}.json"))).write_text(json.dumps(out, indent=1))
    print(json.dumps(out[str(seeds[0])], indent=1)[:3000])


if __name__ == "__main__":
    main()
