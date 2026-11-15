"""M3b (V17): Section 9's deletion-leak test (deletion_leak.py: same document draws, removal rules, leak definition,
lineage costs) for our hierarchical frontier-restricted maintainer.  Its deletion path (ours): deleted entities are
removed from the previous communities, the frontier is every surviving entity adjacent to a removed entity or
relationship (and their neighbours), hier_frontier.update re-optimises only there, identifiers are matched as for the
other maintainers, and reports are rewritten by the Jaccard rule (tau = 0.9); unmatched reports are retired.
Check: re-index + matching is recomputed here and must equal deletion_leak.json.
Inputs: v17/hf_stores_entity.json (graphrag_hier.py HIER_DUMP, order 0). Output: analysis/v17/m3_deletion_hf.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "v16")]
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import hier_frontier as hf  # noqa: E402

G = A / "entity_graph"


def leaks_and_cost(prev, written, new, mp, dead_e, dead_adj, touched_e, full_tok):
    inv = {v: k for k, v in mp.items()}
    kept, cost = [], 0
    for k, w in written.items():
        nk = inv.get(k)
        if nk is None:
            continue
        m = new[nk]
        if len(m & w) / len(m | w) >= 0.9:
            kept.append(k)
        else:
            cost += gh.tok(len(m))
    cost += sum(gh.tok(len(m)) for k, m in new.items() if k not in mp)
    ent = [k for k in kept if written[k] & dead_e]
    rel = [k for k in kept if not (written[k] & dead_e) and any(u in written[k] for v in written[k] for u in dead_adj.get(v, ()))]
    lk = ent + rel
    return dict(leak_share=len(lk) / max(len(kept), 1), cost=cost / full_tok,
                lineage_cost=sum(gh.tok(len(written[k])) for k in lk) / full_tok,
                doc_lineage_cost=sum(gh.tok(len(written[k])) for k in kept if written[k] & touched_e) / full_tok)


def main():
    g = json.loads((G / "graph.json").read_text())
    stores = {"hier_frontier": json.loads((HERE / "hf_stores_entity.json").read_text())["hier_frontier"],
              "reindex+match": json.loads((A / "graphrag_hier_stores_entity.json").read_text())["reindex+match"]}
    d, node_t, edge_t, T = eg.stream(0)
    n, E = len(node_t), d["edges"]
    ent_docs, rel_docs = {}, {}
    for did, v in g["docs"].items():
        for e in v["ents"]:
            ent_docs.setdefault(e, set()).add(did)
        for r in v["rels"]:
            rel_docs.setdefault(r, set()).add(did)
    docs = sorted(g["docs"])
    out = {}
    for p in (0.01, 0.05):
        res = {}
        for draw in range(3):
            rng = np.random.default_rng(draw)
            D = set(rng.choice(docs, int(round(p * len(docs))), replace=False).tolist())
            dead_e = {e for e, ds in ent_docs.items() if ds <= D}
            dead_r = {r for r, ds in rel_docs.items() if ds <= D}
            dead_adj = {}
            for j in dead_r:
                a, b = int(E[0, j]), int(E[1, j])
                if edge_t[j] < T and a not in dead_e and b not in dead_e:
                    dead_adj.setdefault(a, set()).add(b)
                    dead_adj.setdefault(b, set()).add(a)
            touched_e = {e for e, ds in ent_docs.items() if ds & D}
            el = [(int(E[0, j]), int(E[1, j])) for j in range(E.shape[1])
                  if edge_t[j] < T and j not in dead_r and int(E[0, j]) not in dead_e and int(E[1, j]) not in dead_e]
            lcc, keep = gh.lcc_edges(el, np.arange(n))
            fresh = gh.communities(gh.complete_levels(gh.hier(lcc, list(range(n)))[0]))
            full_tok = sum(gh.tok(len(m)) for m in fresh.values())
            adj = {}
            for a, b in lcc:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
            for name, store in stores.items():
                prev = {(x["level"], i): frozenset(x["cur"]) for i, x in enumerate(store)}
                written = {(x["level"], i): set(x["written"]) for i, x in enumerate(store)}
                if name == "reindex+match":
                    new = fresh
                else:
                    pclean = {k: m - dead_e for k, m in prev.items() if m - dead_e}
                    old_e = {(int(E[0, j]), int(E[1, j])) for j in range(E.shape[1]) if edge_t[j] < T}
                    gone = {v for e_ in old_e for v in e_ if v not in dead_e and (e_[0] in dead_e or e_[1] in dead_e)}
                    fr = (gone | set(dead_adj)) & keep
                    fr |= {u for v in list(fr) for u in adj.get(v, ())}

                    def hfn(sub_edges, members):
                        sub = gh.complete_levels(gh.hier(sub_edges, list(range(n)))[0]) if sub_edges else {0: {v: 0 for v in members}}
                        return gh.communities(sub)
                    raw, _ = hf.update(pclean, lcc, keep, fr, gh.MAXC, gh.SEED % (2 ** 31), hfn)
                    new = hf.complete(raw)
                mp = gh.match_ids(prev, new)
                r = leaks_and_cost(prev, written, new, mp, dead_e, dead_adj, touched_e, full_tok)
                for k, v in r.items():
                    res.setdefault(name, {}).setdefault(k, []).append(v)
            print(p, draw, {nm: {k: round(v[-1], 4) for k, v in x.items()} for nm, x in res.items()}, flush=True)
        out[str(p)] = {nm: {k: float(np.mean(v)) for k, v in x.items()} for nm, x in res.items()}
    old = json.loads((A / "deletion_leak.json").read_text())
    out["check_reindex_match"] = {p: dict(new=out[p]["reindex+match"]["leak_share"],
                                         stored=float(np.mean(old[p]["reindex+match"]["leak_share"]))
                                         if isinstance(old[p]["reindex+match"]["leak_share"], list) else old[p]["reindex+match"]["leak_share"])
                                  for p in ("0.01", "0.05") if p in old}
    print(out["check_reindex_match"])
    (HERE / "m3_deletion_hf.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
