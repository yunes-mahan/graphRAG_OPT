"""Deletion leak test: after documents are retracted (or erased, e.g. under GDPR), which community reports still
derive from them?

Setting: LLM entity graph, document order 0, GraphRAG's hierarchical clustering, final report stores of
graphrag_hier.py (tau = 0.9, every community carries a report; each report remembers the member set it was written
for). A random share p of all documents (p in {0.01, 0.05}, 3 draws) is deleted. Entities and relationships
extracted ONLY from deleted documents are removed from the graph. A report "leaks" if it is still served after the
deletion and was written for a member set containing a removed entity, or both endpoints of a removed relationship
(reports are written from member entities and the relationships among them).
Maintenance after the deletion:
  reindex+match / warm : re-cluster the reduced graph (fresh / warm-started from the surviving leaves), match ids
                         level by level, rewrite reports by the Jaccard rule (tau = 0.9); unmatched reports retired
  graphrag_update      : no deletion path; all reports kept
  zep_attach           : no removal step; all reports kept
Lineage rules (any maintainer): entity/relationship lineage rewrites every kept report that leaks as defined above;
document-level lineage rewrites every kept report containing an entity mentioned in a deleted document (also covers
facts in surviving entities). Extra cost in report tokens relative to a full re-index.
Leaks are also checked in text for top-level reports: does the cached report text name a removed entity?
Writes deletion_leak.json.
"""

import json
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

G = HERE / "entity_graph"


def main():
    g = json.loads((G / "graph.json").read_text())
    dump = json.loads((HERE / "graphrag_hier_stores_entity.json").read_text())
    d, node_t, edge_t, T = eg.stream(0)
    n, E = len(node_t), d["edges"]
    ent_docs, rel_docs = {}, {}
    for did, v in g["docs"].items():
        for e in v["ents"]:
            ent_docs.setdefault(e, set()).add(did)
        for r in v["rels"]:
            rel_docs.setdefault(r, set()).add(did)
    hrep = json.loads((G / "cache_hreports.json").read_text())
    import hashlib
    names = g["entities"]
    docs = sorted(g["docs"])
    out = {}
    for p in (0.01, 0.05):
        res = {}
        for draw in range(3):
            rng = np.random.default_rng(draw)
            D = set(rng.choice(docs, int(round(p * len(docs))), replace=False).tolist())
            dead_e = {e for e, ds in ent_docs.items() if ds <= D}
            dead_r = {r for r, ds in rel_docs.items() if ds <= D}
            dead_adj = {}                                   # relationships removed although both entities survive
            for j in dead_r:
                a, b = int(E[0, j]), int(E[1, j])
                if edge_t[j] < T and a not in dead_e and b not in dead_e:
                    dead_adj.setdefault(a, set()).add(b)
                    dead_adj.setdefault(b, set()).add(a)
            touched_e = {e for e, ds in ent_docs.items() if ds & D}   # entities mentioned in a deleted document
            el = [(int(E[0, j]), int(E[1, j])) for j in range(E.shape[1])
                  if edge_t[j] < T and j not in dead_r and int(E[0, j]) not in dead_e and int(E[1, j]) not in dead_e]
            lcc, _ = gh.lcc_edges(el, np.arange(n))
            fresh = gh.communities(gh.complete_levels(gh.hier(lcc, list(range(n)))[0]))
            full_tok = sum(gh.tok(len(m)) for m in fresh.values())
            for name, store in dump.items():
                if name == "reindex":
                    continue
                prev = {(x["level"], i): frozenset(x["cur"]) for i, x in enumerate(store)}
                written = {(x["level"], i): set(x["written"]) for i, x in enumerate(store)}
                if name in ("reindex+match", "warm"):
                    if name == "warm":
                        leaf = {}
                        for (l, c), m in prev.items():
                            for v in m:
                                if v not in dead_e and l >= leaf.get(v, (-1, None))[0]:
                                    leaf[v] = (l, c)
                        ids, start = {}, {}
                        for v, lc in leaf.items():
                            start[str(v)] = ids.setdefault(lc, len(ids))
                        nxt = len(ids)
                        for a, b in lcc:
                            for v in (a, b):
                                if str(v) not in start:
                                    start[str(v)] = nxt
                                    nxt += 1
                        new = gh.communities(gh.complete_levels(gh.hier(lcc, list(range(n)), start)[0]))
                    else:
                        new = fresh
                    mp = gh.match_ids(prev, new)
                    inv = {v: k for k, v in mp.items()}
                    kept, cost = [], 0
                    for k, w in written.items():
                        nk = inv.get(k)
                        if nk is None:
                            continue                                  # retired
                        m = new[nk]
                        if len(m & w) / len(m | w) >= 0.9:
                            kept.append(k)
                        else:
                            cost += gh.tok(len(m))
                    cost += sum(gh.tok(len(m)) for k, m in new.items() if k not in mp)
                else:
                    kept, cost = list(written), 0
                ent_leaks = [k for k in kept if written[k] & dead_e]
                rel_leaks = [k for k in kept if not (written[k] & dead_e)
                             and any(u in written[k] for v in written[k] for u in dead_adj.get(v, ()))]
                leaks = ent_leaks + rel_leaks
                lin_cost = sum(gh.tok(len(written[k])) for k in leaks)
                doc_lin = [k for k in kept if written[k] & touched_e]
                doc_cost = sum(gh.tok(len(written[k])) for k in doc_lin)
                txt_leak = txt_n = 0
                for k in ent_leaks:
                    if k[0] != 0:
                        continue
                    h = hrep.get(hashlib.sha1(("hreport|" + ",".join(map(str, sorted(written[k])))).encode()).hexdigest())
                    if h:
                        txt_n += 1
                        low = h["text"].lower()
                        txt_leak += any(len(names[e]) > 3 and names[e] in low for e in written[k] & dead_e)
                r = res.setdefault(name, dict(kept=[], leaks=[], leak_share=[], ent_leak_share=[], rel_leak_share=[],
                                              cost=[], lineage_cost=[], doc_lineage_cost=[], doc_lineage_share=[],
                                              txt=[0, 0]))
                r["kept"].append(len(kept))
                r["leaks"].append(len(leaks))
                r["ent_leak_share"].append(len(ent_leaks) / max(len(kept), 1))
                r["rel_leak_share"].append(len(rel_leaks) / max(len(kept), 1))
                r["doc_lineage_cost"].append(doc_cost / full_tok)
                r["doc_lineage_share"].append(len(doc_lin) / max(len(kept), 1))
                r["leak_share"].append(len(leaks) / max(len(kept), 1))
                r["cost"].append(cost / full_tok)
                r["lineage_cost"].append(lin_cost / full_tok)
                r["txt"][0] += txt_leak
                r["txt"][1] += txt_n
            print(f"p={p} draw {draw}: {len(D)} documents, {len(dead_e)} entities, {len(dead_r)} relationships removed", flush=True)
        out[str(p)] = {k: {kk: (float(np.mean(vv)) if kk != "txt" else vv) for kk, vv in v.items()} for k, v in res.items()}
        for k, v in out[str(p)].items():
            print(f"  {k:16s} kept {v['kept']:8.0f} leaking {v['leaks']:7.0f} ({100 * v['leak_share']:.1f}%: entity "
                  f"{100 * v['ent_leak_share']:.1f}%, relationship only {100 * v['rel_leak_share']:.1f}%)  rewrite cost {v['cost']:.2f}"
                  f"  lineage fix +{v['lineage_cost']:.3f}  document-level lineage +{v['doc_lineage_cost']:.3f} "
                  f"({100 * v['doc_lineage_share']:.1f}% of reports)  text {v['txt'][0]}/{v['txt'][1]}", flush=True)
    (HERE / "deletion_leak.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
