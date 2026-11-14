"""B4 (V12): total tokens per update on the LLM entity graph (GraphRAG hierarchy, tau = 0.9, every community reported):
    total(q) = report-update tokens + entity-extraction tokens + q x tokens read by one top-level global query
  report-update tokens : MODELLED = maintainer's per-update cost share (graphrag_hier, Table 2, 5 orders) x modelled
                         tokens of a full GraphRAG re-index of that version (tok(S) = 1500 + min(8000, 250|S|), every
                         community), averaged over the updates of order 0
  extraction tokens    : MEASURED, input + output tokens of the cached gpt-5.6-luna extraction calls of the documents
                         of each update (entity_graph/cache_extract.json), same for every maintainer
  query tokens         : MEASURED, report + key-point input tokens of one global query over the top-level reports of
                         the final store (entity_graph/hier_qa.json, map-reduce), per maintainer
Break-even query rates (queries per update at which two maintainers cost the same) are reported.
Output: analysis/v12/b4_tco.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata")]
import entity_graph_eval as eg  # noqa: E402
import graphrag_hier as gh  # noqa: E402
import stability_variants as sv  # noqa: E402

d, node_t, edge_t, T = eg.stream(0)
n, E = len(node_t), d["edges"]
full = []
for t in range(1, T):
    ok = (edge_t >= 0) & (edge_t <= t)
    el = [(int(a), int(b)) for a, b in E[:, ok].T]
    full.append(sum(gh.tok(len(m)) for m in sv.leiden(el, n).values()))
g = json.loads((eg.G / "graph.json").read_text())
cx = json.loads((eg.G / "cache_extract.json").read_text())
rng = np.random.default_rng(0)                       # same document order as eg.stream(0)
years = {dd: v["year"] for dd, v in g["docs"].items()}
doc_t, t = {}, 1
for dd, y in years.items():
    if y <= eg.BASE:
        doc_t[dd] = 0
for y in range(eg.BASE + 1, eg.LAST + 1):
    for part in np.array_split(rng.permutation(sorted(dd for dd, yy in years.items() if yy == y)), eg.BATCHES):
        for dd in part:
            doc_t[dd] = t
        t += 1
ext = [0.0] * T
for dd, tt in doc_t.items():
    u = cx[hashlib.sha1(f"extract|{dd}".encode()).hexdigest()]["usage"]
    ext[tt] += u["inp"] + u["out"]
ext_upd = float(np.mean(ext[1:]))
H5 = json.loads((A / "v2" / "graphrag_hier_entity_5orders.json").read_text())
share = {m: float(np.mean([H5[s][m]["cost"]["0.9|1"] for s in H5])) for m in ("reindex", "reindex+match", "warm", "graphrag_update", "zep_attach")}
full_upd = float(np.mean(full))
Hq = {r["store"]: r for r in json.loads((eg.G / "hier_qa.json").read_text())["summary"]}
qtok = {"reindex": Hq["fresh"]["query_tokens"], "reindex+match": Hq["fresh"]["query_tokens"], "warm": Hq["warm"]["query_tokens"],
        "graphrag_update": Hq["graphrag_update"]["query_tokens"], "zep_attach": Hq["zep_attach"]["query_tokens"]}
out = dict(full_reindex_tokens_per_update_modelled=full_upd, extraction_tokens_per_update_measured=ext_upd,
           extraction_tokens_total_measured=float(sum(ext)), maintainers={})
for m in share:
    rep = share[m] * full_upd
    out["maintainers"][m] = dict(report_tokens_per_update_modelled=rep, query_tokens_measured=float(qtok[m]),
                                 total={str(q): rep + ext_upd + q * qtok[m] for q in (0, 1, 10, 100, 1000)})
be = {}
for a in out["maintainers"]:
    for b in out["maintainers"]:
        ra, rb = out["maintainers"][a], out["maintainers"][b]
        dq = ra["query_tokens_measured"] - rb["query_tokens_measured"]
        dr = rb["report_tokens_per_update_modelled"] - ra["report_tokens_per_update_modelled"]
        if a < b and dq != 0 and dr / dq > 0:
            be[f"{a}|{b}"] = dr / dq
out["break_even_queries_per_update"] = be
print(json.dumps({k: v for k, v in out.items()}, indent=1))
(HERE / "b4_tco.json").write_text(json.dumps(out, indent=1))
