"""W2+W3 (V16) summary: entity graph, 5 orders (v16/w3/entity_s*.json; costs reproduce v2/graphrag_hier_entity_5orders.json).
  - rank agreement of modelled and GraphRAG-format (exact-input) relative costs over maintainers x tau
  - cascade (build_mixed_context) effect per maintainer at tau 0.9 and never
  - break-even query rates of b4_tco.py recomputed with GraphRAG-format report tokens per update (input counted,
    output = mean measured output of our reports) and the measured query tokens of b4_tco.json
Output: analysis/v16/w3_summary.json"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
A = HERE.parent
R = {}
for s in range(5):
    R.update(json.loads((HERE / "w3" / f"entity_s{s}.json").read_text()))
MS = ["reindex", "reindex+match", "warm", "graphrag_update", "zep_attach"]
T = ["0.9", "0.7", "0.5", "0.0"]
mean = lambda m, k, t: float(np.mean([R[s][m][k][t] for s in R]))  # noqa: E731
mod = [float(np.mean([R[s][m]["cost"][t + "|1"] for s in R])) for m in MS for t in T]
ex = [mean(m, "exact_cost", t) for m in MS for t in T]
out = dict(spearman=float(spearmanr(mod, ex)[0]), kendall=float(kendalltau(mod, ex)[0]), cells=len(mod),
           model={m: {t: mod[i * 4 + j] for j, t in enumerate(T)} for i, m in enumerate(MS)},
           exact={m: {t: mean(m, "exact_cost", t) for t in T} for m in MS},
           cascade={m: {t: mean(m, "exact_cost_cascade", t) for t in T} for m in MS},
           cascade_rewrites={m: {t: mean(m, "cascade_rewrites_per_update", t) for t in T} for m in MS},
           exact_abs={m: {t: mean(m, "exact_abs_per_update", t) for t in T} for m in MS})
out["cascade_max_increase"] = max(out["cascade"][m][t] - out["exact"][m][t] for m in MS for t in T)
tco = json.loads((A / "v12" / "b4_tco.json").read_text())["maintainers"]
q = {m: tco[m]["query_tokens_measured"] for m in MS}
rep = {m: out["exact_abs"][m]["0.9"] for m in MS}
be = {}
for a in MS:
    for b in MS:
        dq, dr = q[a] - q[b], rep[b] - rep[a]
        if a < b and dq != 0 and dr / dq > 0:
            be[f"{a}|{b}"] = dr / dq
out["break_even_exact"] = be
out["full_reindex_exact_per_update"] = out["exact_abs"]["reindex"]["0.9"]
print(json.dumps({k: v for k, v in out.items() if k in ("spearman", "kendall", "cascade_max_increase", "break_even_exact", "full_reindex_exact_per_update")}, indent=1))
(HERE / "w3_summary.json").write_text(json.dumps(out, indent=1))
