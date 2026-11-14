"""B2 (V12) summary: where the rewrite cost of re-clustering sits, and how much it depends on the matching.
Inputs: v12/b2/{entity,wikidata}_{greedy,hungarian}.json (graphrag_hier.py, HIER_BREAKDOWN=1, maintainers re-index +
matching and warm start, 3 orders).  breakdown_tau09["level|bucket"] = [rewritten tokens, all tokens, rewritten
reports, all reports] summed over the updates (tau = 0.9, every community reported).
Reports, per maintainer and graph: share of rewritten reports and tokens per hierarchy level and per size bucket
(<=10, 11-50, 51-200, >200 members), and the per-update cost at tau = 0.9 with greedy vs Hungarian matching.
Output: v12/b2_summary.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
out = {}
for g in ("entity", "wikidata"):
    out[g] = {}
    G = json.loads((HERE / "b2" / f"{g}_greedy.json").read_text())
    Hn = json.loads((HERE / "b2" / f"{g}_hungarian.json").read_text())
    for m in ("reindex+match", "warm"):
        lv, bk = defaultdict(lambda: np.zeros(4)), defaultdict(lambda: np.zeros(4))
        for s in G:
            for key, v in G[s][m]["breakdown_tau09"].items():
                level, bucket = key.split("|")
                lv[int(level)] += np.array(v, float)
                bk[bucket] += np.array(v, float)
        rec = dict(
            by_level={str(k): dict(token_share=float(v[0] / v[1]), report_share=float(v[2] / v[3]), reports=int(v[3]))
                      for k, v in sorted(lv.items())},
            by_size={k: dict(token_share=float(v[0] / v[1]), report_share=float(v[2] / v[3]), reports=int(v[3]))
                     for k, v in bk.items()},
            cost_greedy=float(np.mean([G[s][m]["cost"]["0.9|1"] for s in G])),
            cost_hungarian=float(np.mean([Hn[s][m]["cost"]["0.9|1"] for s in Hn])))
        rec["hungarian_minus_greedy"] = rec["cost_hungarian"] - rec["cost_greedy"]
        out[g][m] = rec
        print(g, m, json.dumps({k: rec[k] for k in ("cost_greedy", "cost_hungarian")}),
              {k: round(v["token_share"], 3) for k, v in rec["by_level"].items()},
              {k: round(v["token_share"], 3) for k, v in rec["by_size"].items()}, flush=True)
(HERE / "b2_summary.json").write_text(json.dumps(out, indent=1))
