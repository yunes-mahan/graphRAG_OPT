"""B3 (Table 2 part): cost-model sensitivity of GraphRAG's hierarchical clustering under maintenance.
The 5-order runs (v2/hier5/, graphrag_hier.py with HIER_TOKVARIANTS=1) record the cost of the SAME rewrite decisions
under each cost variant (constants -50%/+50%, one at a time).  For each variant and graph:
  * ranking of the 4 non-trivial maintainers by mean cost at each tau (granularity s=1, as in Table 2) and its Kendall
    correlation with the base ranking;
  * the main-effect eta^2 decomposition of hier_anova.py (re-cluster split; tau within re-clustering).
Output: v2/b3_hier_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
VARS = ["base", "a-50", "a+50", "b-50", "b+50", "c-50", "c+50"]
TAUS = ["0.9", "0.7", "0.5", "0.0"]


def eta(sub):
    g = sub.cost.mean()
    tot = ((sub.cost - g) ** 2).sum()
    return {f: float(sub.groupby(f).cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
            for f in ("tau", "maintainer", "min_size", "seed")}


res = {}
for graph in ("entity", "wikidata"):
    D = json.loads((HERE / f"graphrag_hier_{graph}_5orders.json").read_text())
    out = {}
    for v in VARS:
        rows = []
        for s, r in D.items():
            for m, x in r.items():
                if m == "reindex":
                    continue
                src = x["cost"] if v == "base" else {k.split("|", 1)[1]: c for k, c in x["cost_variants"].items() if k.startswith(v + "|")}
                for k, c in src.items():
                    tau, ms = k.split("|")
                    rows.append(dict(seed=s, maintainer=m, tau=tau, min_size=ms, cost=c))
        df = pd.DataFrame(rows)
        df["recluster"] = df.maintainer.isin(["reindex+match", "warm"])
        g = df.cost.mean()
        tot = ((df.cost - g) ** 2).sum()
        split = float(df.groupby("recluster").cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
        within = eta(df[df.recluster])
        means = {t: df[(df.tau == t) & (df.min_size == "1")].groupby("maintainer").cost.mean().to_dict() for t in TAUS}
        out[v] = dict(split=split, tau_within=within["tau"], maintainer_within=within["maintainer"], cost_means=means)
    for v in VARS:
        out[v]["kendall_vs_base"] = {t: float(stats.kendalltau([out["base"]["cost_means"][t][m] for m in sorted(out["base"]["cost_means"][t])],
                                                               [out[v]["cost_means"][t][m] for m in sorted(out[v]["cost_means"][t])]).statistic)
                                     for t in TAUS}
        print(graph, v, "split %.3f tau_within %.3f" % (out[v]["split"], out[v]["tau_within"]),
              "kendall", {t: round(x, 2) for t, x in out[v]["kendall_vs_base"].items()})
    res[graph] = out
(HERE / "b3_hier_sensitivity.json").write_text(json.dumps(res, indent=1))
