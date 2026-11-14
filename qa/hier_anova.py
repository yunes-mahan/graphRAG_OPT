"""Variance decomposition (main-effect eta^2) of the relative report cost in graphrag_hier*.json over the factors
staleness tolerance tau, maintainer, minimum report size and document order.
Two scopes per graph: all maintainers except the full re-index (whose cost is 1 by construction), and the two
re-clustering maintainers with identifier matching (re-index + matching, warm start).
Writes hier_anova.json.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS


def eta(sub):
    g = sub.cost.mean()
    tot = ((sub.cost - g) ** 2).sum()
    out = {f: float(sub.groupby(f).cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
           for f in ("tau", "maintainer", "min_size", "seed") if f in sub}
    out["residual(interactions)"] = 1 - sum(out.values())
    return out


res = {}
# VLDB revision (B6): HIER_ANOVA_SUFFIX="_5orders" reads v2/graphrag_hier_*_5orders.json and writes v2/hier_anova_5orders.json;
# without it the inputs and output are unchanged.
import os  # noqa: E402
SFX = os.environ.get("HIER_ANOVA_SUFFIX", "")
for graph, fn in (("entity", "graphrag_hier.json"), ("wikidata", "graphrag_hier_wikidata.json")):
    d = json.loads(((HERE / "v2" / f"graphrag_hier_{graph}{SFX}.json") if SFX else (HERE / fn)).read_text())
    rows = []
    for s, r in d.items():
        for m, v in r.items():
            if m == "reindex":
                continue
            for k, c in v["cost"].items():
                tau, ms = k.split("|")
                rows.append(dict(seed=s, maintainer=m, tau=tau, min_size=ms, cost=c))
    df = pd.DataFrame(rows)
    df["recluster"] = df.maintainer.isin(["reindex+match", "warm"])
    g = df.cost.mean()
    tot = ((df.cost - g) ** 2).sum()
    split = float(df.groupby("recluster").cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
    res[graph] = dict(all=eta(df), recluster_split=split,
                      reclustering=eta(df[df.maintainer.isin(["reindex+match", "warm"])]))
    print(graph, json.dumps(res[graph], indent=1))
((HERE / "v2" / f"hier_anova{SFX}.json") if SFX else (HERE / "hier_anova.json")).write_text(json.dumps(res, indent=1))
