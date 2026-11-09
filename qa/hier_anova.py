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
           for f in ("tau", "maintainer", "min_size", "seed")}
    out["residual(interactions)"] = 1 - sum(out.values())
    return out


res = {}
for graph, fn in (("entity", "graphrag_hier.json"), ("wikidata", "graphrag_hier_wikidata.json")):
    d = json.loads((HERE / fn).read_text())
    rows = []
    for s, r in d.items():
        for m, v in r.items():
            if m == "reindex":
                continue
            for k, c in v["cost"].items():
                tau, ms = k.split("|")
                rows.append(dict(seed=s, maintainer=m, tau=tau, min_size=ms, cost=c))
    df = pd.DataFrame(rows)
    res[graph] = dict(all=eta(df), reclustering=eta(df[df.maintainer.isin(["reindex+match", "warm"])]))
    print(graph, json.dumps(res[graph], indent=1))
(HERE / "hier_anova.json").write_text(json.dumps(res, indent=1))
