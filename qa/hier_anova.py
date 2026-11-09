"""Variance decomposition of the relative report cost in graphrag_hier.json over the factors
tolerance tau, maintainer, minimum report size, and seed (main-effect sums of squares, eta^2).
Maintainers with an identifier rule only (reindex has cost 1 by construction and is excluded).
Writes hier_anova.json."""
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
d = json.loads((HERE / "graphrag_hier.json").read_text())
rows = []
for s, r in d.items():
    for m, v in r.items():
        if m == "reindex":
            continue
        for k, c in v["cost"].items():
            tau, ms = k.split("|")
            rows.append(dict(seed=s, maintainer=m, tau=tau, min_size=ms, cost=c))
df = pd.DataFrame(rows)
def eta(sub):
    g = sub.cost.mean(); tot = ((sub.cost - g) ** 2).sum()
    out = {f: float(sub.groupby(f).cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
           for f in ("tau", "maintainer", "min_size", "seed")}
    out["residual(interactions)"] = 1 - sum(out.values())
    return out
res = dict(all=eta(df), identifier_maintainers=eta(df[df.maintainer.isin(["reindex+match", "warm"])]),
           means=df.groupby(["maintainer", "tau"]).cost.mean().unstack().round(3).to_dict())
print(json.dumps(res, indent=1))
(HERE / "hier_anova.json").write_text(json.dumps(res, indent=1))
