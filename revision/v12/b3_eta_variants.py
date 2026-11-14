"""B3 (V12): main-effect eta^2 decomposition (same method as hier_anova.py) on the 5-order data
(v2/graphrag_hier_{entity,wikidata}_5orders.json), for the original factor levels and two restricted designs:
  all          : maintainers {re-index+matching, warm, GraphRAG update, Zep attach} x tau {0.9,0.7,0.5,never} x s {1,5,10}
  no_never     : tau level "never" (0.0) removed
  no_zep       : Zep-style attachment removed
Reports the re-cluster split (2-level factor), and within the two re-clustering maintainers (cold start with matching
vs warm start) the shares of tau, granularity and initialisation.  Output: analysis/v12/b3_eta_variants.json
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent


def eta(sub):
    g = sub.cost.mean()
    tot = ((sub.cost - g) ** 2).sum()
    out = {f: float(sub.groupby(f).cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
           for f in ("tau", "maintainer", "min_size", "seed")}
    out["residual(interactions)"] = 1 - sum(out.values())
    return out


res = {}
for graph in ("entity", "wikidata"):
    d = json.loads((A / "v2" / f"graphrag_hier_{graph}_5orders.json").read_text())
    rows = [dict(seed=s, maintainer=m, tau=k.split("|")[0], min_size=k.split("|")[1], cost=c)
            for s, r in d.items() for m, v in r.items() if m != "reindex" for k, c in v["cost"].items()]
    df = pd.DataFrame(rows)
    df["recluster"] = df.maintainer.isin(["reindex+match", "warm"])
    res[graph] = {}
    for name, sub in (("all", df), ("no_never", df[df.tau != "0.0"]), ("no_zep", df[df.maintainer != "zep_attach"])):
        g = sub.cost.mean()
        tot = ((sub.cost - g) ** 2).sum()
        split = float(sub.groupby("recluster").cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)
        within = eta(sub[sub.recluster])
        res[graph][name] = dict(n=len(sub), split=split, maintainer_all=eta(sub)["maintainer"], tau_all=eta(sub)["tau"],
                                tau_within=within["tau"], size_within=within["min_size"],
                                initialisation_within=within["maintainer"])
        print(graph, name, {k: round(v, 4) for k, v in res[graph][name].items()}, flush=True)
(HERE / "b3_eta_variants.json").write_text(json.dumps(res, indent=1))
