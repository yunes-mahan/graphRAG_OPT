"""M2 (V17): main-effect eta^2 decomposition of v12/b3_eta_variants.py with our hierarchical frontier-restricted
maintainer (v16/w1, same 5 orders, tau and s levels) added:
  (a) a three-level split: global re-clustering {re-index+matching, warm} / partial {hier_frontier} / none {GraphRAG update, Zep}
  (b) the two-level split with hier_frontier counted as re-clustering
for the designs all / no_never / no_zep, plus tau within re-clustering for (b). Output: analysis/v17/m2_eta_hf.json"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent


def share(sub, f):
    g = sub.cost.mean()
    tot = ((sub.cost - g) ** 2).sum()
    return float(sub.groupby(f).cost.apply(lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot)


res = {}
for graph in ("entity", "wikidata"):
    d = json.loads((A / "v2" / f"graphrag_hier_{graph}_5orders.json").read_text())
    for s in range(5):
        w = json.loads((A / "v16" / "w1" / f"{graph}_s{s}.json").read_text())
        d[str(s)]["hier_frontier"] = w[str(s)]["hier_frontier"]
    rows = [dict(seed=s, maintainer=m, tau=k.split("|")[0], min_size=k.split("|")[1], cost=c)
            for s, r in d.items() for m, v in r.items() if m != "reindex" for k, c in v["cost"].items()]
    df = pd.DataFrame(rows)
    df["three"] = df.maintainer.map({"reindex+match": "global", "warm": "global", "hier_frontier": "partial",
                                     "graphrag_update": "none", "zep_attach": "none"})
    df["recl"] = df.three != "none"
    res[graph] = {}
    for name, sub in (("all", df), ("no_never", df[df.tau != "0.0"]), ("no_zep", df[df.maintainer != "zep_attach"])):
        res[graph][name] = dict(n=len(sub), split3=share(sub, "three"), split2_hf_as_recluster=share(sub, "recl"),
                                maintainer_all=share(sub, "maintainer"),
                                tau_within_recluster=share(sub[sub.recl], "tau"))
        print(graph, name, {k: round(v, 3) for k, v in res[graph][name].items()}, flush=True)
(HERE / "m2_eta_hf.json").write_text(json.dumps(res, indent=1))
