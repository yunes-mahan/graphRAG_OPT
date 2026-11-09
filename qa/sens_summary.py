"""Sensitivity of the hierarchical results to GraphRAG's max_cluster_size (10 default; 25, 50).
Per graph and size: cost at tau=0.9/0.5 per maintainer, coverage, Q_top, variance split (re-cluster vs not; tau within
re-clustering) and instability (0.1% edge deletion, last update). Writes hier_sensitivity.json."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
out = {}
for size in (10, 25, 50):
    for graph in ("entity", "wikidata"):
        fn = (("graphrag_hier.json" if graph == "entity" else "graphrag_hier_wikidata.json") if size == 10
              else f"graphrag_hier_{graph}_max{size}.json")
        d = json.loads((HERE / fn).read_text())
        nfn = "noise_hier.json" if size == 10 else f"noise_hier_max{size}.json"
        nh = json.loads((HERE / nfn).read_text())[graph]
        rows = []
        for s, r in d.items():
            for m, v in r.items():
                if m == "reindex":
                    continue
                for k, c in v["cost"].items():
                    tau, ms = k.split("|")
                    rows.append(dict(seed=s, maintainer=m, tau=tau, min_size=ms, cost=c))
        df = pd.DataFrame(rows)
        g = df.cost.mean(); tot = ((df.cost - g) ** 2).sum()
        split = df.assign(rc=df.maintainer.isin(["reindex+match", "warm"])).groupby("rc").cost.apply(
            lambda x: len(x) * (x.mean() - g) ** 2).sum() / tot
        sub = df[df.maintainer.isin(["reindex+match", "warm"])]
        g2 = sub.cost.mean(); tot2 = ((sub.cost - g2) ** 2).sum()
        eta_tau = sub.groupby("tau").cost.apply(lambda x: len(x) * (x.mean() - g2) ** 2).sum() / tot2
        mean = lambda m, f: float(np.mean([f(d[s][m]) for s in d]))  # noqa: E731
        out[f"{graph}|{size}"] = dict(
            split=float(split), eta_tau=float(eta_tau),
            cost09={m: mean(m, lambda r: r["cost"]["0.9|1"]) for m in d["0"]},
            coverage={m: mean(m, lambda r: r["coverage_final"]["1"]) for m in d["0"]},
            qtop={m: mean(m, lambda r: r["Qtop"]) for m in d["0"]},
            drop=float(np.mean(nh["drop0.001"]["0.9"])), last=float(nh["last"]["0.9"]))
        o = out[f"{graph}|{size}"]
        print(f"{graph:8s} max{size:2d} split {o['split']:.2f} tau {o['eta_tau']:.2f} drop {o['drop']:.2f} last {o['last']:.2f} "
              f"cov GU {o['coverage']['graphrag_update']:.2f} Zep {o['coverage']['zep_attach']:.2f} re {o['coverage']['reindex']:.2f} "
              f"match0.9 {o['cost09']['reindex+match']:.2f}")
(HERE / "hier_sensitivity.json").write_text(json.dumps(out, indent=1))
