"""W6 (V16): Leiden seed as a random factor in Table 5. Held-out groups (arXiv-15: orders 1-9, collab-12: 1-9,
arXiv-17: 0-2, collab-10: 0-2); score = Q - lambda C (lambda from kgv2_calibration.json); seed 0 = the stored results,
seeds 1-2 = v16/w6/. Per maintainer and group: mean and 95 % t-interval over seeds x orders; mean over groups;
ranking and best per group compared with seed 0 only. DF Leiden (authors' code) has seed 0 only.
Output: analysis/v16/w6_summary.json"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent
lam = json.loads((A / "kgv2_calibration.json").read_text())["lambda_"]
H = {}
for hf in sorted(list(A.glob("kgv2_heldout*.json")) + list((A / "v2").glob("kgv2_heldout*.json"))):
    if "agent" in hf.name:
        continue
    for k, v in json.loads(hf.read_text()).items():
        for m, x in v.items():
            if m != "_ref" and x:
                H.setdefault(k, {})[m] = x
MS = ["priced_warm", "priced_frontier", "warmstart_leiden", "frontier_leiden", "hybrid", "local_repair", "lazy",
      "recompute_leiden", "graphrag_delta", "df_leiden_sahu"]
GROUPS = {"arxiv15": ("arxiv|2015", range(1, 10)), "collab12": ("collab|2012", range(1, 10)),
          "arxiv17": ("arxiv|2017", range(3)), "collab10": ("collab|2010", range(3))}
sc = lambda r: r["Q"] - lam * r["C"]  # noqa: E731
out = {"lambda": lam, "groups": {}, "mean": {}, "mean_seed0": {}}
for g, (pre, orders) in GROUPS.items():
    out["groups"][g] = {}
    for m in MS:
        s0 = [sc(H[f"{pre}|{o}"][m]) for o in orders if m in H.get(f"{pre}|{o}", {})]
        more = []
        for seed in (1, 2):
            f = HERE / "w6" / f"seed{seed}_{m}.json"
            if f.exists():
                d = json.loads(f.read_text())
                more += [sc(d[f"{pre}|{o}"]) for o in orders if f"{pre}|{o}" in d]
        allv = s0 + more
        if not allv:
            continue
        ci = float(stats.t.ppf(0.975, len(allv) - 1) * np.std(allv, ddof=1) / np.sqrt(len(allv))) if len(allv) > 1 else None
        out["groups"][g][m] = dict(mean=float(np.mean(allv)), ci=ci, n=len(allv), mean_seed0=float(np.mean(s0)) if s0 else None)
for m in MS:
    if all(m in out["groups"][g] for g in GROUPS):
        out["mean"][m] = float(np.mean([out["groups"][g][m]["mean"] for g in GROUPS]))
        out["mean_seed0"][m] = float(np.mean([out["groups"][g][m]["mean_seed0"] for g in GROUPS]))
rank = lambda d: sorted(d, key=lambda m: -d[m])  # noqa: E731
out["rank_all"], out["rank_seed0"] = rank(out["mean"]), rank(out["mean_seed0"])
out["best_all"] = {g: max(out["groups"][g], key=lambda m: out["groups"][g][m]["mean"]) for g in GROUPS}
out["best_seed0"] = {g: max(out["groups"][g], key=lambda m: out["groups"][g][m]["mean_seed0"]) for g in GROUPS}
out["complete"] = all(out["groups"][g][m]["n"] == (len(GROUPS[g][1]) * (1 if m == "df_leiden_sahu" else 3)) for g in GROUPS for m in out["groups"][g])
print(json.dumps({k: out[k] for k in ("rank_all", "rank_seed0", "best_all", "best_seed0", "complete")}, indent=1))
(HERE / "w6_summary.json").write_text(json.dumps(out, indent=1))
