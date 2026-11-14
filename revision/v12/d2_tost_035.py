"""D2 robustness (V12): the +-0.05 recall margin was set after the first 120 results were known.  Re-run the paper's
paired-t TOST / non-inferiority tests with a stricter margin of +-0.035 on the same data, with the same pivots as
make_assets.py:
  more300 : 300 local questions (qa/results.json + qa/results_more.json), the 6 maintained stores vs fresh (gpt)
  own     : each maintainer's tau = 0.9 store vs fresh reports of its OWN final partition (qa/own_fresh.json), 300 questions
  c2      : open-model replication, 4 maintained stores at tau = 0.9 vs fresh (v2/c2_qwen_qa.json)
  c3      : open-model, warm-start / frontier-restricted never vs fresh, 300 questions (v12_qwen/c3_never_300.json)
Non-inferior: lower bound of the 90% interval > -margin; equivalent: TOST p < 0.05 (both bounds inside +-margin).
Output: analysis/v12/d2_tost_035.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent


def test(d, margin):
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p = max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return dict(n=n, diff=float(m), lo=float(m - h), hi=float(m + h), tost_p=float(p), equivalent=bool(p < 0.05),
                noninferior=bool(m - h > -margin))


old = pd.DataFrame(json.loads((A / "qa" / "results.json").read_text())["per_question"])
more = pd.DataFrame(json.loads((A / "qa" / "results_more.json").read_text())["per_question"])
own = pd.DataFrame(json.loads((A / "qa" / "own_fresh.json").read_text())["per_question"])
sets = {}
allq = pd.concat([old[old.store.isin(more.store.unique())], more]).pivot_table(index="q", columns="store", values="recall")
sets["more300"] = {s: (allq[s] - allq["FRESH|recompute all"]).dropna() for s in allq.columns
                   if s not in ("FRESH|recompute all", "papers|vector")}
ao = pd.concat([old, more, own]).pivot_table(index="q", columns="store", values="recall")
ao["OWNFRESH|recompute_leiden"] = ao["FRESH|recompute all"]
sets["own"] = {m: (ao[f"{m}|tau=0.9"] - ao[f"OWNFRESH|{m}"]).dropna()
               for m in ("recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid")}
for name, f in (("c2", A / "v2" / "c2_qwen_qa.json"), ("c3", A / "v12_qwen" / "c3_never_300.json")):
    pq = pd.DataFrame(json.loads(f.read_text())["per_question"]).pivot_table(index="q", columns="store", values="recall")
    sets[name] = {s: (pq[s] - pq["FRESH|recompute all"]).dropna() for s in pq.columns if s != "FRESH|recompute all"}
out = {}
for name, S in sets.items():
    out[name] = {}
    for margin in (0.05, 0.035):
        R = {s: test(d, margin) for s, d in S.items()}
        out[name][str(margin)] = dict(stores=R, n_stores=len(R), n_noninferior=sum(r["noninferior"] for r in R.values()),
                                      n_equivalent=sum(r["equivalent"] for r in R.values()),
                                      min_lower=min(r["lo"] for r in R.values()))
        print(name, margin, "noninferior", out[name][str(margin)]["n_noninferior"], "/", len(R),
              "equivalent", out[name][str(margin)]["n_equivalent"], "min lower %.3f" % out[name][str(margin)]["min_lower"])
(HERE / "d2_tost_035.json").write_text(json.dumps(out, indent=1))
