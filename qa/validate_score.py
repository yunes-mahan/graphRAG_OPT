"""Does the structural score predict answer quality?  And equivalence tests (TOST) instead of non-significance.

(1) Spearman rank correlation, across stores, between structural quantities and answer quality:
    QA stream (16 stores = 4 maintainers x 4 tolerances): maintainer modularity, update tokens, and the
    token-level score proxy (modularity - 0.31 * update tokens / tokens of the recompute tau=0.9 store) versus
    local key-point recall; entity graph (7 maintainer stores): token-level score versus global win rates.
(2) TOST equivalence with margins fixed in advance: local recall difference to fresh within +-0.05 (paired, per
    question); global win rate against fresh within [0.40, 0.60] (per question, averaged over both orders).
Writes validate_score.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
QA = HERE / "qa"
MARGIN_LOCAL, LO_G, HI_G = 0.05, 0.40, 0.60


def tost_paired(d, margin):
    d = np.asarray(d, float)
    n, m, s = len(d), d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n) if s > 0 else 1e-12
    p1 = stats.t.sf((m + margin) / se, n - 1)       # H0: mean <= -margin
    p2 = stats.t.cdf((m - margin) / se, n - 1)      # H0: mean >= +margin
    return float(max(p1, p2)), float(m), (float(m - stats.t.ppf(0.95, n - 1) * se), float(m + stats.t.ppf(0.95, n - 1) * se))


def tost_interval(x, lo, hi):
    x = np.asarray(x, float)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    se = s / np.sqrt(n) if s > 0 else 1e-12
    p1 = stats.t.sf((m - lo) / se, n - 1)
    p2 = stats.t.cdf((m - hi) / se, n - 1)
    return float(max(p1, p2)), float(m)


def main():
    out = {}
    plan = json.loads((QA / "plan.json").read_text())
    res = json.loads((QA / "results.json").read_text())
    per = pd.DataFrame(res["per_question"]).pivot_table(index="q", columns="store", values="recall")
    summ = {r["store"]: r for r in res["summary"]}
    ref = per["FRESH|recompute all"]
    rows = []
    base_tok = summ["recompute_leiden|tau=0.9"]["update_input_tokens"]
    for m in ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]:
        for tau in ("0.9", "0.7", "0.5", "0.0"):
            k = f"{m}|tau={tau}"
            tokn = summ[k]["update_input_tokens"] / base_tok
            p, diff, ci = tost_paired(per[k] - ref, MARGIN_LOCAL)
            rows.append(dict(store=k, Q=plan["methods"][m]["Q"], tokens=tokn, score=plan["methods"][m]["Q"] - 0.31 * tokn,
                             recall=float(per[k].mean()), diff=diff, ci90=ci, tost_p=p, equivalent=p < 0.05))
    df = pd.DataFrame(rows)
    corr = {c: stats.spearmanr(df[c], df.recall) for c in ("Q", "tokens", "score")}
    out["local"] = dict(stores=rows, spearman={c: dict(rho=float(v.statistic), p=float(v.pvalue)) for c, v in corr.items()},
                        n_equivalent=int(df.equivalent.sum()), n=len(df), margin=MARGIN_LOCAL, n_q=int(len(per)))
    print("local: Spearman vs recall:", {c: (round(v.statistic, 2), round(v.pvalue, 2)) for c, v in corr.items()})
    print(f"local TOST (+-{MARGIN_LOCAL}): {int(df.equivalent.sum())}/{len(df)} stores equivalent to fresh")
    # global (QA stream): qa_global rows
    g = pd.DataFrame(json.loads((HERE / "qa_global.json").read_text())["rows"])
    gq = g.groupby(["store", "q"])[["comprehensiveness", "diversity", "empowerment", "directness"]].mean().reset_index()
    grows = []
    for st in gq.store.unique():
        x = gq[gq.store == st]
        for c in ("comprehensiveness", "diversity", "empowerment", "directness"):
            p, m = tost_interval(x[c], LO_G, HI_G)
            grows.append(dict(store=st, criterion=c, mean=m, tost_p=p, equivalent=p < 0.05, n_q=len(x)))
    out["global_qa_stream"] = grows
    gdf = pd.DataFrame(grows)
    print("global (QA stream) TOST [0.4,0.6]:", gdf.groupby("store").equivalent.sum().to_dict())
    # entity graph: score vs global win rates; TOST
    E = json.loads((HERE / "entity_graph" / "qa_entity.json").read_text())
    ev = json.loads((HERE / "entity_graph" / "eval.json").read_text())
    er = pd.DataFrame(E["rows"]).groupby(["store", "q"])[["comprehensiveness", "diversity", "empowerment", "directness"]].mean().reset_index()
    erows = []
    for st in er.store.unique():
        x = er[er.store == st]
        sc = ev[f"{st}|0"]["score"]
        rec = dict(store=st, score=sc)
        for c in ("comprehensiveness", "diversity", "empowerment", "directness"):
            p, m = tost_interval(x[c], LO_G, HI_G)
            rec[c] = m
            rec[c + "_tost_p"] = p
        erows.append(rec)
    edf = pd.DataFrame(erows)
    ecorr = {c: stats.spearmanr(edf.score, edf[c]) for c in ("comprehensiveness", "diversity", "empowerment", "directness")}
    out["entity"] = dict(stores=erows, spearman={c: dict(rho=float(v.statistic), p=float(v.pvalue)) for c, v in ecorr.items()})
    print("entity graph: Spearman(score, win rate):", {c: (round(v.statistic, 2), round(v.pvalue, 2)) for c, v in ecorr.items()})
    print("entity TOST equivalent criteria per store:",
          {r["store"]: sum(r[c + "_tost_p"] < 0.05 for c in ("comprehensiveness", "diversity", "empowerment", "directness")) for r in erows})
    (HERE / "validate_score.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
