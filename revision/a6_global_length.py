"""A6: why does recompute + topic reuse beat fresh reports on global questions (Table 7, 0.73 / 0.69)?
Uses ONLY cached outputs (no LLM call): global answers of every reuse store and of the fresh reference
(qa/cache_answers_global.json) and the per-question pairwise judge outcomes stored in summary_reuse.json.
Measures answer length (words) relative to the fresh answer and its association with the win rate.
Output: analysis/v2/a6_global_length.json
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
QA = HERE.parent / "qa"
R = json.loads((HERE.parent / "summary_reuse.json").read_text())
gq = json.loads((QA / "cache_questions.json").read_text())["global"]["items"]
cag = json.loads((QA / "cache_answers_global.json").read_text())


def ans(store, q):
    return cag[hashlib.sha1(("global|" + store + "|" + q).encode()).hexdigest()]["text"]


words = lambda t: len(t.split())  # noqa: E731
fresh = [words(ans("summaries|fresh", q)) for q in gq]
G = R["global_"]
out, rows = {}, []
for s in R["summary"]:
    name = s["store"]
    w = [words(ans("reuse|" + name, q)) for q in gq]
    comp = {g["q"]: g["comprehensiveness"] for g in G if g["store"] == name}
    d = [w[i] - fresh[i] for i in range(len(gq))]
    longer = [i for i in range(len(gq)) if d[i] > 0]
    out[name] = dict(mean_words=float(np.mean(w)), fresh_mean_words=float(np.mean(fresh)),
                     mean_word_diff=float(np.mean(d)), share_longer=len(longer) / len(gq),
                     comp_win=float(np.mean(list(comp.values()))),
                     comp_win_when_longer=float(np.mean([comp[i] for i in longer])) if longer else None,
                     comp_win_when_not_longer=float(np.mean([comp[i] for i in range(len(gq)) if i not in longer]))
                     if len(longer) < len(gq) else None)
    rows += [(d[i], comp[i]) for i in range(len(gq))]
names = list(out)
rho_store = stats.spearmanr([out[n]["mean_word_diff"] for n in names], [out[n]["comp_win"] for n in names])
rho_q = stats.spearmanr([r[0] for r in rows], [r[1] for r in rows])
res = dict(stores=out, n_questions=len(gq),
           spearman_store_level=dict(rho=float(rho_store.statistic), p=float(rho_store.pvalue), n=len(names)),
           spearman_question_level=dict(rho=float(rho_q.statistic), p=float(rho_q.pvalue), n=len(rows)))
(HERE / "a6_global_length.json").write_text(json.dumps(res, indent=1))
for n in names:
    print(f"{n:42s} words {out[n]['mean_words']:6.1f} (fresh {out[n]['fresh_mean_words']:.1f}) diff {out[n]['mean_word_diff']:+6.1f} "
          f"longer {out[n]['share_longer']:.2f} comp {out[n]['comp_win']:.3f}")
print("store-level", res["spearman_store_level"], "question-level", res["spearman_question_level"])

# multiplicity: Wilcoxon (vs 0.5) for every store x criterion, Holm over all tests (same test as summary_reuse.py)
CRIT = ("comprehensiveness", "diversity", "empowerment", "directness")
tests = []
for n in names:
    for c in CRIT:
        v = np.array([g[c] for g in G if g["store"] == n])
        p = stats.wilcoxon(v - 0.5).pvalue if (v != 0.5).any() else 1.0
        tests.append((n, c, float(v.mean()), float(p)))
ps = np.array([t[3] for t in tests])
order = np.argsort(ps)
m = len(ps)
holm = np.empty(m)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * ps[i]))
    holm[i] = run
res["holm"] = [dict(store=t[0], criterion=t[1], win=t[2], p=t[3], p_holm=float(h)) for t, h in zip(tests, holm)]
res["n_tests"] = m
res["n_sig_uncorrected"] = int((ps < 0.05).sum())
res["n_sig_holm"] = int((holm < 0.05).sum())
res["mean_comp_win_all_stores"] = float(np.mean([out[n]["comp_win"] for n in names]))
res["n_stores_comp_above_half"] = int(sum(out[n]["comp_win"] > 0.5 for n in names))
(HERE / "a6_global_length.json").write_text(json.dumps(res, indent=1))
best = sorted(res["holm"], key=lambda r: r["p"])[:6]
print("tests", m, "sig uncorrected", res["n_sig_uncorrected"], "sig Holm", res["n_sig_holm"],
      "mean comp win", round(res["mean_comp_win_all_stores"], 3), "stores >0.5", res["n_stores_comp_above_half"])
for b in best:
    print(b)
