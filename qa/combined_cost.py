"""Combine the two cost levers: learned trigger (fewer regenerations) x title-only summaries (cheaper each).

For maintainers hybrid and recompute_leiden on the QA stream:
  trigger  'tau'      : regenerate iff the community is new or Jaccard(members now, members at last summary) < 0.9
           'learned'  : ... new, or (Jaccard < 0.9 AND classifier probability >= threshold). The classifier is the
                        structural logistic regression of staleness_learn.py (label: judge change rating >= 3),
                        trained ONLY on pairs of the other three maintainers (no leakage into the evaluated stream);
                        threshold = 10th percentile of training-positive probabilities (target 90% recall).
  input    'full'     : titles + abstracts (qa_run.summarise)   'titles' : up to 300 titles (input_variants)
Reported per store: update-time summary tokens, fidelity of the final store vs full summaries of the current
members (judge rating >= 3 share, as delta_summary.py), and QA key-point recall on all questions.
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import delta_summary as ds  # noqa: E402
import input_variants as iv  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
EVAL = ["hybrid", "recompute_leiden"]
FEATS = ["jac", "log_old", "log_new", "added", "removed"]


def feats(old, new):
    return [len(old & new) / len(old | new), np.log(len(old)), np.log(len(new)),
            len(new - old) / len(new), len(old - new) / len(old)]


def classifier_for(method):
    L = json.loads((HERE / "staleness_learn_major.json").read_text())["rows"]
    df = pd.DataFrame(L)
    tr = df[~df.group.str.startswith(method + ":")]
    y = (tr.change >= 3).astype(int).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[FEATS].values, y)
    p = clf.predict_proba(tr[FEATS].values)[:, 1]
    return clf, float(np.quantile(p[y == 1], 0.1)), len(tr)


def replay(labels, node_t, decide):
    store, events = {}, []
    for t, lb in enumerate(labels):
        nd = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nd, lb):
            cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < qp.MIN_SIZE:
                continue
            old = store.get(cid)
            if old is None or (len(mem & old) / len(mem | old) < 0.9 and decide(old, mem)):
                events.append((t, frozenset(mem)))
                store[cid] = frozenset(mem)
        for cid in list(store):
            if cid not in cur or len(cur[cid]) < qp.MIN_SIZE:
                del store[cid]
    return events, store, {c: frozenset(cur[c]) for c in store}


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs, cv, cr = qr.Cache("summaries"), qr.Cache("variants"), qr.Cache("delta_rate")

    def summ(mem, inp):
        mem = sorted(mem)
        return qr.summarise(mem, docs, cs) if inp == "full" else iv.summarise_variant(mem, docs, "titles<=300", cv)

    stores, info = {}, []
    for m in EVAL:
        labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
        clf, thr, ntr = classifier_for(m)
        rules = {"tau": lambda o, n: True,
                 "learned": lambda o, n, clf=clf, thr=thr: clf.predict_proba([feats(o, n)])[0, 1] >= thr}
        for rule, fn in rules.items():
            ev, final, current = replay(labels, node_t, fn)
            for inp in ("full", "titles"):
                need = [s for _, s in ev] + list(final.values())
                with cf.ThreadPoolExecutor(8) as ex:
                    list(ex.map(lambda s: summ(s, inp), need))
                upd = [s for t, s in ev if t > 0]
                tok = sum(summ(s, inp)["usage"]["inp"] + summ(s, inp)["usage"]["out"] for s in upd)
                name = f"{m}|{rule}|{inp}"
                stores[name] = dict(final=[sorted(final[c]) for c in final],
                                    current=[sorted(current[c]) for c in final], inp=inp)
                info.append(dict(store=name, method=m, rule=rule, input=inp, regenerations=len(upd),
                                 update_tokens=tok, train_pairs=ntr, threshold=thr))
                print(info[-1], flush=True)

    # fidelity vs full summaries of the CURRENT members
    jobs = [(n, summ(f, st["inp"])["text"], qr.summarise(c, docs, cs)["text"])
            for n, st in stores.items() for f, c in zip(st["final"], st["current"])]
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda j: ds.rate(j[1], j[2], cr)["change"], jobs))
    fid = pd.DataFrame([dict(store=j[0], change=r) for j, r in zip(jobs, rates)])
    fid = fid.groupby("store").change.agg(substantive=lambda c: (c >= 3).mean(), mean="mean").reset_index()

    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")

    def work(a):
        n, st, it = a
        sums = [summ(s, st["inp"])["text"] for s in st["final"]]
        ans = qr.answer(it["question"], sums, ca, n if not n.endswith("|tau|full") else n.split("|")[0] + "|tau=0.9")
        return dict(store=n, q=it["question"], recall=float(np.mean(qr.judge(it, ans["text"], cj)["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(n, st, it) for n, st in stores.items() for it in items]))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    piv = pd.concat([base, pd.DataFrame(rows)]).pivot_table(index="q", columns="store", values="recall")
    out = pd.DataFrame(info).merge(fid, on="store")
    ref = piv["FRESH|recompute all"]
    out["recall"] = [piv[n].mean() for n in out.store]
    out["diff_vs_fresh"] = [(piv[n] - ref).mean() for n in out.store]
    out["p_vs_fresh"] = [stats.wilcoxon(piv[n], ref, zero_method="zsplit").pvalue if (piv[n] != ref).any() else 1.0
                         for n in out.store]
    for m in EVAL:
        b = out[(out.method == m) & (out.rule == "tau") & (out.input == "full")].update_tokens.iloc[0]
        out.loc[out.method == m, "token_share"] = out[out.method == m].update_tokens / b
    print(out.drop(columns=["train_pairs"]).round(3).to_string(index=False))
    (HERE / "combined_cost.json").write_text(json.dumps(dict(summary=out.to_dict("records"), rows=rows), indent=1,
                                                        default=float))


if __name__ == "__main__":
    main()
