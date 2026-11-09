"""End-to-end test of the new maintenance pipeline on the QA stream.

Maintainers: priced rollback on warm-started Leiden (priced_warmstart.py), priced rollback on frontier Leiden
(priced_frontier.py), run on the same 11,309-paper stream and schedule as qa_plan.py.
Summary rules (summary_reuse.py): 'id' (Jaccard >= 0.9 with the same ID) and the ID-free topic rule chosen a
priori from summary_reuse.py; new summaries from full inputs or titles only.
Reported like summary_reuse.py: update tokens, fidelity vs full summaries of current members, local QA recall
(120 q) and global QA win rates vs fresh summaries (30 q, both orders).
usage: python qa_pipeline.py RULE   (e.g. topic:0.95)
"""

import concurrent.futures as cf
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import delta_summary as ds  # noqa: E402
import input_variants as iv  # noqa: E402
import qa_global as qg  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402
import summary_reuse as sr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
NEW = ["priced_warm", "priced_frontier"]


def labels_for(m):
    f = QA / f"labels_{m}.json"
    if f.exists():
        return [np.asarray(x) for x in json.loads(f.read_text())]
    d, _ = qp.build_subgraph()
    node_t, edge_t, T = qp.schedule(d)
    path, env = rs.SPECS[m]
    for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "PRICE_LAMBDA"):
        os.environ.pop(k, None)
    os.environ.update(env)
    labels, _ = se.replay(str(path), d, node_t, edge_t, T)
    sc = se.score(labels, d, node_t, edge_t)
    print(f"{m}: Q={sc['Q']:.3f} R={sc['R']:.3f}", flush=True)
    f.write_text(json.dumps([lab.tolist() for lab in labels]))
    return labels


def main(rule):
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    E = np.load(QA / "emb_te3small.npy").astype(np.float64)
    E = E - E[node_t == 0].mean(0)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs, cv, cr = qr.Cache("summaries"), qr.Cache("variants"), qr.Cache("delta_rate")

    def summ(mem, inp):
        return qr.summarise(sorted(mem), docs, cs) if inp == "full" else \
            iv.summarise_variant(sorted(mem), docs, "titles<=300", cv)

    stores, out = {}, []
    for m in NEW + ["warmstart_leiden", "recompute_leiden"]:
        labels = labels_for(m) if m in NEW else [np.asarray(x) for x in plan["methods"][m]["labels"]]
        for r, inp in (("id", "full"), (rule, "full"), (rule, "titles")):
            ev, lib, final, current = sr.replay(labels, node_t, E, r)
            upd = [s for t, s in ev if t > 0]
            name = f"{m}|{r}|{inp}"
            need = upd + [sorted(lib[j]["members"]) for j in final.values()]
            with cf.ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda s: summ(s, inp), need))
                list(ex.map(lambda s: summ(s, "full"), list(current.values())))
            tok = sum(summ(s, inp)["usage"]["inp"] + summ(s, inp)["usage"]["out"] for s in upd)
            stores[name] = dict(texts={cid: summ(sorted(lib[j]["members"]), inp)["text"] for cid, j in final.items()},
                                current=current)
            out.append(dict(store=name, method=m, rule=r, input=inp, written=len(upd), update_tokens=tok))
            print(out[-1], flush=True)
    jobs = [(n, txt, qr.summarise(st["current"][cid], docs, cs)["text"]) for n, st in stores.items()
            for cid, txt in st["texts"].items()]
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda j: ds.rate(j[1], j[2], cr)["change"], jobs))
    fid = pd.DataFrame([dict(store=j[0], change=r) for j, r in zip(jobs, rates)])
    fid = fid.groupby("store").change.agg(substantive=lambda c: float((c >= 3).mean())).reset_index()
    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")
    alias = {"warmstart_leiden|id|full": "warmstart_leiden|tau=0.9", "recompute_leiden|id|full": "recompute_leiden|tau=0.9"}

    def work(a):
        n, it = a
        sums = sorted(set(stores[n]["texts"].values()))
        ans = qr.answer(it["question"], sums, ca, alias.get(n, "reuse|" + n))
        return dict(store=n, q=it["question"], recall=float(np.mean(qr.judge(it, ans["text"], cj)["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(n, it) for n in stores for it in items]))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    piv = pd.concat([base, pd.DataFrame(rows)]).pivot_table(index="q", columns="store", values="recall")
    ref = piv["FRESH|recompute all"]
    gq = qr.Cache("questions").get("global")["items"]
    cag, cjg = qr.Cache("answers_global"), qr.Cache("judge_global")
    fresh = {i: cag.get(hashlib.sha1(("global|summaries|fresh|" + gq[i]).encode()).hexdigest()) for i in range(len(gq))}

    def gwork(a):
        n, i = a
        blocks = [f"Report {k + 1}:\n{t}" for k, t in enumerate(sorted(set(stores[n]["texts"].values())))]
        an = qg.answer(gq[i], blocks, "reuse|" + n, cag)
        res_ = []
        for order in (0, 1):
            x, f = an["text"], fresh[i]["text"]
            r = qg.compare(gq[i], *((x, f) if order == 0 else (f, x)), cjg)["v"]
            mine = 1 if order == 0 else 2
            res_.append({c: 1.0 if r[c] == mine else 0.5 if r[c] == 0 else 0.0 for c in qg.CRIT})
        return dict(store=n, q=i, **{c: np.mean([d[c] for d in res_]) for c in qg.CRIT})

    with cf.ThreadPoolExecutor(8) as ex:
        grows = list(ex.map(gwork, [(n, i) for n in stores for i in range(len(gq))]))
    g = pd.DataFrame(grows)
    O = pd.DataFrame(out).merge(fid, on="store")
    O["local_recall"] = [piv[n].mean() for n in O.store]
    O["local_p"] = [stats.wilcoxon(piv[n], ref, zero_method="zsplit").pvalue if (piv[n] != ref).any() else 1.0
                    for n in O.store]
    for c in qg.CRIT:
        O["g_" + c[:4]] = [g[g.store == n][c].mean() for n in O.store]
    O["g_min_p"] = [min(stats.wilcoxon(g[g.store == n][c] - 0.5).pvalue if (g[g.store == n][c] != 0.5).any()
                        else 1.0 for c in qg.CRIT) for n in O.store]
    ref_tok = O[O.store == "recompute_leiden|id|full"].update_tokens.iloc[0]
    O["tokens_vs_recompute"] = O.update_tokens / ref_tok
    pd.set_option("display.width", 250)
    print(O.drop(columns=["store"]).round(3).to_string(index=False))
    (HERE / "qa_pipeline.json").write_text(json.dumps(dict(rule=rule, summary=O.to_dict("records"), local=rows,
                                                           global_=grows), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
