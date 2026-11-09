"""ID-free summary reuse: decouple community maintenance from summary maintenance.

Standard rule (baseline, 'id'): a community keeps its summary only if it keeps its ID and Jaccard(old, new) >= 0.9.
New rules ignore IDs. A library stores every summary ever written with its member set and topic fingerprint
(mean of mean-centred text-embedding-3-small embeddings of its members; centring with the mean of version-0 nodes,
so it is causal). At every version, each community with >= 5 members looks for a summary to reuse:
  cover:g   reuse the library summary with the largest coverage |C cap S| / |C| if it is >= g
            (a split child is covered by its parent's summary; a merge reuses the dominant part's summary)
  topic:th  reuse the library summary with the largest fingerprint cosine if it is >= th
Only if nothing qualifies is a new summary written (full: titles+abstracts; titles: up to 300 titles) and added.
A community keeps the summary it used last time while that one still qualifies (sticky), to avoid flip-flopping.
Evaluation: update-time summary tokens, number of written summaries, fidelity of the final store vs full summaries
of the current members (judge 1-5, substantive >= 3), local QA (120 q) and global QA (30 q, pairwise vs fresh).
usage: python summary_reuse.py plan   -> costs only (no LLM), python summary_reuse.py run -> everything
"""

import concurrent.futures as cf
import hashlib
import json
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

HERE = config.RESULTS
QA = HERE / "qa"
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]
RULES = ["id", "cover:0.7", "cover:0.5", "topic:0.98", "topic:0.95", "topic:0.90"]


def replay(labels, node_t, E, rule):
    lib, cur_use, events = [], {}, []           # lib entries: dict(members, cen)
    prev_assign = {}
    for t, lb in enumerate(labels):
        nd = np.where((node_t >= 0) & (node_t <= t))[0]
        comms = {}
        for v, c in zip(nd, lb):
            comms.setdefault(int(c), set()).add(int(v))
        assign = {}
        for cid, mem in comms.items():
            if len(mem) < qp.MIN_SIZE:
                continue
            mem_f = frozenset(mem)
            if rule == "id":
                j = prev_assign.get(cid)
                if j is not None and len(mem & lib[j]["members"]) / len(mem | lib[j]["members"]) >= 0.9:
                    assign[cid] = j
                    continue
            else:
                kind, thr = rule.split(":")
                thr = float(thr)
                cen = E[sorted(mem)].mean(0)
                cen /= max(np.linalg.norm(cen), 1e-12)

                def score(j):
                    if kind == "cover":
                        return len(mem & lib[j]["members"]) / len(mem)
                    return float(cen @ lib[j]["cen"])
                last = prev_assign.get(cid)
                if last is not None and score(last) >= thr:
                    assign[cid] = last
                    continue
                if lib:
                    if kind == "cover":
                        cand = {j for v in mem for j in cur_use.get(v, ())}
                        best = max(cand, key=score) if cand else None
                    else:
                        C = np.stack([e["cen"] for e in lib])
                        best = int(np.argmax(C @ cen))
                    if best is not None and score(best) >= thr:
                        assign[cid] = best
                        continue
            cen = E[sorted(mem)].mean(0)
            lib.append(dict(members=mem_f, cen=cen / max(np.linalg.norm(cen), 1e-12)))
            j = len(lib) - 1
            for v in mem:
                cur_use.setdefault(v, set()).add(j)
            events.append((t, sorted(mem)))
            assign[cid] = j
        prev_assign = assign
    final = {cid: j for cid, j in assign.items()}
    current = {cid: sorted(comms[cid]) for cid in final}
    return events, lib, final, current


def main(mode):
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    E = np.load(QA / "emb_te3small.npy").astype(np.float64)
    E = E - E[node_t == 0].mean(0)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    res, stores = [], {}
    for m in METHODS:
        labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
        for rule in RULES:
            ev, lib, final, current = replay(labels, node_t, E, rule)
            upd = [s for t, s in ev if t > 0]
            stores[(m, rule)] = dict(upd=upd, lib=lib, final=final, current=current)
            res.append(dict(method=m, rule=rule, written=len(upd), final_communities=len(final),
                            distinct_final_summaries=len(set(final.values()))))
    df = pd.DataFrame(res)
    print(df.to_string(index=False), flush=True)
    if mode != "run":
        return
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs, cv, cr = qr.Cache("summaries"), qr.Cache("variants"), qr.Cache("delta_rate")

    def summ(mem, inp):
        return qr.summarise(sorted(mem), docs, cs) if inp == "full" else \
            iv.summarise_variant(sorted(mem), docs, "titles<=300", cv)

    # stores to evaluate end to end (fixed a priori): every method x {id, cover:0.5, topic:0.95} with full input,
    # plus title-only versions of the topic rule
    evals = [(m, r, "full") for m in METHODS for r in ("id", "cover:0.5", "topic:0.95")] + \
            [(m, "topic:0.95", "titles") for m in METHODS]
    need = set()
    for m, r, inp in evals:
        st = stores[(m, r)]
        for s in st["upd"] + [sorted(st["lib"][j]["members"]) for j in st["final"].values()] + list(st["current"].values()):
            need.add((tuple(s), inp if s not in st["current"].values() else "full"))
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda x: summ(x[0], x[1]), need))
    out, store_texts = [], {}
    for m, r, inp in evals:
        st = stores[(m, r)]
        tok = sum(summ(s, inp)["usage"]["inp"] + summ(s, inp)["usage"]["out"] for s in st["upd"])
        texts_final = {cid: summ(sorted(st["lib"][j]["members"]), inp)["text"] for cid, j in st["final"].items()}
        name = f"{m}|{r}|{inp}"
        store_texts[name] = texts_final
        out.append(dict(store=name, method=m, rule=r, input=inp, written=len(st["upd"]), update_tokens=tok,
                        distinct_summaries=len(set(texts_final.values()))))
    # fidelity vs full summary of current members
    jobs = [(name, txt, qr.summarise(stores[(n.split("|")[0], n.split("|")[1])]["current"][cid], docs, cs)["text"])
            for name, tf in store_texts.items() for n in [name] for cid, txt in tf.items()]
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda j: ds.rate(j[1], j[2], cr)["change"], jobs))
    fid = pd.DataFrame([dict(store=j[0], change=r) for j, r in zip(jobs, rates)])
    fid = fid.groupby("store").change.agg(substantive=lambda c: float((c >= 3).mean())).reset_index()
    # local QA (dedupe identical texts)
    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")
    alias = {f"{m}|id|full": f"{m}|tau=0.9" for m in METHODS}   # identical to the existing tau=0.9 stores

    def work(a):
        name, it = a
        sums = sorted(set(store_texts[name].values()))
        ans = qr.answer(it["question"], sums, ca, alias.get(name, "reuse|" + name))
        return dict(store=name, q=it["question"], recall=float(np.mean(qr.judge(it, ans["text"], cj)["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(n, it) for n in store_texts for it in items]))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    piv = pd.concat([base, pd.DataFrame(rows)]).pivot_table(index="q", columns="store", values="recall")
    ref = piv["FRESH|recompute all"]
    # global QA vs fresh answers (pairwise, both orders)
    gq = qr.Cache("questions").get("global")["items"]
    cag, cjg = qr.Cache("answers_global"), qr.Cache("judge_global")
    fresh_ans = {i: cag.get(hashlib.sha1(("global|summaries|fresh|" + gq[i]).encode()).hexdigest())
                 for i in range(len(gq))}

    def gwork(a):
        name, i = a
        blocks = [f"Report {k + 1}:\n{t}" for k, t in enumerate(sorted(set(store_texts[name].values())))]
        an = qg.answer(gq[i], blocks, "reuse|" + name, cag)
        res_ = []
        for order in (0, 1):
            x, f = an["text"], fresh_ans[i]["text"]
            r = qg.compare(gq[i], *((x, f) if order == 0 else (f, x)), cjg)["v"]
            mine = 1 if order == 0 else 2
            res_.append({c: 1.0 if r[c] == mine else 0.5 if r[c] == 0 else 0.0 for c in qg.CRIT})
        return dict(store=name, q=i, **{c: np.mean([d[c] for d in res_]) for c in qg.CRIT})

    with cf.ThreadPoolExecutor(8) as ex:
        grows = list(ex.map(gwork, [(n, i) for n in store_texts for i in range(len(gq))]))
    g = pd.DataFrame(grows)
    O = pd.DataFrame(out).merge(fid, on="store")
    O["local_recall"] = [piv[n].mean() for n in O.store]
    O["local_p_vs_fresh"] = [stats.wilcoxon(piv[n], ref, zero_method="zsplit").pvalue if (piv[n] != ref).any()
                             else 1.0 for n in O.store]
    for c in ("comprehensiveness", "diversity"):
        O["global_" + c] = [g[g.store == n][c].mean() for n in O.store]
    O["global_min_p"] = [min(stats.wilcoxon(g[g.store == n][c] - 0.5).pvalue if (g[g.store == n][c] != 0.5).any()
                             else 1.0 for c in qg.CRIT) for n in O.store]
    base_tok = {m: O[(O.method == m) & (O.rule == "id")].update_tokens.iloc[0] for m in METHODS}
    O["token_share_vs_id"] = [r.update_tokens / base_tok[r.method] for r in O.itertuples()]
    ref_tok = O[(O.method == "recompute_leiden") & (O.rule == "id")].update_tokens.iloc[0]
    O["token_share_vs_recompute_id"] = O.update_tokens / ref_tok
    pd.set_option("display.width", 250)
    print(O.drop(columns=["store"]).round(3).to_string(index=False))
    (HERE / "summary_reuse.json").write_text(json.dumps(dict(costs=res, summary=O.to_dict("records"),
                                                             local=rows, global_=grows), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
