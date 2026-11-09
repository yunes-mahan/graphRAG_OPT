"""Incremental ("delta") summary updates instead of full regeneration.

When a community must be regenerated (Jaccard rule, tau = 0.9), the delta updater gives the LLM only
  the community's CURRENT summary + titles/abstracts of ADDED papers + titles of REMOVED papers
and asks for a revised summary. Updates are chained over the stream (a delta summary is the input of the next
delta), so errors can accumulate as in a real deployment. Same model (gpt-5.6-luna) and length limit as the
full summaries in qa_run.py.

Evaluation
  fidelity : gpt-5.6-terra rates the final delta summary of every final community against a FULL summary of its
             CURRENT member set (what a freshly built index would hold) (1 = same content ... 5 = different community); reference points: the stale summary
             (never updated) vs full, and a second independent full summary vs full (LLM sampling noise floor).
  QA       : the 40 topic-level questions of qa_run.py answered from the delta store (key-point recall).
  cost     : input/output tokens of all update-time summary calls.
usage: python delta_summary.py
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
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]
TAU = 0.9
N_NOISE = 60


def replay(labels, node_t):
    """Jaccard tau=0.9 rule with community identity; yields (t, cid, old_members|None, new_members)."""
    store, out = {}, []
    for t, lb in enumerate(labels):
        nd = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nd, lb):
            cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < qp.MIN_SIZE:
                continue
            old = store.get(cid)
            if old is None or len(mem & old) / len(mem | old) < TAU:
                out.append((t, cid, old, frozenset(mem)))
                store[cid] = frozenset(mem)
        for cid in list(store):
            if cid not in cur or len(cur[cid]) < qp.MIN_SIZE:
                del store[cid]
    return out, store, {c: frozenset(cur[c]) for c in store}


def delta_update(old_text, added, removed, docs, cache):
    key = hashlib.sha1((old_text + "|" + ",".join(map(str, sorted(added))) + "|" +
                        ",".join(map(str, sorted(removed)))).encode()).hexdigest()
    hit = cache.get(key)
    if hit:
        return hit
    parts, used = [], 0
    for v in sorted(added):
        t, a = docs[v]
        s = f"- {t}: {a[:600]}"
        if used + len(s) > qr.MAX_CHARS:
            break
        parts.append(s)
        used += len(s)
    rem = "\n".join(f"- {docs[v][0]}" for v in sorted(removed)[:40]) or "(none)"
    prompt = ("You maintain community reports for a GraphRAG index over computer-science papers. Below is the "
              "current report of one community, followed by papers that JOINED and papers that LEFT the community. "
              "Revise the report so that it describes the community now: integrate new themes, methods and trends "
              "from the joined papers and drop content that only the departed papers supported. Keep it at most "
              "200 words with a short title, main research themes, typical methods, and recent trends. If the "
              "changes do not affect the report, return it unchanged.\n\n"
              f"Current report:\n{old_text}\n\nJoined papers ({len(added)}, {len(parts)} shown):\n" +
              ("\n".join(parts) or "(none)") + f"\n\nLeft papers ({len(removed)}):\n{rem}")
    text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage)
    cache.put(key, out)
    return out


def full_again(members, docs, cache):
    """An independent second full summary of the same member set (sampling-noise floor)."""
    k = "again|" + qr.key_of(members)
    hit = cache.get(k)
    if hit:
        return hit
    tmp = qr.Cache.__new__(qr.Cache)
    tmp.data, tmp.path = {}, QA / "_unused.json"
    tmp.put = lambda kk, v: None
    out = qr.summarise(members, docs, tmp)
    cache.put(k, out)
    return out


def rate(a, b, cache):
    key = hashlib.sha1((a + "||" + b).encode()).hexdigest()
    hit = cache.get(key)
    if hit:
        return hit
    prompt = ("Two short reports describe a research community for a question-answering system. Report B is a "
              "reference written from all current papers.\n\n"
              f"Report A:\n{a}\n\nReport B (reference):\n{b}\n\n"
              "How different is the CONTENT of A from B for a user asking topic-level questions (themes, methods, "
              "applications, trends)? Ignore wording, ordering and minor detail. Return JSON "
              '{"change": 1-5 (1 = same content, 2 = minor differences, 3 = a substantive topic missing or wrong, '
              '4 = largely different, 5 = different community)}.')
    text, usage = qr.call(qr.JUDGE_MODEL, [{"role": "user", "content": prompt}], json_mode=True, max_tokens=300)
    out = dict(change=int(json.loads(text)["change"]), usage=usage)
    cache.put(key, out)
    return out


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs, cd, cr = qr.Cache("summaries"), qr.Cache("delta"), qr.Cache("delta_rate")
    stores, cost, check = {}, {}, {}

    def run_method(m):
        labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
        ev, final, current = replay(labels, node_t)
        ref = plan["methods"][m]["taus"][str(TAU)]
        assert len([e for e in ev if e[0] > 0]) == len([e for e in ref["events"] if e[0] > 0]), m
        cur_text, stale_text, inp, outp, n_upd = {}, {}, 0, 0, 0
        for t, cid, old, new in ev:
            if old is None or t == 0:                     # new community: full summary (as in every store)
                s = qr.summarise(sorted(new), docs, cs)
                cur_text[cid] = s["text"]
                stale_text[cid] = s["text"]
                if t > 0:
                    inp, outp = inp + s["usage"]["inp"], outp + s["usage"]["out"]
                continue
            r = delta_update(cur_text[cid], new - old, old - new, docs, cd)
            cur_text[cid] = r["text"]
            inp, outp, n_upd = inp + r["usage"]["inp"], outp + r["usage"]["out"], n_upd + 1
        full_inp = sum(cs.get(qr.key_of(s))["usage"]["inp"] for t, s in ref["events"] if t > 0)
        stores[m] = {cid: cur_text[cid] for cid in final}
        cost[m] = dict(delta_input=inp, delta_output=outp, delta_calls=n_upd, full_input=full_inp)
        check[m] = [(cid, sorted(current[cid]), stale_text.get(cid)) for cid in final]   # reference = CURRENT members
        print(f"{m:17s} updates={n_upd:4d} input tokens delta={inp:,} vs full={full_inp:,} "
              f"({1 - inp / full_inp:.0%} saved)", flush=True)

    with cf.ThreadPoolExecutor(len(METHODS)) as ex:
        list(ex.map(run_method, METHODS))

    # fidelity: final delta summary vs full summary of the same member set; stale vs full; full-again vs full
    need = {qr.key_of(mem): mem for m in METHODS for _, mem, _ in check[m]}
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda mm: qr.summarise(mm, docs, cs), need.values()))
    rng0 = np.random.default_rng(0)
    pool0 = sorted({tuple(mem) for m in METHODS for _, mem, _ in check[m]})
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: full_again(list(pool0[i]), docs, cr),
                    rng0.choice(len(pool0), min(N_NOISE, len(pool0)), replace=False)))
    jobs = []
    for m in METHODS:
        for cid, mem, stale in check[m]:
            full = qr.summarise(mem, docs, cs)["text"]
            jobs.append((m, "delta", stores[m][cid], full))
            if stale is not None:
                jobs.append((m, "stale", stale, full))
    rng = np.random.default_rng(0)
    pool = sorted({tuple(mem) for m in METHODS for _, mem, _ in check[m]})
    for i in rng.choice(len(pool), min(N_NOISE, len(pool)), replace=False):
        mem = list(pool[i])
        jobs.append(("all", "full-again", full_again(mem, docs, cr)["text"], qr.summarise(mem, docs, cs)["text"]))
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda j: rate(j[2], j[3], cr)["change"], jobs))
    fid = pd.DataFrame([dict(method=j[0], kind=j[1], change=r) for j, r in zip(jobs, rates)])
    fid["substantive"] = fid.change >= 3
    print(fid.groupby(["kind"]).agg(n=("change", "size"), mean=("change", "mean"),
                                    substantive=("substantive", "mean")).round(3))
    print(fid.groupby(["method", "kind"]).agg(n=("change", "size"), mean=("change", "mean"),
                                              substantive=("substantive", "mean")).round(3))

    # QA on the delta stores
    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")

    def work(args):
        name, sums, it = args
        a = qr.answer(it["question"], sums, ca, name)
        return dict(store=name, q=it["question"], recall=float(np.mean(qr.judge(it, a["text"], cj)["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(f"{m}|tau={TAU}|delta", list(stores[m].values()), it)
                                  for m in METHODS for it in items]))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    piv = pd.concat([base, pd.DataFrame(rows)]).pivot_table(index="q", columns="store", values="recall")
    qa = []
    for m in METHODS:
        dname, fname = f"{m}|tau={TAU}|delta", f"{m}|tau={TAU}"
        for other in (fname, "FRESH|recompute all"):
            d = piv[dname] - piv[other]
            p = stats.wilcoxon(piv[dname], piv[other], zero_method="zsplit").pvalue if (d != 0).any() else 1.0
            qa.append(dict(store=dname, vs=other, recall=piv[dname].mean(), other=piv[other].mean(),
                           diff=d.mean(), p=p))
    qa = pd.DataFrame(qa)
    print(qa.round(3).to_string(index=False))
    noise_g = fid[fid.kind == "full-again"].change
    (HERE / "delta_summary.json").write_text(json.dumps(dict(
        cost=cost, fidelity=fid.to_dict("records"), qa=qa.to_dict("records"), qa_rows=rows,
        noise_mean=float(noise_g.mean())), indent=1, default=float))


if __name__ == "__main__":
    main()
