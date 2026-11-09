"""Cheaper global search: route each global question to the k communities whose prototype (mean
text-embedding-3-small of members) is closest to the question, and read only their summaries (k = 6, 12)
instead of all summaries. Compared pairwise with the all-summaries FRESH answer exactly as in qa_global.py.
A by-product is the share of summaries that a lazy index would ever have to write for these questions.
Writes global_routing.json.
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import qa_global as qg  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
KS = [6, 12]


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    cs = qr.Cache("summaries")
    cq, ca, cj = qr.Cache("questions"), qr.Cache("answers_global"), qr.Cache("judge_global")
    items = cq.get("global")["items"]
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    present = np.where(node_t >= 0)[0]
    comm = {}
    for v, c in zip(present, lab):
        comm.setdefault(int(c), []).append(int(v))
    sets = {"fresh": [sorted(m) for m in comm.values() if len(m) >= qp.MIN_SIZE],
            "hybrid tau=0.9": [sorted(s) for s in plan["methods"]["hybrid"]["taus"]["0.9"]["final_store"]]}
    E = np.load(QA / "emb_te3small.npy")
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    Qe = np.load(QA / "emb_global_questions.npy")
    Qe /= np.linalg.norm(Qe, axis=1, keepdims=True)
    jobs, used = [], {}
    for sname, ss in sets.items():
        P = np.stack([E[m].mean(0) for m in ss])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        for k in KS:
            name = f"summaries|{sname} proto-top{k}"
            used[name] = set()
            for i in range(len(items)):
                top = list(np.argsort(-(P @ Qe[i]))[:k])
                used[name].update(top)
                blocks = [f"Report {j + 1}:\n{cs.get(qr.key_of(ss[c]))['text']}" for j, c in enumerate(top)]
                jobs.append((name, i, blocks, len(ss)))
    with cf.ThreadPoolExecutor(8) as ex:
        ans = dict(zip([(n, i) for n, i, _, _ in jobs],
                       ex.map(lambda j: qg.answer(items[j[1]], j[2], j[0], ca), jobs)))
    fresh_blocks = None  # FRESH all-summaries answers are cached by qa_global.py
    ref = {i: ca.get(__import__("hashlib").sha1(("global|summaries|fresh|" + items[i]).encode()).hexdigest())
           for i in range(len(items))}
    assert all(ref.values()), "run qa_global.py first"
    del fresh_blocks

    def do(j):
        n, i, order = j
        a, f = ans[(n, i)]["text"], ref[i]["text"]
        r = qg.compare(items[i], *((a, f) if order == 0 else (f, a)), cj)["v"]
        mine = 1 if order == 0 else 2
        return dict(store=n, q=i, **{c: 1.0 if r[c] == mine else 0.5 if r[c] == 0 else 0.0 for c in qg.CRIT})

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(do, [(n, i, o) for n in used for i in range(len(items)) for o in (0, 1)]))
    df = pd.DataFrame(rows).groupby(["store", "q"])[qg.CRIT].mean().reset_index()
    out = []
    for n in used:
        d = df[df.store == n]
        rec = dict(store=n, input_tokens=float(np.mean([ans[(n, i)]["usage"]["inp"] for i in range(len(items))])),
                   ref_input_tokens=float(np.mean([r["usage"]["inp"] for r in ref.values()])),
                   summaries_used=len(used[n]), summaries_total=[j[3] for j in jobs if j[0] == n][0])
        for c in qg.CRIT:
            rec[c] = d[c].mean()
            rec[c + "_p"] = stats.wilcoxon(d[c] - 0.5).pvalue if (d[c] != 0.5).any() else 1.0
        out.append(rec)
    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    (HERE / "global_routing.json").write_text(json.dumps(dict(summary=out, rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()
