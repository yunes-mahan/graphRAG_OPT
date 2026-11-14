"""C2: open-model replication of the core non-inferiority comparison (maintained vs fresh reports, QA stream).

Fully local: Qwen3-14B (v2/qwen_backend.py) writes EVERY report (maintained and fresh), generates the questions,
answers them and marks key points; no gpt-5.6 output is used on either side.  The pipeline is the paper's own code
(analysis/qa_run.py: summarise, answer = TF-IDF top-6 reports, judge = key-point coverage; question generation with the
same prompt and the same seed-paper draws as the main experiment, qa_run.make_questions rng 7 -> 120 questions and
qa_more.more_questions rng 8 -> 180 questions), with qa_run.call replaced by the local model and every cache redirected
to separate files qa/cache_*_c2qwen.json.
Stores (as in qa_run.main): fresh reference (recomputed final partition) and the four maintainers at tau = 0.9, whose
final reports are written from the member sets at their last rewrite.
Statistics as in the paper: per-store mean key-point recall with bootstrap CI; paired difference to fresh, Wilcoxon p,
TOST with margin 0.05 (paired t, equivalent if p < 0.05; non-inferior if the lower 90% bound > -0.05).
Output: analysis/v2/c2_qwen_qa.json
usage: python v2/c2_qwen_qa.py [n_questions]
"""
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("OPENAI_API_KEY", "unused-local-only")
import qwen_backend as qb  # noqa: E402
import qa_run as qr  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_more as qm  # noqa: E402

SUFFIX = "_c2qwen"
_orig_init = qr.Cache.__init__
qr.Cache.__init__ = lambda self, name: _orig_init(self, name + SUFFIX)
qr.call = qb.call                     # every LLM call of qa_run / qa_more now goes to the local model
QA = HERE.parent / "qa"
STORES = ["FRESH|recompute all", "recompute_leiden|tau=0.9", "warmstart_leiden|tau=0.9", "frontier_leiden|tau=0.9",
          "hybrid|tau=0.9"]
WORKERS = int(os.environ.get("C2_WORKERS", "2"))


def tost(d, margin=0.05):
    d = np.asarray(d, float)
    n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    p = max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
    h = stats.t.ppf(0.95, n - 1) * se
    return float(p), float(m - h), float(m + h)


def main(nq):
    plan = json.loads((QA / "plan.json").read_text())
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    years = nodes.year.values
    cs, cq, cq2, ca, cj = qr.Cache("summaries"), qr.Cache("questions"), qr.Cache("questions_more"), qr.Cache("answers"), qr.Cache("judge")
    stores = {}
    for m in ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]:
        for tau, v in plan["methods"][m]["taus"].items():      # all tolerances (V12/C3 uses "never" = 0.0)
            stores[f"{m}|tau={tau}"] = v["final_store"]
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    node_t = np.load(QA / "node_t.npy")
    comm = {}
    for v, c in zip(np.where(node_t >= 0)[0], lab):
        comm.setdefault(int(c), []).append(int(v))
    stores["FRESH|recompute all"] = [sorted(m_) for m_ in comm.values() if len(m_) >= qp.MIN_SIZE]
    need = {qr.key_of(s): s for n in STORES for s in stores[n]}
    print(f"reports needed: {len(need)} (cached {sum(1 for k in need if cs.get(k))})", flush=True)
    with cf.ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda s: qr.summarise(s, docs, cs), need.values()))
    items = qr.make_questions(docs, years, cq)
    if nq > len(items):
        items = items + qm.more_questions(docs, years, {it["seed"] for it in items}, cq2)
    items = items[:nq]
    print(f"questions: {len(items)}", flush=True)

    def work(a):
        name, it = a
        sums = [cs.get(qr.key_of(s))["text"] for s in stores[name]]
        ans = qr.answer(it["question"], sums, ca, name)
        j = qr.judge(it, ans["text"], cj)
        return dict(store=name, q=it["question"], recall=float(np.mean(j["covered"])))

    with cf.ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(work, [(n, it) for it in items for n in STORES]))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="q", columns="store", values="recall")
    ref = piv["FRESH|recompute all"]
    summ = []
    for n in STORES:
        r = piv[n]
        boot = [np.mean(np.random.default_rng(b).choice(r.values, len(r))) for b in range(2000)]
        row = dict(store=n, recall=float(r.mean()), ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                   n_q=int(len(r)))
        if n != "FRESH|recompute all":
            d = r - ref
            p_t, lo, hi = tost(d)
            row.update(diff=float(d.mean()), wilcoxon_p=float(stats.wilcoxon(r, ref, zero_method="zsplit").pvalue) if (d != 0).any() else 1.0,
                       tost_p=p_t, ci90=[lo, hi], equivalent=p_t < 0.05, noninferior=lo > -0.05)
        summ.append(row)
        print(row, flush=True)
    usage = {}
    for c in (cs, cq, cq2, ca, cj):
        for v in c.data.values():
            u = v.get("usage") if isinstance(v, dict) else None
            if isinstance(u, dict):
                t = usage.setdefault(u["model"], [0, 0])
                t[0] += u["inp"]
                t[1] += u["out"]
    (HERE / "c2_qwen_qa.json").write_text(json.dumps(dict(settings=dict(qb.SETTINGS, ollama=qb.ollama_version()),
                                                          stores=STORES, n_questions=len(items), summary=summ,
                                                          per_question=rows, usage_tokens=usage), indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
