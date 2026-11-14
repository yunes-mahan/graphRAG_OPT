"""C2 (V12): A/A test of the pairwise global judge - what "no difference" looks like.  QA stream, the 30 global
questions of the paper (shared input), fresh reference partition (recomputed final partition, as in the paper).
Everything else with local Qwen3-14B (qwen12.py); Qwen judges pairwise, both answer orders.

  report set A : Qwen fresh reports of V11/C2 (temperature 0; copied from qa/cache_summaries_c2qwen.json, read-only)
  report set B : an independent second sample of the same reports (temperature 0.7, seed 1)
  A/A answers  : two independent answers from report set A (temperature 0.7, seeds 1 and 2)      -> judge A1 vs A2
  A/A reports  : answer from set A vs answer from set B (both temperature 0)                       -> judge A0 vs B0
Answers: qa_global.answer over ALL reports (as the paper's global QA).  Judge: qa_global.compare, 4 criteria.
Reports win rate of the first-named side (0.5 = no difference), bootstrap 95% CI over questions, tie rate and
position bias (share of non-tie verdicts that pick the answer shown first).  Output: v12_qwen/c2_aa_global.json
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE)]
os.environ.setdefault("OPENAI_API_KEY", "unused-local-only")
import qwen12 as qb  # noqa: E402
import qa_run as qr  # noqa: E402
import qa_global as qg  # noqa: E402
import qa_plan as qp  # noqa: E402

qr.call = qb.call


class Cache(qr.Cache):
    def __init__(self, name):
        self.path = HERE / f"cache_{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}


def main():
    src = A / "qa" / "cache_summaries_c2qwen.json"
    if not (HERE / "cache_c2_reportsA.json").exists():
        shutil.copy(src, HERE / "cache_c2_reportsA.json")            # never write to the V11 cache
    plan = json.loads((A / "qa" / "plan.json").read_text())
    node_t = np.load(A / "qa" / "node_t.npy")
    nodes = pd.read_csv(A / "qa" / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    comm = {}
    for v, c in zip(np.where(node_t >= 0)[0], lab):
        comm.setdefault(int(c), []).append(int(v))
    fresh = [sorted(m) for m in comm.values() if len(m) >= qp.MIN_SIZE]
    cA, cB = Cache("c2_reportsA"), Cache("c2_reportsB")
    repA = [qr.summarise(m, docs, cA)["text"] for m in fresh]
    assert len(cA.data) == len(json.loads(src.read_text())), "report set A must come entirely from the V11 cache"
    with qb.sampled(0.7, 1):
        repB = [qr.summarise(m, docs, cB)["text"] for m in fresh]
    print("reports A/B:", len(repA), len(repB), "identical texts:", sum(a == b for a, b in zip(repA, repB)), flush=True)
    gq = json.loads((A / "qa" / "cache_questions.json").read_text())["global"]["items"]
    ca, cj = Cache("c2_answers"), Cache("c2_judge")
    blocks = lambda reps: [f"Report {k + 1}:\n{t}" for k, t in enumerate(sorted(set(reps)))]  # noqa: E731
    ans = {}
    for i, q in enumerate(gq):
        ans[("A0", i)] = qg.answer(q, blocks(repA), "aa|A0", ca)["text"]
        ans[("B0", i)] = qg.answer(q, blocks(repB), "aa|B0", ca)["text"]
        for s in (1, 2):
            with qb.sampled(0.7, s):
                ans[(f"A{s}", i)] = qg.answer(q, blocks(repA), f"aa|A{s}", ca)["text"]
    print("answers done", flush=True)
    out = dict(settings=qb.SETTINGS, n_questions=len(gq), n_reports=len(fresh), tests={})
    for name, (x, y) in {"answers_A1_vs_A2": ("A1", "A2"), "reports_A0_vs_B0": ("A0", "B0")}.items():
        rows = []
        for i, q in enumerate(gq):
            for order in (0, 1):
                a, b = ans[(x, i)], ans[(y, i)]
                v = qg.compare(q, *((a, b) if order == 0 else (b, a)), cj)["v"]
                mine = 1 if order == 0 else 2
                rows.append(dict(q=i, order=order, **{c: int(v[c]) for c in qg.CRIT},
                                 **{c + "_s": 1.0 if v[c] == mine else 0.5 if v[c] == 0 else 0.0 for c in qg.CRIT}))
        verdicts = [r[c] for r in rows for c in qg.CRIT]
        nontie = [v for v in verdicts if v != 0]
        rec = dict(tie_rate=float(np.mean([v == 0 for v in verdicts])),
                   first_position_share=float(np.mean([v == 1 for v in nontie])) if nontie else None, criteria={})
        for c in qg.CRIT:
            per_q = [np.mean([r[c + "_s"] for r in rows if r["q"] == i]) for i in range(len(gq))]
            boot = [np.mean(np.random.default_rng(b).choice(per_q, len(per_q))) for b in range(2000)]
            rec["criteria"][c] = dict(win=float(np.mean(per_q)), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))])
        rec["rows"] = rows
        out["tests"][name] = rec
        print(name, {k: v for k, v in rec.items() if k != "rows"}, flush=True)
    (HERE / "c2_aa_global.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
