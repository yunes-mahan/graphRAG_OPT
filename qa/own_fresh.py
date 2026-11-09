"""Control for the 'maintained vs fresh' comparison: fresh reports of each maintainer's OWN final partition.

The fresh reference of qa_run.py reports the recomputed partition, so maintained-vs-fresh mixes staleness with
partition differences. Here every community (>= 5 members) of a maintainer's final partition gets a report written
from its current members; the store is answered and judged on the same 300 local questions (120 + 180).
Stores: OWNFRESH|hybrid, OWNFRESH|warmstart_leiden.  Writes qa/own_fresh.json.
usage: python own_fresh.py plan | run
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

QA = HERE / "qa"
METHODS = ["hybrid", "warmstart_leiden"]


def main(mode):
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    present = np.where(node_t >= 0)[0]
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs = qr.Cache("summaries")
    stores = {}
    for m in METHODS:
        comm = {}
        for v, c in zip(present, plan["methods"][m]["labels"][-1]):
            comm.setdefault(int(c), []).append(int(v))
        stores[f"OWNFRESH|{m}"] = [sorted(x) for x in comm.values() if len(x) >= qp.MIN_SIZE]
    for n, st in stores.items():
        miss = sum(1 for s in st if not cs.get(qr.key_of(s)))
        print(f"{n}: {len(st)} communities, {miss} reports not cached", flush=True)
    if mode != "run":
        return
    need = [s for st in stores.values() for s in st]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda s: qr.summarise(s, docs, cs), need))
    old = qr.Cache("questions").get("questions")["items"]
    new = qr.Cache("questions_more").get("questions")["items"]
    ca, cj = qr.Cache("answers_more"), qr.Cache("judge_more")

    def work(a):
        n, it = a
        sums = [cs.get(qr.key_of(s))["text"] for s in stores[n]]
        ans = qr.answer(it["question"], sums, ca, n)
        j = qr.judge(it, ans["text"], cj)
        return dict(store=n, q=it["question"], recall=float(np.mean(j["covered"])))

    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(work, [(n, it) for n in stores for it in old + new]))
    print(pd.DataFrame(rows).groupby("store").recall.agg(["mean", "count"]).round(3))
    (QA / "own_fresh.json").write_text(json.dumps(dict(per_question=rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
