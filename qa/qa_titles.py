"""QA with title-only summaries (input_variants.py, variant 'titles<=300') for the FRESH partition and the hybrid
tau=0.9 store; compared with the corresponding full-summary stores of qa_run.py. Writes input_variants_qa.json."""

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
import input_variants as iv  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
V = "titles<=300"


def stores_titles(plan, node_t):
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    comm = {}
    for v, c in zip(np.where(node_t >= 0)[0], lab):
        comm.setdefault(int(c), []).append(int(v))
    fresh = [sorted(m) for m in comm.values() if len(m) >= qp.MIN_SIZE]
    return {f"FRESH|{V}": fresh, f"hybrid|tau=0.9|{V}": [sorted(s) for s in
                                                       plan["methods"]["hybrid"]["taus"]["0.9"]["final_store"]]}


def main():
    QA = qr.QA
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cv = qr.Cache("variants")
    stores = stores_titles(plan, node_t)
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda s: iv.summarise_variant(s, docs, V, cv), [s for st in stores.values() for s in st]))
    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")

    def work(a):
        n, st, it = a
        sums = [cv.get(f"{V}|{qr.key_of(s)}")["text"] for s in st]
        ans = qr.answer(it["question"], sums, ca, n)
        return dict(store=n, q=it["question"], recall=float(np.mean(qr.judge(it, ans["text"], cj)["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(n, st, it) for n, st in stores.items() for it in items]))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    piv = pd.concat([base, pd.DataFrame(rows)]).pivot_table(index="q", columns="store", values="recall")
    res = []
    for a, b in [(f"FRESH|{V}", "FRESH|recompute all"), (f"hybrid|tau=0.9|{V}", "hybrid|tau=0.9"),
                 (f"hybrid|tau=0.9|{V}", "FRESH|recompute all")]:
        d = piv[a] - piv[b]
        p = stats.wilcoxon(piv[a], piv[b], zero_method="zsplit").pvalue if (d != 0).any() else 1.0
        res.append(dict(store=a, vs=b, recall=piv[a].mean(), other=piv[b].mean(), diff=d.mean(), p=p, n_q=len(d)))
        print(res[-1])
    (HERE / "input_variants_qa.json").write_text(json.dumps(dict(qa=res, rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main()
