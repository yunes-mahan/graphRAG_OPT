"""QA experiment, idea B: SEMANTIC staleness rule for summary regeneration.

Same subgraph stream, maintainers (labels from qa/plan.json), questions, retrieval, answer and judge
models as qa_run.py. Only the regeneration rule differs:
    regenerate the summary of community k at version t  iff  k is new, or
    cosine(centroid of k's members when its summary was written, centroid of k's members now) < theta
Centroids are means of the 128-d text embeddings (ogbn-arxiv node features). theta in {0.99, 0.97, 0.95}
fixed a priori. Communities need >= 5 members to carry a summary (as before).
usage: python qa_semantic.py plan   -> count new summaries needed (no LLM calls)
       python qa_semantic.py run    -> generate missing summaries, answer, judge, write qa/results_semantic.json
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
import qa_plan as qp  # noqa: E402

QA = config.RESULTS / "qa"
import os  # noqa: E402
THETAS = [float(x) for x in os.environ.get("THETAS", "0.99,0.97,0.95").split(",")]
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]


def centroid(members, feat):
    c = feat[sorted(members)].astype(np.float64).mean(0)
    return c / max(np.linalg.norm(c), 1e-12)


def semantic_events(labels, node_t, feat, theta):
    store, cen, events = {}, {}, []
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nodes, lab):
            cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < qp.MIN_SIZE:
                continue
            now = centroid(mem, feat)
            regen = cid not in store or float(np.dot(cen[cid], now)) < theta
            if regen:
                store[cid] = frozenset(mem)
                cen[cid] = now
                events.append((t, sorted(mem)))
        for cid in list(store):
            if cid not in cur or len(cur[cid]) < qp.MIN_SIZE:
                del store[cid]
                cen.pop(cid, None)
    return events, [sorted(s) for s in store.values()]


def build_stores():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    d, _ = qp.build_subgraph()
    feat = np.asarray(d["feat"], dtype=np.float32)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-9)
    stores = {}
    for m in METHODS:
        labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
        for th in THETAS:
            ev, final = semantic_events(labels, node_t, feat, th)
            stores[f"{m}|sem={th}"] = dict(events=ev, final=final)
    return stores


def main(mode):
    import qa_run as qr
    stores = build_stores()
    cs = qr.Cache("summaries")
    need = {qr.key_of(s): s for st in stores.values() for s in [e[1] for e in st["events"]] + st["final"]}
    missing = [s for k, s in need.items() if not cs.get(k)]
    for n, st in stores.items():
        print(f"{n:28s} generations={len([e for e in st['events'] if e[0] > 0]):4d} final={len(st['final'])}")
    print(f"summaries needed: {len(need)}, already cached: {len(need) - len(missing)}, new: {len(missing)}")
    if mode != "run":
        return
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda s: qr.summarise(s, docs, cs), missing))
    items = qr.Cache("questions").get("questions")["items"]
    ca, cj = qr.Cache("answers"), qr.Cache("judge")

    def work(args):
        name, st, it = args
        sums = [cs.get(qr.key_of(s))["text"] for s in st["final"]]
        a = qr.answer(it["question"], sums, ca, name)
        j = qr.judge(it, a["text"], cj)
        return dict(store=name, q=it["question"], recall=float(np.mean(j["covered"])))

    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, [(n, st, it) for n, st in stores.items() for it in items]))
    base = json.loads((QA / "results.json").read_text())
    ref = pd.DataFrame(base["per_question"]).pivot_table(index="q", columns="store", values="recall")["FRESH|recompute all"]
    per = pd.DataFrame(rows).pivot_table(index="q", columns="store", values="recall")
    out = []
    for n, st in stores.items():
        ev = [e for e in st["events"] if e[0] > 0]
        inp = sum(cs.get(qr.key_of(s))["usage"]["inp"] for _, s in ev)
        d = (per[n] - ref).values
        p = stats.wilcoxon(per[n], ref, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
        boot = [np.mean(np.random.default_rng(b).choice(per[n].values, len(per))) for b in range(2000)]
        out.append(dict(store=n, recall=float(per[n].mean()), ci_lo=float(np.percentile(boot, 2.5)),
                        ci_hi=float(np.percentile(boot, 97.5)), diff=float(d.mean()), p=float(p),
                        update_generations=len(ev), update_input_tokens=int(inp), final_summaries=len(st["final"])))
    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    (QA / "results_semantic.json").write_text(json.dumps(dict(summary=out, per_question=rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
