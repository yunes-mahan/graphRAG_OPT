"""More local questions for power: 180 additional questions (new seed papers, same generator and prompts as
qa_run.py), answered by 6 key summary stores and by plain vector retrieval of papers, judged by gpt-5.6-terra.
Stores: FRESH, recompute/warm-start/hybrid at tau=0.9, hybrid and recompute never refreshed, papers|vector.
Together with the 120 original questions this gives 300 questions for these stores.
Caches: qa/cache_questions_more.json, cache_answers_more.json, cache_judge_more.json.  Output: qa/results_more.json
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import proto_rag as pr  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

QA = HERE / "qa"
N_MORE = 180
STORES = ["FRESH|recompute all", "recompute_leiden|tau=0.9", "warmstart_leiden|tau=0.9", "hybrid|tau=0.9",
          "hybrid|tau=0.0", "recompute_leiden|tau=0.0", "frontier_leiden|tau=0.9"]


def more_questions(docs, years, old_seeds, cache):
    hit = cache.get("questions")
    if hit:
        return hit["items"]
    idx = np.arange(len(docs))
    corpus = [docs[i][0] + " " + docs[i][1] for i in idx]
    X = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True).fit_transform(corpus)
    rng = np.random.default_rng(8)
    recent = np.setdiff1d(idx[years >= 2019], list(old_seeds))
    seeds = rng.choice(recent, N_MORE * 2, replace=False)
    items = []

    def gen(s):
        sims = linear_kernel(X[s], X).ravel()
        nb = np.argsort(-sims)[:8]
        ctx = "\n".join(f"[{j}] {docs[j][0]}: {docs[j][1][:700]}" for j in nb)
        prompt = ("From the papers below, write ONE broad, topic-level question that a user of a research "
                  "knowledge base might ask (e.g. 'What are the main approaches and trends in ...?'), and exactly "
                  "4 short key points that a good answer should mention. Each key point must be a general theme, "
                  "method family or trend that is supported by at least two of the papers; do NOT use numbers, "
                  "dataset names, system names or details of a single paper. Return JSON "
                  "{\"question\": str, \"key_points\": [str, str, str, str]}.\n\n" + ctx)
        text, u = qr.call(qr.Q_MODEL, [{"role": "user", "content": prompt}], json_mode=True)
        try:
            j = json.loads(text)
            if isinstance(j.get("question"), str) and len(j.get("key_points", [])) == 4:
                return dict(seed=int(s), papers=[int(x) for x in nb], **j)
        except json.JSONDecodeError:
            pass
        return None

    with cf.ThreadPoolExecutor(8) as ex:
        cand = list(ex.map(gen, seeds[: int(N_MORE * 1.1)]))
    items = [c for c in cand if c][:N_MORE]
    cache.put("questions", dict(items=items))
    return items


def main():
    plan = json.loads((QA / "plan.json").read_text())
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    years = nodes.year.values
    node_t = np.load(QA / "node_t.npy")
    present = np.where(node_t >= 0)[0]
    old = qr.Cache("questions").get("questions")["items"]
    items = more_questions(docs, years, {it["seed"] for it in old}, qr.Cache("questions_more"))
    print("new questions:", len(items), flush=True)
    cs, ca, cj = qr.Cache("summaries"), qr.Cache("answers_more"), qr.Cache("judge_more")
    stores = {}
    for m in ["recompute_leiden", "warmstart_leiden", "hybrid", "frontier_leiden"]:
        for tau, v in plan["methods"][m]["taus"].items():
            stores[f"{m}|tau={tau}"] = v["final_store"]
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    comm = {}
    for v, c in zip(present, lab):
        comm.setdefault(int(c), []).append(int(v))
    stores["FRESH|recompute all"] = [sorted(x) for x in comm.values() if len(x) >= qp.MIN_SIZE]
    E = np.load(QA / "emb_te3small.npy")
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    qf = QA / "emb_questions_more.npy"
    if qf.exists():
        Qe = np.load(qf)
    else:
        r = qr.client.embeddings.create(model="text-embedding-3-small", input=[it["question"] for it in items])
        Qe = np.asarray([d.embedding for d in r.data], dtype=np.float32)
        np.save(qf, Qe)
    Qe /= np.linalg.norm(Qe, axis=1, keepdims=True)
    paper = lambda v: f"- {docs[v][0]}: {docs[v][1][:600]}"  # noqa: E731

    def work(a):
        name, qi = a
        it = items[qi]
        if name == "papers|vector":
            best = present[np.argsort(-(E[present] @ Qe[qi]))[:pr.K_PAPERS]]
            ans = pr.answer_ctx(it["question"], [paper(int(v)) for v in best], ca, name)
        else:
            ans = qr.answer(it["question"], [cs.get(qr.key_of(s))["text"] for s in stores[name]], ca, name)
        j = qr.judge(it, ans["text"], cj)
        return dict(store=name, q=it["question"], recall=float(np.mean(j["covered"])))

    jobs = [(n, i) for n in STORES + ["papers|vector"] for i in range(len(items))]
    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(work, jobs))
    df = pd.DataFrame(rows)
    print(df.groupby("store").recall.agg(["mean", "count"]).round(3), flush=True)
    (QA / "results_more.json").write_text(json.dumps(dict(per_question=rows, questions=items), indent=1))


if __name__ == "__main__":
    main()
