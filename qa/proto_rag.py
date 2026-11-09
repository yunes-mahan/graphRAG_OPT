"""Do we need community summaries at all?  Prototype (centroid-embedding) routing vs summaries.

All stores answer the same 120 questions with the same answer model, judge and context budget (about 1.8k tokens):
  summaries|tfidf      : 6 most relevant community summaries by TF-IDF  (= 'FRESH|recompute all' of qa_run.py)
  summaries|proto      : 6 communities whose prototype (mean text-embedding-3-small of members) is closest to the
                         question embedding; their summaries  (= lazy summarisation: only retrieved ones are needed)
  papers|proto         : the same 6 communities, then the 12 member papers closest to the question (no summaries)
  papers|vector        : plain vector RAG, 12 papers closest to the question (no communities, no summaries)
Partition: final recomputed Leiden partition (communities >= 5 members), as for the FRESH reference.
Cost accounting: update-time LLM tokens (summaries) vs per-question input tokens.
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
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
K_COMM, K_PAPERS = 6, 12


def answer_ctx(question, ctx_blocks, cache, store):
    import hashlib
    k = hashlib.sha1((store + "|" + question).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    ctx = "\n\n".join(ctx_blocks)
    prompt = ("Answer the question using ONLY the context below. Be specific and concise "
              "(at most 150 words). If the context does not contain the information, say so.\n\n"
              f"Question: {question}\n\n{ctx}")
    text, usage = qr.call(qr.ANS_MODEL, [{"role": "user", "content": prompt}], max_tokens=1200)
    out = dict(text=text, usage=usage)
    cache.put(k, out)
    return out


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    E = np.load(QA / "emb_te3small.npy")
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    present = np.where(node_t >= 0)[0]
    comm = {}
    for v, c in zip(present, lab):
        comm.setdefault(int(c), []).append(int(v))
    comms = [sorted(m) for m in comm.values() if len(m) >= qp.MIN_SIZE]
    P = np.stack([E[m].mean(0) for m in comms])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    cs = qr.Cache("summaries")
    sums = [cs.get(qr.key_of(m))["text"] for m in comms]
    items = qr.Cache("questions").get("questions")["items"]
    qcache = QA / "emb_questions.npy"
    if qcache.exists():
        Qe = np.load(qcache)
    else:
        r = qr.client.embeddings.create(model="text-embedding-3-small", input=[it["question"] for it in items])
        Qe = np.asarray([d.embedding for d in r.data], dtype=np.float32)
        np.save(qcache, Qe)
    Qe /= np.linalg.norm(Qe, axis=1, keepdims=True)
    ca, cj = qr.Cache("answers_proto"), qr.Cache("judge")
    paper = lambda v: f"- {docs[v][0]}: {docs[v][1][:600]}"  # noqa: E731
    retrieved = set()

    def work(a):
        store, qi = a
        it, q = items[qi], Qe[qi]
        top_c = list(np.argsort(-(P @ q))[:K_COMM])
        if store == "summaries|proto":
            retrieved.update(top_c)
            blocks = [f"Report {i + 1}:\n{sums[c]}" for i, c in enumerate(top_c)]
        elif store == "papers|proto":
            cand = np.array([v for c in top_c for v in comms[c]])
            best = cand[np.argsort(-(E[cand] @ q))[:K_PAPERS]]
            blocks = [paper(int(v)) for v in best]
        else:
            best = present[np.argsort(-(E[present] @ q))[:K_PAPERS]]
            blocks = [paper(int(v)) for v in best]
        a_ = answer_ctx(it["question"], blocks, ca, store)
        j = qr.judge(it, a_["text"], cj)
        return dict(store=store, q=it["question"], recall=float(np.mean(j["covered"])), inp=a_["usage"]["inp"])

    jobs = [(s, i) for s in ("summaries|proto", "papers|proto", "papers|vector") for i in range(len(items))]
    with cf.ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(work, jobs))
    base = pd.DataFrame(json.loads((QA / "results.json").read_text())["per_question"])
    base = base[base.store == "FRESH|recompute all"].assign(store="summaries|tfidf")
    ca_main = qr.Cache("answers")
    import hashlib
    base["inp"] = [ca_main.get(hashlib.sha1(("FRESH|recompute all|" + q).encode()).hexdigest())["usage"]["inp"]
                   for q in base.q]
    df = pd.concat([base[["store", "q", "recall", "inp"]], pd.DataFrame(rows)])
    piv = df.pivot_table(index="q", columns="store", values="recall")
    ref = piv["summaries|tfidf"]
    out = []
    for s in piv.columns:
        d = piv[s] - ref
        p = stats.wilcoxon(piv[s], ref, zero_method="zsplit").pvalue if (d != 0).any() else 1.0
        out.append(dict(store=s, recall=piv[s].mean(), diff_vs_summaries=d.mean(), p=p,
                        answer_input_tokens=df[df.store == s].inp.mean()))
    full_tok = sum(cs.get(qr.key_of(m))["usage"]["inp"] + cs.get(qr.key_of(m))["usage"]["out"] for m in comms)
    lazy_tok = sum(cs.get(qr.key_of(comms[c]))["usage"]["inp"] + cs.get(qr.key_of(comms[c]))["usage"]["out"]
                   for c in retrieved)
    res = pd.DataFrame(out)
    print(res.round(3).to_string(index=False))
    print(f"summary tokens for all {len(comms)} communities: {full_tok:,}; lazily, only the {len(retrieved)} "
          f"retrieved by 120 questions: {lazy_tok:,} ({lazy_tok / full_tok:.0%})")
    (HERE / "proto_rag.json").write_text(json.dumps(dict(summary=out, rows=rows, n_comm=len(comms),
                                                         n_retrieved=len(retrieved), full_summary_tokens=full_tok,
                                                         lazy_summary_tokens=lazy_tok), indent=1, default=float))


if __name__ == "__main__":
    main()
