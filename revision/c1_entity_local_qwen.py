"""C1: local questions on the LLM entity graph with GraphRAG's hierarchical clustering at levels 0-2 (GraphRAG's default
range), maintained index (warm-started hierarchical Leiden with identifier matching, tau = 0.9) vs fresh re-index.
Open-model run: Qwen3-14B (v2/qwen_backend.py) writes the reports of BOTH stores, generates the questions, answers and
marks key points; caches entity_graph/cache_*_c1qwen.json / qa/cache_*_c1qwen.json, separate from all gpt caches.

Stores (final version of document order 0; analysis/graphrag_hier_stores_entity.json, tau = 0.9, every community):
  fresh : full re-index of the final graph;  warm : maintained store, each report written for the member set at its
  last rewrite ("written"), which can differ from the current members ("cur").
Questions: N local questions; seed = random document of 2019-2020 (rng 11), group = seed + its 7 most similar documents
  of the entity graph (TF-IDF over abstracts), question + 4 key points with the paper's question prompt
  (qa_run.make_questions).  Questions are generated from documents, independently of both stores.
Retrieval: for each store, every community at levels 0-2 is represented by its report INPUT (current members' entity
  names and the relationship descriptions among them, same order and 4k + 8k character budget as hier_qa.report); the
  6 communities with the highest TF-IDF cosine to the question are retrieved (one vectorizer fitted on both stores),
  their reports are written lazily by Qwen (hier_qa.report on the "written" members), and the question is answered
  from these 6 reports with qa_run.answer and judged with qa_run.judge (key-point recall), as in the paper.
Statistics as in the paper (bootstrap CI, Wilcoxon, TOST margin 0.05 / 90% interval).
Output: analysis/v2/c1_entity_local_qwen.json      usage: python v2/c1_entity_local_qwen.py [N]
"""
import concurrent.futures as cf
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A))
sys.path.insert(0, str(HERE))
os.environ.setdefault("OPENAI_API_KEY", "unused-local-only")
import qwen_backend as qb  # noqa: E402
import qa_run as qr  # noqa: E402
import qa_plan as qp  # noqa: E402
import hier_qa as hq  # noqa: E402
from c2_qwen_qa import tost  # noqa: E402

SUFFIX = "_c1qwen"
_orig_init = qr.Cache.__init__
qr.Cache.__init__ = lambda self, name: _orig_init(self, name + SUFFIX)
qr.call = qb.call
G = A / "entity_graph"
STORES = {"fresh": "reindex", "warm": "warm"}
LEVELS = (0, 1, 2)
WORKERS = int(os.environ.get("C1_WORKERS", "2"))
QPROMPT = ("From the papers below, write ONE broad, topic-level question that a user of a research "
           "knowledge base might ask (e.g. 'What are the main approaches and trends in ...?'), and exactly "
           "4 short key points that a good answer should mention. Each key point must be a general theme, "
           "method family or trend that is supported by at least two of the papers; do NOT use numbers, "
           "dataset names, system names or details of a single paper. Return JSON "
           "{\"question\": str, \"key_points\": [str, str, str, str]}.\n\n")   # = qa_run.make_questions


def member_text(members, g, edge_index):
    """Report input of hier_qa.report (entities by degree, then relationships), used for retrieval."""
    ms = set(members)
    deg = {v: sum(1 for j in edge_index.get(v, ()) if set(g["edges"][j]) <= ms) for v in members}
    order = sorted(members, key=lambda v: -deg[v])
    lines, used = [], 0
    for v in order:
        s = f"{g['entities'][v]} ({g['types'][v]})"
        if used + len(s) > 4000:
            break
        lines.append(s)
        used += len(s)
    rel, used = [], 0
    for v in order:
        for j in edge_index.get(v, ()):
            a, b = g["edges"][j]
            if a == v and b in ms:
                s = f"{g['entities'][a]} {g['entities'][b]} {g['edge_desc'][j]}"
                if used + len(s) > 8000:
                    break
                rel.append(s)
                used += len(s)
    return " ".join(lines) + " " + " ".join(rel)


def main(n):
    g = json.loads((G / "graph.json").read_text())
    dump = json.loads((A / "graphrag_hier_stores_entity.json").read_text())
    edge_index = {}
    for j, (a, b) in enumerate(g["edges"]):
        edge_index.setdefault(a, []).append(j)
        edge_index.setdefault(b, []).append(j)
    comms = {s: [x for x in dump[src] if x["level"] in LEVELS] for s, src in STORES.items()}
    ctext = {s: [member_text(x["cur"], g, edge_index) for x in comms[s]] for s in STORES}
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(ctext["fresh"] + ctext["warm"])
    CX = {s: vec.transform(ctext[s]) for s in STORES}
    nodes = pd.read_csv(A / "qa" / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    did = sorted(int(d) for d in g["docs"])
    year = {int(d): v["year"] for d, v in g["docs"].items()}
    X = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True).fit_transform(
        [docs[d][0] + " " + docs[d][1] for d in did])
    cq, ch, ca, cj = qr.Cache("questions_entity"), qr.Cache("hreports"), qr.Cache("answers_entity"), qr.Cache("judge_entity")
    hit = cq.get("items")
    if hit and len(hit) >= n:
        items = hit[:n]
    else:
        rng = np.random.default_rng(11)
        recent = [i for i, d in enumerate(did) if year[d] >= 2019]
        seeds = rng.choice(recent, 2 * n, replace=False)

        def gen(i):
            nb = np.argsort(-linear_kernel(X[i], X).ravel())[:8]
            ctx = "\n".join(f"[{did[j]}] {docs[did[j]][0]}: {docs[did[j]][1][:700]}" for j in nb)
            text, _ = qb.call(None, [{"role": "user", "content": QPROMPT + ctx}], json_mode=True)
            try:
                j = json.loads(text)
                if isinstance(j.get("question"), str) and len(j.get("key_points", [])) == 4:
                    return dict(seed=int(did[i]), papers=[int(did[x]) for x in nb], **j)
            except json.JSONDecodeError:
                pass
            return None
        items = []
        with cf.ThreadPoolExecutor(WORKERS) as ex:
            for it in ex.map(gen, seeds):
                if it is not None and len(items) < n:
                    items.append(it)
        cq.put("items", items)
    print(f"questions: {len(items)}", flush=True)

    def work(a):
        s, it = a
        sims = linear_kernel(vec.transform([it["question"]]), CX[s]).ravel()
        top = list(np.argsort(-sims)[:6])
        reps = [hq.report(sorted(comms[s][k]["written"]), g, edge_index, ch)["text"] for k in top]
        ans = qr.answer(it["question"], reps, ca, f"c1|{s}")
        j = qr.judge(it, ans["text"], cj)
        stale = sum(1 for k in top if comms[s][k]["written"] != comms[s][k]["cur"])
        return dict(store=s, q=it["question"], recall=float(np.mean(j["covered"])), stale_retrieved=stale,
                    levels=[comms[s][k]["level"] for k in top])

    with cf.ThreadPoolExecutor(WORKERS) as ex:
        rows = list(ex.map(work, [(s, it) for it in items for s in STORES]))
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="q", columns="store", values="recall")
    summ = []
    for s in STORES:
        r = piv[s]
        boot = [np.mean(np.random.default_rng(b).choice(r.values, len(r))) for b in range(2000)]
        row = dict(store=s, recall=float(r.mean()), ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                   n_q=int(len(r)), stale_retrieved_share=float(df[df.store == s].stale_retrieved.sum() / (6 * len(r))),
                   level_share={str(l): float(np.mean([l in x for x in df[df.store == s].levels])) for l in LEVELS})
        if s != "fresh":
            d = r - piv["fresh"]
            p_t, lo, hi = tost(d)
            row.update(diff=float(d.mean()), wilcoxon_p=float(stats.wilcoxon(r, piv["fresh"], zero_method="zsplit").pvalue) if (d != 0).any() else 1.0,
                       tost_p=p_t, ci90=[lo, hi], equivalent=p_t < 0.05, noninferior=lo > -0.05)
        summ.append(row)
        print(row, flush=True)
    n_rep = {s: len(comms[s]) for s in STORES}
    (HERE / "c1_entity_local_qwen.json").write_text(json.dumps(dict(
        settings=dict(qb.SETTINGS, ollama=qb.ollama_version()), levels=list(LEVELS), communities_levels_0_2=n_rep,
        reports_written=len(ch.data), n_questions=len(items), summary=summ, per_question=rows), indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
