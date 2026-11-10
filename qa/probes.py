"""Targeted probes: can broad topic questions see staleness and coverage loss at all?

STALENESS probe (QA stream). For the four never-refreshed stores (tau = never: recompute, warm start, frontier,
hybrid), take final communities whose report was written for an older member set and that have since gained at
least 4 papers. For 25 such communities per store (random, seed 0), gpt-5.6-terra writes one topic-level question
with 4 key points supported ONLY by the papers that joined after the report was written (same prompt as the local
questions). The question is answered by the stale store and by fresh reports of the same maintainer's own final
partition (same partition, current reports), each from its 6 most relevant reports (TF-IDF), and judged as before.

COVERAGE probe (LLM entity graph, hierarchical top level). For GraphRAG's incremental update and Zep-style
attachment, take documents added after version 0 whose extracted entities are mostly (>= 75%) outside every
top-level community of that store (50 documents per store, random, seed 0). gpt-5.6-terra writes one question with
4 key points supported by the document and its 3 most similar uncovered documents. The question is answered from
the 6 most relevant top-level reports of that store and of the fresh re-index, and judged as before.

usage: python probes.py plan | run      -> writes probes.json
"""

import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import entity_graph_eval as eg  # noqa: E402
import hier_qa as hq  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

import os  # noqa: E402
TAU = os.environ.get("PROBE_TAU", "0.0")          # "0.0" = never refreshed; "0.9" = kept by the Jaccard rule
ONLY = os.environ.get("PROBE_ONLY", "both")        # staleness | coverage | both
SUFFIX = os.environ.get("PROBE_SUFFIX", "")
QA = HERE / "qa"
G = HERE / "entity_graph"
MAINT = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]
OWN = {"recompute_leiden": "FRESH|recompute all", "warmstart_leiden": "OWNFRESH|warmstart_leiden",
       "frontier_leiden": "OWNFRESH|frontier_leiden", "hybrid": "OWNFRESH|hybrid"}
QPROMPT = ("From the papers below, write ONE broad, topic-level question that a user of a research knowledge base might "
           "ask (e.g. 'What are the main approaches and trends in ...?'), and exactly 4 short key points that a good "
           "answer should mention. Each key point must be a general theme, method family or trend that is supported by "
           "the papers; do NOT use numbers, dataset names, system names or details of a single paper. Return JSON "
           "{\"question\": str, \"key_points\": [str, str, str, str]}.\n\n")


def gen_question(papers_txt, tag, cq):
    k = hashlib.sha1(("probe|" + tag + "|" + papers_txt).encode()).hexdigest()
    hit = cq.get(k)
    if hit:
        return hit
    for _ in range(3):
        text, u = qr.call(qr.Q_MODEL, [{"role": "user", "content": QPROMPT + papers_txt}], json_mode=True)
        try:
            j = json.loads(text)
            if isinstance(j.get("question"), str) and len(j.get("key_points", [])) == 4:
                cq.put(k, j)
                return j
        except json.JSONDecodeError:
            pass
    return None


def answer_from(question, reports, ca, store):
    return qr.answer(question, reports, ca, "probe|" + store)


def main(mode):
    rng = np.random.default_rng(0)
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    present = np.where(node_t >= 0)[0]
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    paper = lambda v: f"[{v}] {docs[v][0]}: {docs[v][1][:700]}"  # noqa: E731
    cs = qr.Cache("summaries")
    # ---------- staleness probe: (maintainer, community) pairs
    stale_jobs = []
    for m in MAINT:
        final = plan["methods"][m]["taus"][TAU]["final_store"]       # member sets at the last write
        lab = plan["methods"][m]["labels"][-1]
        cur = {}
        for v, c in zip(present, lab):
            cur.setdefault(int(c), set()).add(int(v))
        cand = []
        for w in final:
            ws = set(w)
            best = max(cur.values(), key=lambda c: len(c & ws))      # current community of that report
            joined = sorted(best - ws)
            if TAU != "0.0" and len(best & ws) / len(best | ws) < float(TAU):
                continue                                              # only reports the Jaccard rule keeps
            if len(joined) >= 4:
                cand.append(joined)
        pick = [cand[i] for i in rng.permutation(len(cand))[:25]]
        stale_jobs += [(m, j) for j in pick]
        print(f"staleness {m:18s} stale reports with >=4 joined papers: {len(cand)}; probes {len(pick)}", flush=True)
    # ---------- coverage probe: documents outside the top level of a store
    g = json.loads((G / "graph.json").read_text())
    dump = json.loads((HERE / "graphrag_hier_stores_entity.json").read_text())
    d0, nt, et, T = eg.stream(0)
    cov_jobs = []
    for s in ("graphrag_update", "zep_attach"):
        covered = {v for x in dump[s] if x["level"] == 0 for v in x["cur"]}
        cand = []
        for did, v in g["docs"].items():
            ents = v["ents"]
            if not ents or min(nt[e] for e in ents) <= 0:
                continue
            if np.mean([e not in covered for e in ents]) >= 0.75:
                cand.append(int(did))
        pick = [cand[i] for i in rng.permutation(len(cand))[:50]]
        cov_jobs += [(s, d) for d in pick]
        print(f"coverage  {s:18s} new documents >=75% uncovered: {len(cand)}; probes {len(pick)}", flush=True)
    if mode != "run":
        print(f"LLM calls: {len(stale_jobs) + len(cov_jobs)} questions, {2 * (len(stale_jobs) + len(cov_jobs))} answers and judgements")
        return
    cq, ca, cj = qr.Cache("questions_probe"), qr.Cache("answers_probe"), qr.Cache("judge_probe")
    # stale probe stores
    own = {}
    for m in MAINT:
        comm = {}
        for v, c in zip(present, plan["methods"][m]["labels"][-1]):
            comm.setdefault(int(c), []).append(int(v))
        own[m] = [cs.get(qr.key_of(sorted(x)))["text"] for x in comm.values() if len(x) >= qp.MIN_SIZE]
    stale = {m: [cs.get(qr.key_of(s))["text"] for s in plan["methods"][m]["taus"][TAU]["final_store"]] for m in MAINT}

    def run_stale(a):
        m, joined = a
        txt = "\n".join(paper(v) for v in joined[:8])
        it = gen_question(txt, m, cq)
        if it is None:
            return []
        out = []
        for kind, reps in (("stale", stale[m]), ("own_fresh", own[m])):
            ans = answer_from(it["question"], reps, ca, f"{kind}|{m}" + ("" if TAU == "0.0" or kind != "stale" else f"|tau{TAU}"))
            j = qr.judge(it, ans["text"], cj)
            out.append(dict(probe="staleness", maintainer=m, store=kind, q=it["question"], recall=float(np.mean(j["covered"]))))
        return out

    # coverage probe stores (top-level reports, written by hier_qa.py)
    hcache = qr.Cache.__new__(qr.Cache)
    hcache.path = G / "cache_hreports.json"
    hcache.data = json.loads(hcache.path.read_text())
    edge_index = {}
    for j, (a, b) in enumerate(g["edges"]):
        edge_index.setdefault(a, []).append(j)
        edge_index.setdefault(b, []).append(j)
    top = {s: sorted({tuple(x["written"]) for x in dump[hq.SRC.get(s, s)] if x["level"] == 0})
           for s in ("fresh", "graphrag_update", "zep_attach")}
    top_txt = {s: [hq.report(list(mm), g, edge_index, hcache)["text"] for mm in v] for s, v in top.items()}
    dtxt = {d: docs[d][0] + " " + docs[d][1] for d, _ in [(x[1], 0) for x in cov_jobs]}
    alld = [int(d) for d in g["docs"]]
    X = TfidfVectorizer(stop_words="english").fit([docs[d][0] + " " + docs[d][1] for d in alld])

    def run_cov(a):
        s, d = a
        pool = [x[1] for x in cov_jobs if x[0] == s and x[1] != d]
        sims = linear_kernel(X.transform([dtxt[d]]), X.transform([dtxt[p] for p in pool])).ravel()
        group = [d] + [pool[i] for i in np.argsort(-sims)[:3]]
        it = gen_question("\n".join(paper(v) for v in group), "cov|" + s, cq)
        if it is None:
            return []
        out = []
        for kind in (s, "fresh"):
            ans = answer_from(it["question"], top_txt[kind], ca, f"cov|{kind}")
            j = qr.judge(it, ans["text"], cj)
            out.append(dict(probe="coverage", maintainer=s, store="fresh" if kind == "fresh" else "maintained",
                            q=it["question"], recall=float(np.mean(j["covered"]))))
        return out

    with cf.ThreadPoolExecutor(12) as ex:
        rows = [r for rs in ex.map(run_stale, stale_jobs) for r in rs]
        if ONLY == "staleness":
            rows = [r for rs in ex.map(run_stale, stale_jobs) for r in rs]
        elif ONLY == "coverage":
            rows = [r for rs in ex.map(run_cov, cov_jobs) for r in rs]
        else:
            rows += [r for rs in ex.map(run_cov, cov_jobs) for r in rs]
    df = pd.DataFrame(rows)
    summ = []
    for (probe, m), x in df.groupby(["probe", "maintainer"]):
        pv = x.pivot_table(index="q", columns="store", values="recall").dropna()
        a, b = pv.iloc[:, 0], pv.iloc[:, 1]
        cols = list(pv.columns)
        mine = "stale" if "stale" in cols else "maintained"
        other = [c for c in cols if c != mine][0]
        d = pv[mine] - pv[other]
        p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue) if (d != 0).any() else 1.0
        summ.append(dict(probe=probe, maintainer=m, n=len(pv), maintained=float(pv[mine].mean()),
                         reference=float(pv[other].mean()), diff=float(d.mean()), p=p))
    res = pd.DataFrame(summ)
    print(res.round(3).to_string(index=False))
    for probe in sorted(df.probe.unique()):
        x = df[df.probe == probe]
        pv = x.pivot_table(index=["maintainer", "q"], columns="store", values="recall").dropna()
        mine = "stale" if probe == "staleness" else "maintained"
        other = "own_fresh" if probe == "staleness" else "fresh"
        d = pv[mine] - pv[other]
        print(probe, "pooled n", len(d), "diff", round(d.mean(), 3), "p",
              round(stats.wilcoxon(d, zero_method="zsplit").pvalue, 4))
    (HERE / f"probes{SUFFIX}.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
