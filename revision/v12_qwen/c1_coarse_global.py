"""C1 (V12): do the stable hierarchies (k-core, consensus) give worse GLOBAL answers than GraphRAG's Leiden at the
coarse levels that overview queries read?  Entity graph, final version (order 0); everything with local Qwen3-14B
(qwen12.py): reports of every store, map-reduce answers, pairwise judge.  Shared inputs: the 30 global questions of
the paper (gpt-generated, cached) and the entity graph.

Stores (same report budget as GraphRAG's top level, B = number of GraphRAG top-level communities = 111):
  leiden_top     GraphRAG hierarchical Leiden, level 0 (all B communities)                    [reference]
  kcore_top      authors' k-core (RkH), level 0 = 1 community (the whole LCC) -> 1 report
  kcore_l1_top   authors' k-core (RkH), level 1, the B largest communities
  consensus_top  consensus (100 runs), level 0, the B largest communities
Reports: hier_qa.report (member entities by degree + relationships, 4k + 8k characters, the paper's prompt).
Answers: hier_qa's map-reduce (key points from batches of 75 reports, then qa_global.answer), Qwen.
Judge: qa_global.compare (comprehensiveness, diversity, empowerment, directness), pairwise vs leiden_top, both
answer orders; reports win rate (1 win, 0.5 tie, 0 loss, averaged over both orders), tie rate and position bias
(share of non-tie verdicts that pick the answer shown first).
Caches: v12_qwen/cache_*.json (separate from every earlier cache).  Output: v12_qwen/c1_coarse_global.json
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE)]
os.environ.setdefault("OPENAI_API_KEY", "unused-local-only")
import qwen12 as qb  # noqa: E402
import qa_run as qr  # noqa: E402
import qa_global as qg  # noqa: E402
import hier_qa as hq  # noqa: E402

qr.call = qb.call


class Cache(qr.Cache):
    def __init__(self, name):
        self.path = HERE / f"cache_{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}


BATCH = 75
STORES = ["leiden_top", "kcore_top", "kcore_l1_top", "consensus_top"]


def main():
    g = json.loads((A / "entity_graph" / "graph.json").read_text())
    H = json.loads((HERE / "c1_hierarchies.json").read_text())
    B = sum(1 for l, _ in H["leiden"] if l == 0)
    top = lambda name, lvl, k=None: sorted((tuple(m) for l, m in H[name] if l == lvl), key=lambda m: -len(m))[:k]  # noqa: E731
    stores = {"leiden_top": top("leiden", 0), "kcore_top": top("kcore_rkh", 0), "kcore_l1_top": top("kcore_rkh", 1, B),
              "consensus_top": top("consensus", 0, B)}
    lcc = {v for _, m in H["leiden"] for v in m}
    cover = {s: len({v for m in stores[s] for v in m} & lcc) / len(lcc) for s in STORES}
    for s in STORES:
        print(s, "reports", len(stores[s]), "entity coverage %.3f" % cover[s], flush=True)
    edge_index = {}
    for j, (a, b) in enumerate(g["edges"]):
        edge_index.setdefault(a, []).append(j)
        edge_index.setdefault(b, []).append(j)
    ch, ca, cj = Cache("c1_reports"), Cache("c1_answers"), Cache("c1_judge")
    need = sorted({m for v in stores.values() for m in v})
    with cf.ThreadPoolExecutor(1) as ex:                          # serial: one Ollama slot
        reps = dict(zip(need, ex.map(lambda m: hq.report(list(m), g, edge_index, ch), need)))
    print("reports written/cached:", len(reps), flush=True)
    texts = {s: [reps[m]["text"] for m in stores[s]] for s in STORES}
    gq = json.loads((A / "qa" / "cache_questions.json").read_text())["global"]["items"]

    def ans(name, i):
        rs, points = texts[name], []
        for b in range(0, len(rs), BATCH):
            chunk = rs[b:b + BATCH]
            key = hashlib.sha1(("hmap|" + gq[i] + "|" + "||".join(chunk)).encode()).hexdigest()
            hit = ca.get(key)
            if not hit:
                prompt = ("From the community reports below, list at most 8 key points that help answer the question, "
                          "each with an importance score 0-100. Use only the reports.\n\n"
                          f"Question: {gq[i]}\n\n" + "\n\n".join(f"Report {k + 1}:\n{t}" for k, t in enumerate(chunk)))
                text, usage = qr.call(None, [{"role": "user", "content": prompt}], max_tokens=2000)
                hit = dict(text=text, usage=usage)
                ca.put(key, hit)
            points.append(hit)
        blocks = [f"Key points {k + 1}:\n{p['text']}" for k, p in enumerate(points)]
        a_ = qg.answer(gq[i], blocks, "c1-reduce|" + name, ca)
        return a_["text"]

    answers = {(s, i): ans(s, i) for s in STORES for i in range(len(gq))}
    print("answers done", flush=True)
    rows = []
    for s in STORES[1:]:
        for i in range(len(gq)):
            for order in (0, 1):
                x, f = answers[(s, i)], answers[("leiden_top", i)]
                v = qg.compare(gq[i], *((x, f) if order == 0 else (f, x)), cj)["v"]
                mine = 1 if order == 0 else 2
                rows.append(dict(store=s, q=i, order=order, **{c: int(v[c]) for c in qg.CRIT},
                                 **{c + "_score": 1.0 if v[c] == mine else 0.5 if v[c] == 0 else 0.0 for c in qg.CRIT}))
    out = dict(settings=qb.SETTINGS, budget=B, coverage=cover, n_questions=len(gq), stores={})
    for s in STORES[1:]:
        R = [r for r in rows if r["store"] == s]
        rec = dict(reports=len(stores[s]))
        verdicts = [r[c] for r in R for c in qg.CRIT]
        nontie = [x for x in verdicts if x != 0]
        rec["tie_rate"] = float(np.mean([x == 0 for x in verdicts]))
        rec["first_position_share"] = float(np.mean([x == 1 for x in nontie])) if nontie else None
        for c in qg.CRIT:
            per_q = [np.mean([r[c + "_score"] for r in R if r["q"] == i]) for i in range(len(gq))]
            boot = [np.mean(np.random.default_rng(b).choice(per_q, len(per_q))) for b in range(2000)]
            rec[c] = dict(win=float(np.mean(per_q)), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                          p=float(stats.wilcoxon(np.array(per_q) - 0.5).pvalue) if any(x != 0.5 for x in per_q) else 1.0)
        out["stores"][s] = rec
        print(s, json.dumps(rec), flush=True)
    out["rows"] = rows
    (HERE / "c1_coarse_global.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
