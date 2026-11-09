"""End-to-end global QA on GraphRAG's hierarchical clustering (LLM entity graph, document order 0).

Stores (final version, staleness tolerance 0.9, every community carries a report), top level (level 0):
  fresh          : full re-index of the final graph (graspologic hierarchical Leiden, GraphRAG settings)
  warm           : warm-started hierarchical Leiden with identifier matching
  graphrag_update: GraphRAG's incremental update (old top-level communities kept, delta communities appended)
  zep_attach     : Zep-style attachment (top-level reports written for the member set at the last rewrite)
Re-index + matching has the same final top level as the fresh index and is omitted.
Reports: gpt-5.6-luna, GraphRAG-style prompt from member entities and their relationships (at most 4k + 8k
characters, i.e. half of the flat experiment's budget). A store's report describes the member set at the time it was
last written. Questions: the 30 global questions; answers by map-reduce over all top-level reports (75 reports per
map call); pairwise judge gpt-5.6-terra against the fresh answer, both orders. Writes entity_graph/hier_qa.json.
usage: python hier_qa.py plan | run
"""

import concurrent.futures as cf
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_global as qg  # noqa: E402
import qa_run as qr  # noqa: E402

G = HERE / "entity_graph"
STORES = ["fresh", "warm", "graphrag_update", "zep_attach"]
SRC = {"fresh": "reindex", "warm": "warm", "graphrag_update": "graphrag_update", "zep_attach": "zep_attach"}
BATCH = 75


def report(members, g, edge_index, cache):
    k = hashlib.sha1(("hreport|" + ",".join(map(str, members))).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    ms = set(members)
    deg = {v: sum(1 for j in edge_index.get(v, ()) if set(g["edges"][j]) <= ms) for v in members}
    order = sorted(members, key=lambda v: -deg[v])          # most connected entities first
    lines, used = [], 0
    for v in order:
        s = f"- {g['entities'][v]} ({g['types'][v]})"
        if used + len(s) > 4000:
            break
        lines.append(s)
        used += len(s)
    rel, used = [], 0
    for v in order:
        for j in edge_index.get(v, ()):
            a, b = g["edges"][j]
            if a == v and b in ms:
                s = f"- {g['entities'][a]} -- {g['entities'][b]}: {g['edge_desc'][j]}"
                if used + len(s) > 8000:
                    break
                rel.append(s)
                used += len(s)
    prompt = ("You write a community report for a GraphRAG index built from computer-science research papers. Below "
              f"are the entities of one community ({len(members)} entities, {len(lines)} shown) and the relationships "
              "between them. Write a report of at most 200 words: a short title, the main topics, key methods or "
              "systems, and how they relate. Use only the given information.\n\nEntities:\n" + "\n".join(lines) +
              "\n\nRelationships:\n" + "\n".join(rel))
    text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage)
    cache.put(k, out)
    return out


def main(mode):
    g = json.loads((G / "graph.json").read_text())
    dump = json.loads((HERE / "graphrag_hier_stores_entity.json").read_text())
    stores = {s: sorted({tuple(x["written"]) for x in dump[SRC[s]] if x["level"] == 0}) for s in STORES}
    n_ent = sum(1 for _ in g["entities"])
    for s in STORES:
        cur = set(v for x in dump[SRC[s]] if x["level"] == 0 for v in x["cur"])
        stale = sum(1 for x in dump[SRC[s]] if x["level"] == 0 and x["cur"] != x["written"])
        print(f"{s:16s} top-level reports {len(stores[s]):4d}  stale {stale:3d}  entities covered {len(cur)}", flush=True)
    need = sorted({m for v in stores.values() for m in v})
    est = sum(min(12000, 60 * len(m)) for m in need) / 4
    print(f"unique reports {len(need)}, approx report input tokens {est / 1e6:.2f}M", flush=True)
    if mode != "run":
        return
    edge_index = {}
    for j, (a, b) in enumerate(g["edges"]):
        edge_index.setdefault(a, []).append(j)
        edge_index.setdefault(b, []).append(j)
    cache = qr.Cache.__new__(qr.Cache)
    cache.path = G / "cache_hreports.json"
    cache.data = json.loads(cache.path.read_text()) if cache.path.exists() else {}
    with cf.ThreadPoolExecutor(12) as ex:
        reps = dict(zip(need, ex.map(lambda m: report(list(m), g, edge_index, cache), need)))
    texts = {s: sorted({reps[m]["text"] for m in stores[s]}) for s in STORES}
    gq = qr.Cache("questions").get("global")["items"]
    ca, cj = qr.Cache("answers_hier"), qr.Cache("judge_global")

    def ans(a):
        name, i = a
        rs = texts[name]
        points = []
        for b in range(0, len(rs), BATCH):
            chunk = rs[b:b + BATCH]
            key = hashlib.sha1(("hmap|" + gq[i] + "|" + "||".join(chunk)).encode()).hexdigest()
            hit = ca.get(key)
            if not hit:
                prompt = ("From the community reports below, list at most 8 key points that help answer the question, "
                          "each with an importance score 0-100. Use only the reports.\n\n"
                          f"Question: {gq[i]}\n\n" + "\n\n".join(f"Report {k + 1}:\n{t}" for k, t in enumerate(chunk)))
                text, usage = qr.call(qr.ANS_MODEL, [{"role": "user", "content": prompt}], max_tokens=2000)
                hit = dict(text=text, usage=usage)
                ca.put(key, hit)
            points.append(hit)
        blocks = [f"Key points {k + 1}:\n{p['text']}" for k, p in enumerate(points)]
        a_ = qg.answer(gq[i], blocks, "hier-reduce|" + name, ca)
        q_in = sum(p["usage"]["inp"] for p in points) + a_["usage"]["inp"]
        return (name, i), dict(text=a_["text"], query_tokens=q_in)

    with cf.ThreadPoolExecutor(12) as ex:
        answers = dict(ex.map(ans, [(n, i) for n in STORES for i in range(len(gq))]))

    def cmp(a):
        name, i, order = a
        x, f = answers[(name, i)]["text"], answers[("fresh", i)]["text"]
        v = qg.compare(gq[i], *((x, f) if order == 0 else (f, x)), cj)["v"]
        mine = 1 if order == 0 else 2
        return dict(store=name, q=i, **{c: 1.0 if v[c] == mine else 0.5 if v[c] == 0 else 0.0 for c in qg.CRIT})

    with cf.ThreadPoolExecutor(12) as ex:
        rows = list(ex.map(cmp, [(n, i, o) for n in STORES[1:] for i in range(len(gq)) for o in (0, 1)]))
    df = pd.DataFrame(rows).groupby(["store", "q"])[qg.CRIT].mean().reset_index()
    out = []
    for s in STORES:
        rec = dict(store=s, reports=len(texts[s]),
                   report_tokens=int(sum(reps[m]["usage"]["out"] for m in stores[s])),
                   query_tokens=float(np.mean([answers[(s, i)]["query_tokens"] for i in range(len(gq))])))
        if s != "fresh":
            dd = df[df.store == s]
            for c in qg.CRIT:
                rec[c] = float(dd[c].mean())
                rec[c + "_p"] = float(stats.wilcoxon(dd[c] - 0.5).pvalue) if (dd[c] != 0.5).any() else 1.0
        out.append(rec)
    print(pd.DataFrame(out).round(3).to_string(index=False))
    (G / "hier_qa.json").write_text(json.dumps(dict(summary=out, rows=rows), indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
