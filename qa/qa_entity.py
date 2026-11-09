"""End-to-end GraphRAG answers on the LLM-extracted entity graph (document order 0).

Community reports (GraphRAG style) are written by gpt-5.6-luna from the member entities (name, type) and the
relationships inside the community with their extracted descriptions (at most 24k characters). As in GraphRAG's
global search over a coarse level, communities with at least 10 entities carry a report. Per maintainer, reports are
(re)written during the stream by the identifier rule (new identifier or Jaccard < 0.9); GraphRAG's delta update
never rewrites existing reports. A fresh reference reports all communities of the recomputed final partition.
The 30 global questions of qa_global.py are answered from all reports of a store and compared pairwise with the
fresh answer (comprehensiveness, diversity, empowerment, directness; both orders).
usage: python qa_entity.py plan | run
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
import entity_graph_eval as eg  # noqa: E402
import qa_global as qg  # noqa: E402
import qa_run as qr  # noqa: E402

G = HERE / "entity_graph"
MIN = 10
STORES = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "lazy", "graphrag_delta", "priced_warm",
          "priced_frontier"]


def events(labels, node_t):
    store, ev = {}, []
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nodes, lab):
            cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < MIN:
                continue
            old = store.get(cid)
            if old is None or len(mem & old) / len(mem | old) < 0.9:
                store[cid] = frozenset(mem)
                ev.append((t, sorted(mem)))
        for cid in list(store):
            if cid not in cur or len(cur[cid]) < MIN:
                del store[cid]
    return ev, [sorted(s) for s in store.values()]


def report(members, g, edge_index, t_edges, cache):
    k = hashlib.sha1(("ereport|" + ",".join(map(str, members))).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    ms = set(members)
    lines, used = [], 0
    for v in members:
        s = f"- {g['entities'][v]} ({g['types'][v]})"
        if used + len(s) > 8000:
            break
        lines.append(s)
        used += len(s)
    rel = []
    for v in members:
        for j in edge_index.get(v, ()):
            a, b = g["edges"][j]
            if a == v and b in ms and j in t_edges:
                rel.append(f"- {g['entities'][a]} -- {g['entities'][b]}: {g['edge_desc'][j]}")
    rel_txt, used = [], 0
    for s in rel:
        if used + len(s) > 16000:
            break
        rel_txt.append(s)
        used += len(s)
    prompt = ("You write a community report for a GraphRAG index built from computer-science research papers. Below "
              f"are the entities of one community ({len(members)} entities, {len(lines)} shown) and the relationships "
              "between them. Write a report of at most 200 words: a short title, the main topics, key methods or "
              "systems, and how they relate. Use only the given information.\n\nEntities:\n" + "\n".join(lines) +
              "\n\nRelationships:\n" + "\n".join(rel_txt))
    text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage)
    cache.put(k, out)
    return out


def main(mode):
    g = json.loads((G / "graph.json").read_text())
    r = json.loads((G / "eval.json").read_text())
    d, node_t, edge_t, T = eg.stream(0)
    edge_index = {}
    for j, (a, b) in enumerate(g["edges"]):
        edge_index.setdefault(a, []).append(j)
        edge_index.setdefault(b, []).append(j)
    stores = {}
    for m in STORES:
        labels = [np.asarray(x) for x in r[f"{m}|labels0"]]
        ev, final = events(labels, node_t)
        stores[m] = dict(ev=[(t, s) for t, s in ev if t > 0], all_ev=ev, final=final)
    lab = np.asarray(r["recompute_leiden|labels0"][-1])
    nodes = np.where(node_t >= 0)[0]
    comm = {}
    for v, c in zip(nodes, lab):
        comm.setdefault(int(c), []).append(int(v))
    fresh = [sorted(x) for x in comm.values() if len(x) >= MIN]
    need = {tuple(s) for st in stores.values() for _, s in st["all_ev"]} | {tuple(s) for s in fresh}
    for m, st in stores.items():
        print(f"{m:18s} update reports {len(st['ev']):4d}  final reports {len(st['final'])}", flush=True)
    print(f"fresh reports {len(fresh)}; unique reports to write {len(need)}", flush=True)
    if mode != "run":
        return
    cache = qr.Cache.__new__(qr.Cache)
    cache.path = G / "cache_reports.json"
    cache.data = json.loads(cache.path.read_text()) if cache.path.exists() else {}
    T_all = set(range(len(g["edges"])))

    def rep(s):
        return report(list(s), g, edge_index, T_all, cache)

    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(rep, sorted(need)))
    gq = qr.Cache("questions").get("global")["items"]
    ca, cj = qr.Cache("answers_global"), qr.Cache("judge_global")
    texts = {m: sorted({rep(tuple(s))["text"] for s in st["final"]}) for m, st in stores.items()}
    texts["fresh"] = sorted({rep(tuple(s))["text"] for s in fresh})
    answers = {}

    BATCH = 75

    def ans(a):
        """GraphRAG global search: map over batches of reports, reduce the key points into one answer."""
        name, i = a
        reps = texts[name]
        if len(reps) <= BATCH:
            blocks = [f"Report {k + 1}:\n{t}" for k, t in enumerate(reps)]
            return (name, i), qg.answer(gq[i], blocks, "entity|" + name, ca)
        points = []
        for b in range(0, len(reps), BATCH):
            chunk = reps[b:b + BATCH]
            key = hashlib.sha1(("emap|" + gq[i] + "|" + "||".join(chunk)).encode()).hexdigest()
            hit = ca.get(key)
            if not hit:
                prompt = ("From the community reports below, list at most 8 key points that help answer the question, "
                          "each with an importance score 0-100. Use only the reports.\n\n"
                          f"Question: {gq[i]}\n\n" + "\n\n".join(f"Report {k + 1}:\n{t}" for k, t in enumerate(chunk)))
                text, usage = qr.call(qr.ANS_MODEL, [{"role": "user", "content": prompt}], max_tokens=2000)
                hit = dict(text=text, usage=usage)
                ca.put(key, hit)
            points.append(hit["text"])
        blocks = [f"Key points {k + 1}:\n{t}" for k, t in enumerate(points)]
        return (name, i), qg.answer(gq[i], blocks, "entity-reduce|" + name, ca)

    with cf.ThreadPoolExecutor(16) as ex:
        answers = dict(ex.map(ans, [(n, i) for n in texts for i in range(len(gq))]))

    def cmp(a):
        name, i, order = a
        x, f = answers[(name, i)]["text"], answers[("fresh", i)]["text"]
        v = qg.compare(gq[i], *((x, f) if order == 0 else (f, x)), cj)["v"]
        mine = 1 if order == 0 else 2
        return dict(store=name, q=i, **{c: 1.0 if v[c] == mine else 0.5 if v[c] == 0 else 0.0 for c in qg.CRIT})

    with cf.ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(cmp, [(n, i, o) for n in STORES for i in range(len(gq)) for o in (0, 1)]))
    df = pd.DataFrame(rows).groupby(["store", "q"])[qg.CRIT].mean().reset_index()
    out = []
    for m in STORES:
        dd = df[df.store == m]
        tok = sum(rep(tuple(s))["usage"]["inp"] + rep(tuple(s))["usage"]["out"] for _, s in stores[m]["ev"])
        rec = dict(store=m, update_reports=len(stores[m]["ev"]), update_tokens=tok, final_reports=len(texts[m]))
        for c in qg.CRIT:
            rec[c] = float(dd[c].mean())
            rec[c + "_p"] = float(stats.wilcoxon(dd[c] - 0.5).pvalue) if (dd[c] != 0.5).any() else 1.0
        out.append(rec)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    (G / "qa_entity.json").write_text(json.dumps(dict(summary=out, rows=rows, fresh_reports=len(texts["fresh"])),
                                                 indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "plan")
