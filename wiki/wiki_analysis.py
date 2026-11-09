"""Evolving-text stream (Wikipedia ML/AI articles, yearly snapshots 2017-2025): when do summaries go stale?

Unlike the arXiv stream, article TEXT changes while membership may not. Pipeline:
 1. graph per snapshot = undirected wikilinks inside the article set (links can also disappear)
 2. communities: Leiden (modularity) at the first snapshot, then warm-started Leiden on every later snapshot
    (previous labels as initial membership, new pages singletons); identities kept by one-to-one majority overlap
 3. a summary (gpt-5.6-luna, same prompt style as qa_run.py) for every community with >= 5 members at every
    snapshot, written from the members' CURRENT titles + lead texts (<= 24k characters)
 4. label: gpt-5.6-terra rates last year's summary vs this year's summary of the same community (1-5);
    substantive = rating >= 3 (same judge prompt as delta_summary.rate)
 5. predict substantive change from signals available WITHOUT an LLM call:
      membership : Jaccard, log sizes, added, removed            (all the arXiv trigger could use)
      links      : Jaccard of the community's internal edge sets
      text-diff  : mean character-level change of members' lead texts (difflib), share of members edited
      embedding  : drift of the centroid of members' text-embedding-3-small embeddings (current texts, mean-centred
                   per snapshot pool), and mean per-member embedding drift
    logistic regression, community-grouped 5-fold CV, ROC-AUC with bootstrap CI
Writes wiki/results.json.
"""

import concurrent.futures as cf
import difflib
import hashlib
import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qa"))
import config  # noqa: E402
HERE = config.RESULTS / "wiki"
HERE.mkdir(parents=True, exist_ok=True)
import delta_summary as ds  # noqa: E402
import qa_run as qr  # noqa: E402

MIN_SIZE = 5
MODQ = {}
MAX_CHARS = 24000


class Cache(qr.Cache):
    def __init__(self, name):
        self.path = HERE / f"cache_{name}.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}


def communities(stream):
    years = [str(y) for y in stream["years"]]
    labels, prev = {}, None
    for y in years:
        snap = stream["snapshots"][y]
        nodes = sorted(int(k) for k in snap)
        pos = {v: i for i, v in enumerate(nodes)}
        edges = {(min(pos[int(a)], pos[b]), max(pos[int(a)], pos[b])) for a, s in snap.items() for b in s["links"]
                 if b in pos and b != int(a)}
        g = ig.Graph(n=len(nodes), edges=sorted(edges))
        if prev is None:
            init = None
        else:
            nxt = max(prev.values()) + 1
            init = []
            for v in nodes:
                if v in prev:
                    init.append(prev[v])
                else:
                    init.append(nxt)
                    nxt += 1
            uniq = {c: i for i, c in enumerate(sorted(set(init)))}
            init = [uniq[c] for c in init]
        part = la.find_partition(g, la.ModularityVertexPartition, initial_membership=init, seed=0)
        memb = {v: part.membership[i] for i, v in enumerate(nodes)}
        # persistent identities: one-to-one majority overlap with previous communities
        if prev is not None:
            prevc, curc = {}, {}
            for v, c in prev.items():
                prevc.setdefault(c, set()).add(v)
            for v, c in memb.items():
                curc.setdefault(c, set()).add(v)
            pairs = sorted(((len(m & prevc[p]), c, p) for c, m in curc.items() for p in
                            {prev[v] for v in m if v in prev}), reverse=True)
            used_c, used_p, mp = set(), set(), {}
            for ov, c, p in pairs:
                if c not in used_c and p not in used_p:
                    mp[c] = p
                    used_c.add(c)
                    used_p.add(p)
            nxt = max(max(prev.values()), max(mp.values(), default=0)) + 1
            for c in curc:
                if c not in mp:
                    mp[c] = nxt
                    nxt += 1
            memb = {v: mp[c] for v, c in memb.items()}
        labels[y] = memb
        prev = memb
        q = g.modularity(part.membership)
        MODQ[y] = q
        print(f"{y}: {len(nodes)} pages {g.ecount()} edges, {len(set(memb.values()))} communities, Q~{q:.3f}",
              flush=True)
    return labels


def summarise(members, texts, titles, cache):
    body, used = [], 0
    for v in sorted(members):
        s = f"- {titles[v]}: {texts[v][:600]}"
        if used + len(s) > MAX_CHARS:
            break
        body.append(s)
        used += len(s)
    k = hashlib.sha1("\n".join(body).encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    prompt = ("You write community reports for a GraphRAG index over Wikipedia articles on AI and machine learning. "
              f"Below are the articles of one community ({len(members)} articles, {len(body)} shown) with their "
              "current lead sections. Write a report of at most 200 words: a short title, the main topics, key "
              "methods or systems, and current developments. Use only the given articles.\n\n" + "\n".join(body))
    text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": prompt}], max_tokens=1500)
    out = dict(text=text, usage=usage)
    cache.put(k, out)
    return out


def embed(texts_list):
    """text-embedding-3-small (512 dims), cached by text hash in emb_cache.npz."""
    f = HERE / "emb_cache.npz"
    cache = dict(np.load(f)) if f.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts_list]
    todo = sorted({(k, t) for k, t in zip(keys, texts_list) if k not in cache})
    for i in range(0, len(todo), 256):
        chunk = todo[i:i + 256]
        r = qr.client.embeddings.create(model="text-embedding-3-small", dimensions=512,
                                        input=[t[:6000] or "empty" for _, t in chunk])
        for (k, _), d in zip(chunk, r.data):
            cache[k] = np.asarray(d.embedding, dtype=np.float32)
        if (i // 256) % 20 == 19 or i + 256 >= len(todo):
            np.savez(f, **cache)
    return np.stack([cache[k] for k in keys])


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    stream = json.loads((HERE / "stream.json").read_text())
    titles = stream["titles"]
    years = [str(y) for y in stream["years"]]
    labels = communities(stream)
    T = {y: {int(k): v["text"] for k, v in stream["snapshots"][y].items()} for y in years}
    L = {y: {int(k): set(v["links"]) for k, v in stream["snapshots"][y].items()} for y in years}

    # embeddings of every (page, year) text, mean-centred per year
    E = {}
    for y in years:
        ids = sorted(T[y])
        X = embed([titles[v] + ". " + T[y][v] for v in ids])
        X = X - X.mean(0)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        E[y] = dict(zip(ids, X))

    # summaries for every community at every snapshot
    cs = Cache("summaries")
    comm = {}
    for y in years:
        for v, c in labels[y].items():
            comm.setdefault((y, c), set()).add(v)
    comm = {k: m for k, m in comm.items() if len(m) >= MIN_SIZE}
    with cf.ThreadPoolExecutor(8) as ex:
        sums = dict(zip(comm, ex.map(lambda k: summarise(comm[k], T[k[0]], titles, cs), comm)))
    print(f"summaries: {len(sums)}", flush=True)
    (HERE / "communities.json").write_text(json.dumps(dict(
        labels={y: {str(v): int(c) for v, c in labels[y].items()} for y in years},
        summaries={f"{y}|{c}": sums[(y, c)]["text"] for (y, c) in sums})))

    # pairs (same community id, consecutive snapshots)
    pairs = [((y0, c), (y1, c)) for y0, y1 in zip(years, years[1:]) for (yy, c) in comm if yy == y1
             and (y0, c) in comm]
    cr = Cache("rate")
    with cf.ThreadPoolExecutor(8) as ex:
        rates = list(ex.map(lambda p: ds.rate(sums[p[0]]["text"], sums[p[1]]["text"], cr)["change"], pairs))

    def internal(y, m):
        return {(a, b) for a in m for b in L[y].get(a, ()) if b in m and a < b} | \
               {(b, a) for a in m for b in L[y].get(a, ()) if b in m and b < a}

    rows = []
    for (a, b), r in zip(pairs, rates):
        (y0, c), (y1, _) = a, b
        o, n = comm[a], comm[b]
        both = sorted(o & n)
        ratios = [1 - difflib.SequenceMatcher(None, T[y0][v], T[y1][v], autojunk=False).quick_ratio() for v in both]
        e0 = np.mean([E[y0][v] for v in o], 0)
        e1 = np.mean([E[y1][v] for v in n], 0)
        ie0, ie1 = internal(y0, o), internal(y1, n)
        rows.append(dict(
            group=str(c), year=y1, change=r, substantive=int(r >= 3),
            jac=len(o & n) / len(o | n), log_old=np.log(len(o)), log_new=np.log(len(n)),
            added=len(n - o) / len(n), removed=len(o - n) / len(o),
            link_jac=len(ie0 & ie1) / max(len(ie0 | ie1), 1),
            text_diff=float(np.mean(ratios)) if ratios else 0.0,
            edited=float(np.mean([x > 0.01 for x in ratios])) if ratios else 0.0,
            emb_centroid=float(e0 @ e1 / max(np.linalg.norm(e0) * np.linalg.norm(e1), 1e-12)),
            emb_member=float(np.mean([E[y0][v] @ E[y1][v] for v in both])) if both else 1.0))
    df = pd.DataFrame(rows)
    y_, g = df.substantive.values, df.group.values
    print(f"pairs {len(df)}, substantive {y_.sum()} ({y_.mean():.0%}); identical membership pairs: "
          f"{(df.jac == 1).sum()} of which substantive {df[df.jac == 1].substantive.sum()}", flush=True)
    MEM = ["jac", "log_old", "log_new", "added", "removed"]
    sets = {"membership (arXiv trigger)": MEM, "+ links": MEM + ["link_jac"],
            "+ text diff (no embedding)": MEM + ["link_jac", "text_diff", "edited"],
            "+ embeddings": MEM + ["link_jac", "emb_centroid", "emb_member"],
            "all": MEM + ["link_jac", "text_diff", "edited", "emb_centroid", "emb_member"],
            "embeddings only": ["emb_centroid", "emb_member"]}
    res = []
    for name, cols in sets.items():
        X = df[cols].values
        prob = np.zeros(len(y_))
        for tr, te in GroupKFold(5).split(X, y_, g):
            prob[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(
                X[tr], y_[tr]).predict_proba(X[te])[:, 1]
        bs = []
        for b in range(1000):
            i = np.random.default_rng(b).choice(len(y_), len(y_))
            if len(set(y_[i])) == 2:
                bs.append(roc_auc_score(y_[i], prob[i]))
        res.append(dict(features=name, auc=roc_auc_score(y_, prob), lo=float(np.percentile(bs, 2.5)),
                        hi=float(np.percentile(bs, 97.5))))
    r = pd.DataFrame(res)
    print(r.round(3).to_string(index=False))
    use = {}
    for c in (cs, cr):
        for v in c.data.values():
            u = v["usage"]
            use.setdefault(u["model"], [0, 0])
            use[u["model"]][0] += u["inp"]
            use[u["model"]][1] += u["out"]
    print("tokens:", use)
    (HERE / "results.json").write_text(json.dumps(dict(results=res, rows=rows, n_pages={y: len(T[y]) for y in years}, modularity=MODQ,
                                                       n_links={y: sum(len(v) for v in L[y].values()) for y in years},
                                                       tokens=use), indent=1, default=float))


if __name__ == "__main__":
    main()
