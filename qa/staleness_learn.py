"""Learned staleness trigger: predict whether a community summary really needs regeneration.

Pairs: every Jaccard-triggered regeneration (tau = 0.9) in the QA stream for the four maintainers whose
old and new summaries are both cached (qa/cache_summaries.json, written by qa_run.py).
Label  : an LLM judge (gpt-5.6-terra) reads the old and the new summary and decides whether a reader relying
         on the old one would miss or be misled about a substantive topic of the new one ("stale").
         The judge never sees the features below.
Features (all computable WITHOUT an LLM call at update time):
  struct  : Jaccard, log sizes, fraction added, fraction removed
  w2v     : drift of mean-centred centroids of the ogbn-arxiv 128-d word2vec features
  oai     : drift of mean-centred centroids of pretrained text embeddings (text-embedding-3-small, abstracts)
Model  : logistic regression, community-grouped 5-fold CV (a community never in train and test).
Report : ROC-AUC per feature set and the share of regenerations avoided when 90% of stale summaries are kept
         fresh (threshold chosen inside each training fold).
usage  : python staleness_learn.py   (resumable; caches embeddings and judge labels)
"""

import concurrent.futures as cf
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

HERE = config.RESULTS
QA = HERE / "qa"
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]
EMB_MODEL = "text-embedding-3-small"
TARGET_RECALL = 0.9
import os  # noqa: E402
LABEL = os.environ.get("LABEL", "stale")  # "stale" (judge yes/no, pre-specified) or "major" (change >= 3)


def pairs():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    cs = qr.Cache("summaries")
    out = {}
    for m in METHODS:
        store = {}
        for t, lb in enumerate(plan["methods"][m]["labels"]):
            nd = np.where((node_t >= 0) & (node_t <= t))[0]
            cur = {}
            for v, c in zip(nd, lb):
                cur.setdefault(int(c), set()).add(int(v))
            for cid, mem in cur.items():
                if len(mem) < qp.MIN_SIZE:
                    continue
                old = store.get(cid)
                if old is None:
                    store[cid] = frozenset(mem)
                elif t > 0 and len(mem & old) / len(mem | old) < 0.9:
                    ko, kn = qr.key_of(sorted(old)), qr.key_of(sorted(mem))
                    if cs.get(ko) and cs.get(kn):
                        out.setdefault((ko, kn), dict(old=sorted(old), new=sorted(mem), group=f"{m}:{cid}", t=t))
                    store[cid] = frozenset(mem)
            for cid in list(store):
                if cid not in cur or len(cur[cid]) < qp.MIN_SIZE:
                    del store[cid]
    return out


def text_embeddings():
    f = QA / "emb_te3small.npy"
    if f.exists():
        return np.load(f)
    import pandas as pd
    nodes = pd.read_csv(QA / "nodes.csv").sort_values("idx")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = [(". ".join(texts.get(int(p), ("", ""))))[:2000].strip() or "empty" for p in nodes.paper_id]
    embs = []
    for i in range(0, len(docs), 256):
        r = qr.client.embeddings.create(model=EMB_MODEL, input=docs[i:i + 256])
        embs += [d.embedding for d in r.data]
    E = np.asarray(embs, dtype=np.float32)
    np.save(f, E)
    return E


def judge(ko, kn, cache):
    key = f"{ko}|{kn}"
    hit = cache.get(key)
    if hit:
        return hit
    cs = qr.Cache("summaries")
    prompt = (
        "A retrieval system answers questions from short summaries of research communities. Summary A was "
        "written earlier; the community has since changed and Summary B describes it now.\n\n"
        f"Summary A (old):\n{cs.get(ko)['text']}\n\nSummary B (current):\n{cs.get(kn)['text']}\n\n"
        "If the system kept using A instead of B, would a user asking topic-level questions about this "
        "community miss a substantive topic, method or application of B, or be misled by content of A that no "
        "longer applies? Ignore wording, ordering and minor detail. Return JSON "
        '{"stale": true|false, "change": 1-5 (1 = same content, 5 = different community)}.')
    text, usage = qr.call(qr.JUDGE_MODEL, [{"role": "user", "content": prompt}], json_mode=True, max_tokens=400)
    v = json.loads(text)
    out = dict(stale=bool(v["stale"]), change=int(v["change"]), usage=usage)
    cache.put(key, out)
    return out


def centred(X):
    X = X - X.mean(0)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def cos_drift(old, new, X):
    a, b = X[old].mean(0), X[new].mean(0)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    P = pairs()
    print(f"pairs: {len(P)}")
    d, _ = qp.build_subgraph()
    W = centred(np.asarray(d["feat"], np.float32))
    E = centred(text_embeddings())
    cj = qr.Cache("stale_judge")
    keys = list(P)
    with cf.ThreadPoolExecutor(8) as ex:
        labs = list(ex.map(lambda k: judge(k[0], k[1], cj), keys))
    rows = []
    for (ko, kn), lab in zip(keys, labs):
        p = P[(ko, kn)]
        o, n = set(p["old"]), set(p["new"])
        rows.append(dict(
            group=p["group"], stale=int(lab["stale"]) if LABEL == "stale" else int(lab["change"] >= 3), change=lab["change"],
            jac=len(o & n) / len(o | n), log_old=np.log(len(o)), log_new=np.log(len(n)),
            added=len(n - o) / len(n), removed=len(o - n) / len(o),
            w2v=cos_drift(p["old"], p["new"], W), oai=cos_drift(p["old"], p["new"], E)))
    import pandas as pd
    df = pd.DataFrame(rows)
    y, g = df.stale.values, df.group.values
    print(f"stale: {y.sum()}/{len(y)} ({y.mean():.0%}); judge change score dist: "
          f"{df.change.value_counts().sort_index().to_dict()}")
    sets = {"jaccard only": ["jac"], "structural": ["jac", "log_old", "log_new", "added", "removed"],
            "w2v drift only": ["w2v"], "pretrained-emb drift only": ["oai"],
            "structural + w2v": ["jac", "log_old", "log_new", "added", "removed", "w2v"],
            "structural + pretrained-emb": ["jac", "log_old", "log_new", "added", "removed", "oai"],
            "all": ["jac", "log_old", "log_new", "added", "removed", "w2v", "oai"]}
    res = []
    for name, cols in sets.items():
        X = df[cols].values
        prob = np.zeros(len(y))
        saved, rec = [], []
        for tr, te in GroupKFold(5).split(X, y, g):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[tr], y[tr])
            prob[te] = clf.predict_proba(X[te])[:, 1]
            ptr = clf.predict_proba(X[tr])[:, 1]
            thr = np.quantile(ptr[y[tr] == 1], 1 - TARGET_RECALL)      # keep 90% of stale in training fold
            regen = prob[te] >= thr
            saved.append(1 - regen.mean())
            rec.append(regen[y[te] == 1].mean() if (y[te] == 1).any() else np.nan)
        auc = roc_auc_score(y, prob)
        bs = [roc_auc_score(y[i], prob[i]) for i in
              (np.random.default_rng(b).choice(len(y), len(y)) for b in range(1000)) if len(set(y[i])) == 2]
        res.append(dict(features=name, auc=auc, auc_lo=float(np.percentile(bs, 2.5)),
                        auc_hi=float(np.percentile(bs, 97.5)), regen_avoided=float(np.mean(saved)),
                        stale_recall=float(np.nanmean(rec))))
    r = pd.DataFrame(res)
    print(r.round(3).to_string(index=False))
    use = qr.Cache("stale_judge")
    inp = sum(use.get(f"{a}|{b}")["usage"]["inp"] for a, b in keys)
    print(f"judge tokens in: {inp}")
    (HERE / f"staleness_learn_{LABEL}.json").write_text(json.dumps(dict(n=len(y), stale=int(y.sum()), results=res,
                                                               rows=rows), indent=1))


if __name__ == "__main__":
    main()
