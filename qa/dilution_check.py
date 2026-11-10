"""Dilution check for the staleness probe: can a fresh report carry the papers that joined a community?

For the probe communities (same selection as probes.py: never-refreshed stores, and reports kept at tau = 0.9 that
gained >= 4 papers), the fresh own-partition report is written from the current members (qa_run.summarise: members
in index order, at most 24k characters, about 34 papers). We measure
  input share : share of joined papers that enter the fresh report's input at all
  text match  : TF-IDF cosine between the fresh report text and each joined paper, relative to the median cosine of
                the community's older members (1 = joined papers are represented like old members)
Writes dilution_check.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

QA = HERE / "qa"
MAINT = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]


def select(plan, present, tau):
    rng = np.random.default_rng(0)
    out = []
    for m in MAINT:
        final = plan["methods"][m]["taus"][tau]["final_store"]
        cur = {}
        for v, c in zip(present, plan["methods"][m]["labels"][-1]):
            cur.setdefault(int(c), set()).add(int(v))
        cand = []
        for w in final:
            ws = set(w)
            best = max(cur.values(), key=lambda c: len(c & ws))
            joined = sorted(best - ws)
            if tau != "0.0" and len(best & ws) / len(best | ws) < float(tau):
                continue
            if len(joined) >= 4:
                cand.append((sorted(best), joined, sorted(best & ws)))
        out += [(m, *cand[i]) for i in rng.permutation(len(cand))[:25]]
    return out


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    present = np.where(node_t >= 0)[0]
    nodes = pd.read_csv(QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    cs = json.loads((QA / "cache_summaries.json").read_text())
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(
        [docs[v][0] + " " + docs[v][1] for v in present])
    res = {}
    for tau, label in (("0.0", "never"), ("0.9", "tau09")):
        rows = []
        for m, cur, joined, old in select(plan, present, tau):
            shown, used = set(), 0
            for v in sorted(cur):
                s = len(f"- {docs[v][0]}: {docs[v][1][:600]}")
                if used + s > qr.MAX_CHARS:
                    break
                shown.add(v)
                used += s
            rep = cs[qr.key_of(cur)]["text"]
            R = vec.transform([rep])
            sim = lambda vs: (vec.transform([docs[v][0] + " " + docs[v][1] for v in vs]) @ R.T).toarray().ravel()  # noqa: E731
            sj, so = sim(joined), sim(old)
            rows.append(dict(maintainer=m, size=len(cur), joined=len(joined), joined_share=len(joined) / len(cur),
                             input_share=float(np.mean([v in shown for v in joined])),
                             rel_match=float(np.median(sj) / max(np.median(so), 1e-9))))
        df = pd.DataFrame(rows)
        res[label] = dict(n=len(df), median_size=float(df["size"].median()), joined_share=float(df.joined_share.mean()),
                          input_share=float(df.input_share.mean()), rel_match=float(df.rel_match.median()),
                          rows=rows)
        print(label, {k: round(v, 3) for k, v in res[label].items() if k != "rows"}, flush=True)
    (HERE / "dilution_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
