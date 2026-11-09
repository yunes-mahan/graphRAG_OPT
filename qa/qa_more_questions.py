"""Add 80 questions to the QA benchmark (40 -> 120) with the same generator, model and prompt as qa_run.py.

Seed papers are drawn with a NEW random seed from 2019-2020 papers that were not seeds or context papers of the
existing 40 questions. The existing questions are kept unchanged (their cached answers stay valid).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

N_NEW = 80


def main():
    nodes = pd.read_csv(qr.QA / "nodes.csv")
    texts = qp.load_texts(nodes.paper_id.values)
    docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
    years = nodes.sort_values("idx").year.values
    cq = qr.Cache("questions")
    cur = cq.get("questions")
    items, usage = cur["items"], cur["usage"]
    if len(items) >= 40 + N_NEW:
        print("already extended:", len(items))
        return
    used = {it["seed"] for it in items} | {p for it in items for p in it["papers"]}
    idx = np.arange(len(docs))
    X = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True).fit_transform(
        [docs[i][0] + " " + docs[i][1] for i in idx])
    rng = np.random.default_rng(11)
    pool = np.array([i for i in idx[years >= 2019] if i not in used])
    new = []
    for s in rng.permutation(pool):
        if len(new) >= N_NEW:
            break
        if s in used:
            continue
        nb = np.argsort(-linear_kernel(X[s], X).ravel())[:8]
        ctx = "\n".join(f"[{j}] {docs[j][0]}: {docs[j][1][:700]}" for j in nb)
        prompt = ("From the papers below, write ONE broad, topic-level question that a user of a research "
                  "knowledge base might ask (e.g. 'What are the main approaches and trends in ...?'), and exactly "
                  "4 short key points that a good answer should mention. Each key point must be a general theme, "
                  "method family or trend that is supported by at least two of the papers; do NOT use numbers, "
                  "dataset names, system names or details of a single paper. Return JSON "
                  "{\"question\": str, \"key_points\": [str, str, str, str]}.\n\n" + ctx)
        text, u = qr.call(qr.Q_MODEL, [{"role": "user", "content": prompt}], json_mode=True)
        usage.append(u)
        try:
            j = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(j.get("question"), str) and len(j.get("key_points", [])) == 4 and \
                j["question"] not in {it["question"] for it in items + new}:
            new.append(dict(seed=int(s), papers=[int(x) for x in nb], batch=2, **j))
            used |= {int(s)} | {int(x) for x in nb}
    cq.put("questions", dict(items=items + new, usage=usage))
    print(f"questions: {len(items)} + {len(new)} = {len(items) + len(new)}")


if __name__ == "__main__":
    main()
