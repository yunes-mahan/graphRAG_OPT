"""Cold-start placement: put a new node into a community BEFORE its links are known, by its nearest community
prototype (mean of mean-centred member embeddings). Ground truth: the community the recomputed Leiden partition
assigns using the links. Last batch of the QA stream; candidates are communities with >= 5 earlier members.
Writes cold_start.json."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import qa_plan as qp  # noqa: E402

HERE = config.RESULTS


def main():
    plan = json.loads((HERE / "qa/plan.json").read_text())
    node_t = np.load(HERE / "qa/node_t.npy")
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    nodes = np.where(node_t >= 0)[0]
    L = dict(zip(nodes.tolist(), lab))
    d, _ = qp.build_subgraph()
    embs = {"word2vec": np.asarray(d["feat"], np.float32), "text-embedding-3-small": np.load(HERE / "qa/emb_te3small.npy")}
    T = node_t.max()
    old = [v for v in nodes if node_t[v] < T]
    sizes = {}
    for v in old:
        sizes[L[v]] = sizes.get(L[v], 0) + 1
    cand = [c for c, s in sizes.items() if s >= qp.MIN_SIZE]
    new = [v for v in nodes if node_t[v] == T and L[v] in cand]
    truth = np.array([cand.index(L[v]) for v in new])
    out = dict(n_new=len(new), n_communities=len(cand), random_top1=1 / len(cand),
               majority_top1=float(max(np.mean(truth == i) for i in range(len(cand)))))
    for name, X in embs.items():
        X = X - X[old].mean(0)                       # centring uses earlier nodes only
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        C = np.stack([X[[v for v in old if L[v] == c]].mean(0) for c in cand])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        rank = np.argsort(-(X[new] @ C.T), 1)
        out[name] = dict(top1=float((rank[:, 0] == truth).mean()),
                         top3=float(np.mean([t in r[:3] for t, r in zip(truth, rank)])))
    print(json.dumps(out, indent=1))
    (HERE / "cold_start.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
