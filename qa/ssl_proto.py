"""Self-supervised prototype learning for cold-start placement (SwAV-style, Caron et al. 2020).

A small adapter (MLP 1536 -> 512 -> 128, L2-normalised) is trained on top of frozen text-embedding-3-small
embeddings, using ONLY nodes and links that exist before the last batch of the QA stream (causal):
  infonce : a node and a linked node are positives, other nodes in the batch negatives (temperature 0.1)
  swav    : node and linked node are two "views"; 128 learnable prototypes; Sinkhorn-Knopp codes of one view
            are predicted from the other (swapped prediction, temperature 0.1, 3 Sinkhorn iterations)
Fixed a priori (no tuning on the test batch): 30 epochs, batch 1024, Adam lr 1e-3, 3 seeds.
Evaluation (as cold_start.py): new papers of the last batch are placed in the community with the nearest
prototype (mean adapted embedding of its earlier members); ground truth = recomputed Leiden from the links.
Writes ssl_proto.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import qa_plan as qp  # noqa: E402

HERE = config.RESULTS
EPOCHS, BATCH, LR, TEMP, K_PROTO = 30, 1024, 1e-3, 0.1, 128


class Adapter(torch.nn.Module):
    def __init__(self, d_in, k):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, 512), torch.nn.GELU(), torch.nn.Linear(512, 128))
        self.proto = torch.nn.Linear(128, k, bias=False)

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


@torch.no_grad()
def sinkhorn(scores, iters=3, eps=0.05):
    Q = torch.exp(scores / eps).T
    Q /= Q.sum()
    K, B = Q.shape
    for _ in range(iters):
        Q /= Q.sum(1, keepdim=True)
        Q /= K
        Q /= Q.sum(0, keepdim=True)
        Q /= B
    return (Q * B).T


def train(X, pairs, mode, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Adapter(X.shape[1], K_PROTO)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    Xt = torch.tensor(X)
    for ep in range(EPOCHS):
        perm = rng.permutation(len(pairs))
        for i in range(0, len(perm), BATCH):
            u, v = pairs[perm[i:i + BATCH]].T
            zu, zv = m(Xt[u]), m(Xt[v])
            if mode == "infonce":
                logits = zu @ zv.T / TEMP
                t = torch.arange(len(u))
                loss = (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t)) / 2
            else:
                with torch.no_grad():
                    w = m.proto.weight.data
                    m.proto.weight.data = F.normalize(w, dim=1)
                su, sv = m.proto(zu), m.proto(zv)
                qu, qv = sinkhorn(su.detach()), sinkhorn(sv.detach())
                loss = -0.5 * ((qu * F.log_softmax(sv / TEMP, 1)).sum(1).mean() +
                               (qv * F.log_softmax(su / TEMP, 1)).sum(1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return m(Xt).numpy()


def place(Z, old, new, L, cand):
    C = np.stack([Z[[v for v in old if L[v] == c]].mean(0) for c in cand])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    truth = np.array([cand.index(L[v]) for v in new])
    rank = np.argsort(-(Z[new] @ C.T), 1)
    return float((rank[:, 0] == truth).mean()), float(np.mean([t in r[:3] for t, r in zip(truth, rank)]))


def main():
    plan = json.loads((HERE / "qa/plan.json").read_text())
    node_t = np.load(HERE / "qa/node_t.npy")
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    nodes = np.where(node_t >= 0)[0]
    L = dict(zip(nodes.tolist(), lab))
    T = node_t.max()
    old = [v for v in nodes if node_t[v] < T]
    oldset = set(old)
    sizes = {}
    for v in old:
        sizes[L[v]] = sizes.get(L[v], 0) + 1
    cand = [c for c, s in sizes.items() if s >= qp.MIN_SIZE]
    new = [v for v in nodes if node_t[v] == T and L[v] in cand]
    d, _ = qp.build_subgraph()
    e = d["edges"].T
    pairs = np.array([(a, b) for a, b in e if a in oldset and b in oldset and a != b])
    pairs = np.concatenate([pairs, pairs[:, ::-1]])
    X = np.load(HERE / "qa/emb_te3small.npy").astype(np.float32)
    X = X - X[old].mean(0)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    res = dict(n_new=len(new), n_communities=len(cand), train_edges=len(pairs) // 2)
    res["raw"] = dict(zip(("top1", "top3"), place(X, old, new, L, cand)))
    print("raw te3-small", res["raw"], flush=True)
    for mode in ("infonce", "swav"):
        runs = []
        for seed in range(3):
            Z = train(X[:], pairs, mode, seed)
            runs.append(place(Z, old, new, L, cand))
            print(mode, seed, runs[-1], flush=True)
        r = np.array(runs)
        res[mode] = dict(top1=float(r[:, 0].mean()), top1_sd=float(r[:, 0].std()), top3=float(r[:, 1].mean()))
        # combination: concatenate raw and adapted (keeps text semantics, adds graph structure)
        runs_c = []
        for seed in range(3):
            Z = train(X[:], pairs, mode, seed)
            Zc = np.concatenate([X, Z], 1) / np.sqrt(2)
            runs_c.append(place(Zc, old, new, L, cand))
        r = np.array(runs_c)
        res[mode + "+raw"] = dict(top1=float(r[:, 0].mean()), top1_sd=float(r[:, 0].std()), top3=float(r[:, 1].mean()))
        print(mode, "summary", res[mode], "with raw", res[mode + "+raw"], flush=True)
    (HERE / "ssl_proto.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
