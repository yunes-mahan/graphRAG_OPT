"""How informative is the topic-reuse threshold (centred centroid cosine >= 0.95)?

For every community (>= 5 members) of the final recomputed partition of the QA stream, replace members so that
Jaccard(old, new) = J and measure the cosine between the centred te3-small centroids of old and new:
  random    replacements drawn uniformly from all papers
  community replacements drawn from the other communities (distinct topics, same size distribution)
J in {0.9, 0.7, 0.5, 0.3, 0.1, 0}; also the share of communities whose cosine stays >= 0.95.
Writes cosine_baseline.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
QA = HERE / "qa"


def main():
    plan = json.loads((QA / "plan.json").read_text())
    node_t = np.load(QA / "node_t.npy")
    E = np.load(QA / "emb_te3small.npy").astype(np.float64)
    E = E - E[node_t == 0].mean(0)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    lab = plan["methods"]["recompute_leiden"]["labels"][-1]
    nodes = np.where(node_t >= 0)[0]
    comm = {}
    for v, c in zip(nodes, lab):
        comm.setdefault(int(c), []).append(int(v))
    comms = [np.array(m) for m in comm.values() if len(m) >= 5]
    rng = np.random.default_rng(0)

    def cen(ix):
        c = E[ix].mean(0)
        return c / np.linalg.norm(c)

    out = {}
    for J in (0.9, 0.7, 0.5, 0.3, 0.1, 0.0):
        res = {"random": [], "community": []}
        for i, m in enumerate(comms):
            n = len(m)
            keep = int(round(2 * J * n / (1 + J)))   # |old cap new| = k, |new| = n -> J = k / (2n - k)
            kept = rng.choice(m, keep, replace=False)
            for kind in res:
                if kind == "random":
                    pool = np.setdiff1d(nodes, m)
                else:
                    pool = np.concatenate([comms[j] for j in rng.permutation(len(comms))[:5] if j != i])
                new = np.concatenate([kept, rng.choice(pool, n - keep, replace=len(pool) < n - keep)])
                res[kind].append(float(cen(m) @ cen(new)))
        out[str(J)] = {k: dict(mean=float(np.mean(v)), p10=float(np.percentile(v, 10)),
                               share_ge_095=float(np.mean(np.array(v) >= 0.95))) for k, v in res.items()}
        print(J, {k: (round(x["mean"], 3), round(x["share_ge_095"], 3)) for k, x in out[str(J)].items()}, flush=True)
    sizes = [len(m) for m in comms]
    out["communities"] = dict(n=len(comms), median_size=float(np.median(sizes)))
    (HERE / "cosine_baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
