"""Topic drift of Jaccard-triggered summary regenerations (paper: 'Why stale summaries suffice').

For each maintainer in the QA stream, every time a community's summary would be regenerated under the
Jaccard rule (tau = 0.9), compare the mean-centred text-embedding centroid of its members at the last
summary with the centroid now. References: two random 20-node sets, and two disjoint 20-node halves of
the same final community. Writes qa/semantic_drift_centred.json.
"""
import json, sys, os, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
os.chdir(config.RESULTS)
import qa_plan as qp
from qa_semantic import centroid
plan = json.loads(open("qa/plan.json").read()); node_t = np.load("qa/node_t.npy")
d, _ = qp.build_subgraph(); raw = np.asarray(d["feat"], np.float32)
feat = raw - raw.mean(0); feat /= np.linalg.norm(feat, axis=1, keepdims=True)   # mean-centred
rng = np.random.default_rng(0)
rand = [float(np.dot(centroid(rng.choice(len(feat), 20, replace=False), feat), centroid(rng.choice(len(feat), 20, replace=False), feat))) for _ in range(500)]
# same-community reference: two disjoint random halves of the same final community (recompute partition)
lab = plan["methods"]["recompute_leiden"]["labels"][-1]; nodes = np.where(node_t >= 0)[0]
comm = {}
for v, c in zip(nodes, lab): comm.setdefault(c, []).append(int(v))
same = []
for mem in comm.values():
    if len(mem) >= 40:
        p = rng.permutation(mem); same.append(float(np.dot(centroid(p[:20], feat), centroid(p[20:40], feat))))
print(f"random 20 vs random 20: median cos={np.median(rand):.3f} | two halves of the SAME community: median cos={np.median(same):.3f}")
res = {}
for m in ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]:
    labels = plan["methods"][m]["labels"]; store, cos, jac = {}, [], []
    for t, lb in enumerate(labels):
        nd = np.where((node_t >= 0) & (node_t <= t))[0]; cur = {}
        for v, c in zip(nd, lb): cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < 5: continue
            old = store.get(cid)
            if old is None: store[cid] = frozenset(mem)
            elif t > 0:
                j = len(mem & old) / len(mem | old)
                if j < 0.9:
                    cos.append(float(np.dot(centroid(old, feat), centroid(mem, feat)))); jac.append(j); store[cid] = frozenset(mem)
        for cid in list(store):
            if cid not in cur or len(cur[cid]) < 5: del store[cid]
    cos = np.array(cos)
    res[m] = dict(n=len(cos), median_cos=float(np.median(cos)), p10=float(np.percentile(cos, 10)), median_jac=float(np.median(jac)))
    print(f"{m:17s} n={len(cos):4d} median Jaccard={np.median(jac):.2f}  centred centroid cos: median={np.median(cos):.3f} p10={np.percentile(cos,10):.3f}")
json.dump(dict(per_method=res, random_median=float(np.median(rand)), same_community_halves_median=float(np.median(same))),
          open("qa/semantic_drift_centred.json", "w"), indent=1)
