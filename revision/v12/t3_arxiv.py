"""T3 (V13): checks behind Table 4's arXiv rows (b1_streams.arxiv_stream, seeds 0-2).
  pool        : number of 2020 papers that can arrive (batches of 1,000 x 10 updates exhaust it -> mean "new" < 1,000)
  per maintainer and batch size (unchanged programs, replayed as in b1_flat_sweep):
    coverage_new : mean over updates of the share of the papers that arrived in that update which lie in a reported
                   community (>= 5 members) at that version
    coverage     : mean over versions of the share of present papers in a reported community (= Table 4 definition)
  GraphRAG delta only: new reported communities per update and their mean size (why the cost jumps with batch size:
    a delta community needs >= 5 new papers to carry a report).
Output: analysis/v12/t3_arxiv.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE)]
import b1_streams as bs  # noqa: E402
import b1_flat_sweep as fs  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

out_f = HERE / "t3_arxiv.json"
res = json.loads(out_f.read_text()) if out_f.exists() else {}
import evaluate as ev  # noqa: E402
y = ev.load("arxiv")["node_year"]
res["pool_2020"] = int((y == 2020).sum())
for B in (10, 100, 1000):
    for seed in (0, 1, 2):
        d, node_t, edge_t, T = bs.arxiv_stream(B, 10, seed)
        for m in ("recompute_leiden", "warmstart_leiden", "frontier_leiden", "graphrag_delta"):
            key = f"{B}|{seed}|{m}"
            if key in res:
                continue
            path, env = rs.SPECS[m]
            for k in fs.ENVK:
                os.environ.pop(k, None)
            os.environ.update(env)
            labels, _ = se.replay(str(path), d, node_t, edge_t, T)
            cov, cov_new, nrep, msize = [], [], [], []
            prev_ids = set()
            for t, lab in enumerate(labels):
                nodes = np.where((node_t >= 0) & (node_t <= t))[0]
                uniq, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
                rep = cnt[inv] >= 5
                cov.append(float(rep.mean()))
                if t > 0:
                    new = node_t[nodes] == t
                    cov_new.append(float(rep[new].mean()) if new.any() else float("nan"))
                    fresh = [c for c, n in zip(uniq.tolist(), cnt.tolist()) if n >= 5 and c not in prev_ids]
                    nrep.append(len(fresh))
                    msize.append(float(np.mean([n for c, n in zip(uniq.tolist(), cnt.tolist()) if c in set(fresh)])) if fresh else 0.0)
                prev_ids = {c for c, n in zip(uniq.tolist(), cnt.tolist()) if n >= 5}
            res[key] = dict(coverage=float(np.mean(cov)), coverage_new=float(np.nanmean(cov_new)),
                            new_reported_per_update=float(np.mean(nrep)), new_reported_mean_size=float(np.mean(msize)),
                            new_nodes=float(np.mean([int((node_t == t).sum()) for t in range(1, T)])))
            print(key, {k: round(v, 4) for k, v in res[key].items()}, flush=True)
            out_f.write_text(json.dumps(res, indent=1))
out_f.write_text(json.dumps(res, indent=1))
