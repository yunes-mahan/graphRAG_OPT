"""A1 check: does the hybrid's drift refresh (modularity drop > delta=0.15) ever fire?

Replays the hybrid exactly as kgv2_calibrate.py does (same program, env, schedule), counts refreshes per
update, re-scores the replay with the same metric and compares Q/C with the stored held-out result.
Output: analysis/v2/a1_hybrid_refresh.json
usage: python v2/a1_hybrid_refresh.py
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kgv2_calibrate as kc  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

JOBS = [("arxiv", 2015, 2020, 0), ("collab", 2012, 2017, 0)] + list(kc.HELDOUT)


def run(job):
    stream, base, last, seed = job
    path, env = rs.SPECS["hybrid"]
    for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER"):
        os.environ.pop(k, None)
    os.environ.update(env)
    d = kc.ev.load(stream)
    node_t, edge_t, T = kc.ev.schedule(d, base, last, seed)
    maint = se.load_program(str(path))()
    import numpy as np
    edges, feat, year = d["edges"], d["feat"], d["node_year"]
    labels, fired, qdrop = [], [], []
    for t in range(T):
        new_nodes = np.where(node_t == t)[0]
        new_edges = edges[:, edge_t == t]
        all_nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        if t == 0:
            lab = maint.initialize(nodes=new_nodes.copy(), edges=new_edges.copy(),
                                   features=feat[new_nodes].astype(np.float32), years=year[new_nodes].copy())
        else:
            before = maint.refreshes
            lab = maint.update(new_nodes=new_nodes.copy(), new_edges=new_edges.copy(),
                               new_features=feat[new_nodes].astype(np.float32),
                               new_years=year[new_nodes].copy(), all_nodes=all_nodes.copy())
            fired.append(int(maint.refreshes > before))
        labels.append(np.asarray(lab, dtype=np.int64))
    key = f"{stream}|{base}|{seed}"
    stored = None
    if seed > 0 or base in (2017, 2010):
        h = json.loads((HERE.parent / "kgv2_heldout_hybrid.json").read_text())
        ref = h[key]["_ref"]
        stored = h[key]["hybrid"]
        r = kc.score_any(labels, stream, base, last, seed, ref)
        rescored = dict(Q=r["mean_modularity"], C=r["mean_relative_token_cost"])
    else:
        rescored = None
    return key, dict(delta=float(env["HYBRID_DELTA"]), refreshes=int(sum(fired)), fired_at=[i + 1 for i, f in enumerate(fired) if f],
                     n_updates=len(fired), rescored=rescored, stored=stored)


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        out = dict(p.map(run, JOBS))
    (HERE / "a1_hybrid_refresh.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(k, "refreshes", v["refreshes"], v["fired_at"], v["rescored"], v["stored"] and {x: v["stored"][x] for x in ("Q", "C")})
