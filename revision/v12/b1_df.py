"""B1 (V12): Sahu's DF Leiden (authors' code, v2/dfleiden_driver, single thread, container) on the small-batch arXiv
streams of b1_streams.arxiv_stream (B in {1000, 100, 10}, N = 10, seeds 0-2).  Labels are mapped to persistent ids
with the same map_to_existing_ids as all maintainers and scored with b1_flat_sweep.score (tau 0.9 / never, Q, coverage)
against the recompute_leiden reference sizes of the same stream (recomputed here with the unchanged program).
Output: merged into analysis/v12/b1/flat_arxiv.json under keys "B|seed|df_leiden_sahu".
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(HERE), str(A / "programs")]
import b1_streams as bs  # noqa: E402
import b1_flat_sweep as fs  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402
from hybrid_refresh import map_to_existing_ids  # noqa: E402

W = HERE / "b1" / "df_work"
W.mkdir(parents=True, exist_ok=True)
JOBS = [(B, s) for B in (1000, 100, 10) for s in (0, 1, 2)]

if __name__ == "__main__":
    meta = {}
    for B, s in JOBS:
        d, node_t, edge_t, T = bs.arxiv_stream(B, 10, s)
        E = d["edges"]
        with open(W / f"b{B}_s{s}.in", "w") as f:
            f.write(f"{len(node_t)} {T}\n")
            for t in range(T):
                e = E[:, edge_t == t]
                f.write(f"{e.shape[1]}\n" + "".join(f"{a} {b}\n" for a, b in e.T))
        meta[(B, s)] = (d, node_t, edge_t, T)
    script = " && ".join(f"/drv/df_driver < /work/b{B}_s{s}.in > /work/b{B}_s{s}.out 2>/dev/null" for B, s in JOBS)
    subprocess.run(["docker", "run", "--rm", "-v", f"{A / 'v2' / 'dfleiden_driver'}:/drv:ro", "-v", f"{W}:/work",
                    "docker.io/library/rust:1.83", "sh", "-c", script], check=True)
    out_f = HERE / "b1" / "flat_arxiv.json"
    res = json.loads(out_f.read_text())
    for B, s in JOBS:
        d, node_t, edge_t, T = meta[(B, s)]
        raw = [np.array(l.split(), dtype=np.int64) for l in (W / f"b{B}_s{s}.out").read_text().splitlines()]
        label, nxt, labels = {}, 0, []
        for t in range(T):
            nodes = [int(v) for v in np.where((node_t >= 0) & (node_t <= t))[0]]
            memb = raw[t][nodes]
            if t == 0:
                _, comp = np.unique(memb, return_inverse=True)
                label, nxt = {v: int(c) for v, c in zip(nodes, comp)}, int(comp.max()) + 1
            else:
                mp, nxt = map_to_existing_ids(nodes, memb.tolist(), label, nxt)
                label = {v: mp[int(c)] for v, c in zip(nodes, memb)}
            labels.append(np.asarray([label[v] for v in nodes], dtype=np.int64))
        r = fs.score(labels, d, node_t, edge_t)
        path, env = rs.SPECS["recompute_leiden"]
        ref_labels, _ = se.replay(str(path), d, node_t, edge_t, T)
        ref = fs.score(ref_labels, d, node_t, edge_t)["full"]
        res[f"{B}|{s}|df_leiden_sahu"] = dict(
            C09=float(np.mean([fs.tok(x) / fs.tok(f) for x, f in zip(r["rg9"], ref[1:])])),
            Cnever=float(np.mean([fs.tok(x) / fs.tok(f) for x, f in zip(r["rg0"], ref[1:])])),
            Q=float(np.mean(r["Q"])), coverage=float(np.mean(r["cov"])),
            new_nodes=float(np.mean([int((node_t == t).sum()) for t in range(1, T)])))
        print(B, s, res[f"{B}|{s}|df_leiden_sahu"], flush=True)
    out_f.write_text(json.dumps(res, indent=1))
