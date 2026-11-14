"""T4 (V13): Sahu's Dynamic Frontier Leiden (authors' code, v2/dfleiden_driver, single thread, rust:1.83 container) on
the full ogbl-citation2 stream of scale_citation2.py (same dataset, schedule seed 0, same recompute reference from
scale.json).  Labels are mapped to persistent identifiers with map_to_existing_ids, then scored with wd_eval.score as
all other citation2 cells.  Output: analysis/v12/t4/scale_df_leiden_sahu.json
usage: python v12/t4_cit2_df.py export | run | score
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata"), str(A / "programs")]
import scale_citation2 as sc  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402
from hybrid_refresh import map_to_existing_ids  # noqa: E402

W = HERE / "t4" / "df_cit2"
OUT = HERE / "t4" / "scale_df_leiden_sahu.json"


def stream():
    d = sc.dataset(1.0)
    node_t, edge_t, T = se.schedule(d, 0)
    return d, node_t, edge_t, T


mode = sys.argv[1]
if mode == "export":
    W.mkdir(parents=True, exist_ok=True)
    d, node_t, edge_t, T = stream()
    E = d["edges"]
    with open(W / "cit2.in", "w") as f:
        f.write(f"{len(node_t)} {T}\n")
        for t in range(T):
            e = E[:, edge_t == t]
            f.write(f"{e.shape[1]}\n")
            np.savetxt(f, e.T, fmt="%d")
elif mode == "run":
    subprocess.run(["docker", "run", "--rm", "-v", f"{A / 'v2' / 'dfleiden_driver'}:/drv:ro", "-v", f"{W}:/work",
                    "docker.io/library/rust:1.83", "sh", "-c", "/drv/df_driver < /work/cit2.in > /work/cit2.out 2> /work/cit2.err"],
                   check=True)
elif mode == "score":
    d, node_t, edge_t, T = stream()
    label, nxt, labels = {}, 0, []
    with open(W / "cit2.out") as f:
        for t, line in enumerate(f):
            raw = np.array(line.split(), dtype=np.int64)
            nodes = np.where((node_t >= 0) & (node_t <= t))[0]
            memb = raw[nodes]
            if t == 0:
                _, comp = np.unique(memb, return_inverse=True)
                label, nxt = dict(zip(nodes.tolist(), comp.tolist())), int(comp.max()) + 1
            else:
                mp, nxt = map_to_existing_ids(nodes.tolist(), memb.tolist(), label, nxt)
                label = {v: mp[int(c)] for v, c in zip(nodes.tolist(), memb.tolist())}
            labels.append(np.asarray([label[v] for v in nodes.tolist()], dtype=np.int64))
    ref = json.loads((A / "scale.json").read_text())["1.0|_ref"]
    r = wd.score(labels, d, node_t, edge_t, ref)
    res = {"1.0|df_leiden_sahu": dict(Q=r["Q"], C=r["C"], score=r["Q"] - sc.LAM * r["C"])}
    print(res)
    OUT.write_text(json.dumps(res, indent=1))
