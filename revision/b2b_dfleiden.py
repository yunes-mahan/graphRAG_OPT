"""B2b: Sahu's own Dynamic Frontier Leiden (C++/OpenMP, github.com/puzzlef/leiden-communities-openmp-dynamic @ 28c304f,
unmodified; driver v2/dfleiden_driver/df_driver.cxx, single thread) next to our frontier-restricted Leiden, on the
Table 4 streams: development order 0 of arXiv-15 / collab-12 and the 24 held-out orders.

Sahu's labels carry no persistent identity, so they are mapped to previous identifiers by the SAME one-to-one
majority-overlap function our frontier, warm-start and recompute maintainers use (programs/hybrid_refresh.py
map_to_existing_ids), then scored with the same metric and the same reference tokens as Table 4
(kgv2_calibrate.score_any; held-out references from kgv2_heldout.json, development references from
tasks/kgversion2/resources/reference_tokens.json).
Outputs: analysis/v2/kgv2_heldout_df_leiden_sahu.json (same layout as analysis/kgv2_heldout_*.json),
         analysis/v2/kgv2_dev_df_leiden_sahu.json
usage: python v2/b2b_dfleiden.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "programs"))
import kgv2_calibrate as kc  # noqa: E402
from hybrid_refresh import map_to_existing_ids  # noqa: E402

WORK = HERE / "dfleiden_work"
WORK.mkdir(exist_ok=True)
DEV = [("arxiv", 2015, 2020, 0), ("collab", 2012, 2017, 0)]
JOBS = DEV + list(kc.HELDOUT)
NAME = "df_leiden_sahu"


def tag(j):
    return f"{j[0]}_{j[1]}_{j[3]}"


def export(job):
    stream, base, last, seed = job
    d = kc.ev.load(stream)
    node_t, edge_t, T = kc.ev.schedule(d, base, last, seed)
    E = d["edges"]
    with open(WORK / f"{tag(job)}.in", "w") as f:
        f.write(f"{len(node_t)} {T}\n")
        for t in range(T):
            e = E[:, edge_t == t]
            f.write(f"{e.shape[1]}\n")
            f.write("".join(f"{a} {b}\n" for a, b in e.T))
    return node_t, T


if __name__ == "__main__":
    meta = {tag(j): export(j) for j in JOBS}
    script = " && ".join(f"/drv/df_driver < /work/{tag(j)}.in > /work/{tag(j)}.out 2> /work/{tag(j)}.err" for j in JOBS)
    V2 = HERE
    subprocess.run(["docker", "run", "--rm", "-v", f"{V2 / 'dfleiden_driver'}:/drv:ro", "-v", f"{WORK}:/work",
                    "docker.io/library/rust:1.83", "sh", "-c", script], check=True)
    held = json.loads((HERE.parent / "kgv2_heldout.json").read_text())
    dev_ref = json.loads((Path(os.environ.get("GCM_TASK", Path(__file__).resolve().parents[2] / "tasks" / "kgversion2")) / "resources" / "reference_tokens.json").read_text())
    out_h, out_d = {}, {}
    for j in JOBS:
        stream, base, last, seed = j
        node_t, T = meta[tag(j)]
        raw = [np.array(line.split(), dtype=np.int64) for line in (WORK / f"{tag(j)}.out").read_text().splitlines()]
        assert len(raw) == T
        secs = sum(float(x) for x in re.findall(r"time ([0-9.]+) ms", (WORK / f"{tag(j)}.err").read_text())) / 1e3
        label, next_id, labels = {}, 0, []
        for t in range(T):
            nodes = [int(v) for v in np.where((node_t >= 0) & (node_t <= t))[0]]
            memb = raw[t][nodes]
            if t == 0:
                _, comp = np.unique(memb, return_inverse=True)
                label = {v: int(c) for v, c in zip(nodes, comp)}
                next_id = int(comp.max()) + 1
            else:
                mapping, next_id = map_to_existing_ids(nodes, memb.tolist(), label, next_id)
                label = {v: mapping[int(c)] for v, c in zip(nodes, memb)}
            labels.append(np.asarray([label[v] for v in nodes], dtype=np.int64))
        key = f"{stream}|{base}|{seed}"
        if j in DEV:
            r = kc.score_any(labels, stream, base, last, seed, dev_ref[stream])
            out_d[key] = dict(Q=r["mean_modularity"], C=r["mean_relative_token_cost"], seconds=secs)
        else:
            r = kc.score_any(labels, stream, base, last, seed, held[key]["_ref"])
            out_h[key] = {NAME: dict(Q=r["mean_modularity"], C=r["mean_relative_token_cost"], seconds=secs)}
        print(key, out_d.get(key) or out_h[key][NAME], flush=True)
    cal = json.loads((HERE.parent / "kgv2_calibration.json").read_text())
    lam = cal["lambda_"]
    out_d["combined"] = float(np.mean([out_d[k]["Q"] - lam * out_d[k]["C"] for k in out_d]))
    (HERE / f"kgv2_heldout_{NAME}.json").write_text(json.dumps(out_h, indent=1))
    (HERE / f"kgv2_dev_{NAME}.json").write_text(json.dumps(out_d, indent=1))
    print("dev combined", out_d["combined"])
