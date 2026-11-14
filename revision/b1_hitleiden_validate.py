"""B1: validate the UNOFFICIAL third-party HIT-Leiden implementation (github.com/randomvariable/hit-leiden,
commit 9247424, Rust) before using it.  There is no official implementation: the authors' repository
github.com/swordzlim/HIT-Leiden (cited as [53] in their paper) contains only the full-version PDF.

Test case: arXiv-15, order 0: initial run on version 0, one incremental update with batch 1 (the first real batch of
the benchmark).  Checks, all on the library's public API (CommunityState::initial / ::update):
  1. flat partition: modularity recomputed with igraph equals the library's; compared with igraph Leiden (seed 0)
     on the same graph, from scratch (plausibility, cf. HIT-Leiden paper: maintained modularity ~ static Leiden);
  2. hierarchy: number of levels, nesting (every finer community inside one coarser community), and whether the
     coarser levels are updated at all after the update (HIT-Leiden maintains every level).
Build/run: Rust 1.83 container (podman, via the docker shim), the third-party source mounted read-only.
Output: analysis/v2/b1_hitleiden_validate.json
"""
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("GCM_TASK", Path(__file__).resolve().parents[2] / "tasks" / "kgversion2"))
os.environ["TASK_SOURCE_DIR"] = str(TASK)
sys.path.insert(0, str(TASK))
import evaluate as ev  # noqa: E402

SRC = Path(sys.argv[1])            # third-party source (read-only mount)
WORK = Path(sys.argv[2])           # scratch dir for data + cargo target
CRATE = HERE / "hitleiden_check"
WORK.mkdir(parents=True, exist_ok=True)

d = ev.load("arxiv")
node_t, edge_t, T = ev.schedule(d, 2015, 2020, 0)
nodes1 = np.where((node_t >= 0) & (node_t <= 1))[0]
idx = -np.ones(len(node_t), dtype=np.int64)
idx[nodes1] = np.arange(len(nodes1))          # version-0 nodes come first? not required: ids are global compact ids
E = d["edges"]
e0 = E[:, edge_t == 0]
e1 = E[:, edge_t == 1]
n0 = int(idx[np.where(node_t == 0)[0]].max()) + 1
n1 = len(nodes1)
# compact ids such that version-0 nodes are 0..n0-1 (library grows node ids by max index)
order = np.concatenate([np.where(node_t == 0)[0], np.where(node_t == 1)[0]])
idx[order] = np.arange(len(order))
n0 = int((node_t == 0).sum())
for name, ee, n in (("v0.txt", e0, n0), ("delta.txt", e1, n1)):
    with open(WORK / name, "w") as f:
        f.write(f"{n}\n")
        for a, b in ee.T:
            f.write(f"{idx[a]} {idx[b]}\n")

cmd = ["docker", "run", "--rm", "-v", f"{SRC}:/hit:ro", "-v", f"{CRATE}:/check", "-v", f"{WORK}:/work",
       "-e", "CARGO_TARGET_DIR=/work/target", "-e", "CARGO_HOME=/work/cargo", "-w", "/check", "docker.io/library/rust:1.83",
       "sh", "-c", "cargo build --release -q 2>/work/build.log && /work/target/release/hitcheck /work/v0.txt /work/delta.txt > /work/out.tsv 2>/work/run.log"]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout, r.stderr, (WORK / "build.log").read_text()[-3000:] if (WORK / "build.log").exists() else "")
    raise SystemExit("build/run failed")

rows = {}
for line in (WORK / "out.tsv").read_text().splitlines():
    k, *v = line.split("\t")
    rows[k] = v
q0_lib, q1_lib, t_init, t_upd, nlev0, nlev1 = map(float, rows["meta"])
flat0 = np.array(rows["flat0"][0].split(), dtype=np.int64)
flat1 = np.array(rows["flat1"][0].split(), dtype=np.int64)
lv0 = [np.array(rows[f"lv0_{i}"][0].split(), dtype=np.int64) for i in range(int(nlev0))]
lv1 = [np.array(rows[f"lv1_{i}"][0].split(), dtype=np.int64) for i in range(int(nlev1))]

g0 = ig.Graph(n=n0, edges=[(int(idx[a]), int(idx[b])) for a, b in e0.T])
g1 = ig.Graph(n=n1, edges=[(int(idx[a]), int(idx[b])) for a, b in E[:, edge_t <= 1].T])


def nested(fine, coarse):
    m = defaultdict(set)
    for f, c in zip(fine, coarse):
        m[f].add(c)
    return float(np.mean([len(s) == 1 for s in m.values()]))


def q(g, lab):
    _, c = np.unique(lab[: g.vcount()], return_inverse=True)
    return g.modularity(c.tolist())


ref0 = la.find_partition(g0, la.ModularityVertexPartition, seed=0)
ref1 = la.find_partition(g1, la.ModularityVertexPartition, seed=0)
out = dict(
    implementation="github.com/randomvariable/hit-leiden @ 9247424 (unofficial, third party)",
    test="arXiv-15 order 0: initial = version 0, update = batch 1",
    n_nodes_v0=n0, n_edges_v0=int(e0.shape[1]), n_nodes_v1=n1, n_edges_delta=int(e1.shape[1]),
    seconds_initial=t_init, seconds_update=t_upd,
    modularity_lib_reported=dict(initial=q0_lib, update=q1_lib),
    modularity_igraph_recomputed=dict(initial=q(g0, flat0), update=q(g1, flat1)),
    modularity_static_leiden_seed0=dict(initial=ref0.modularity, update=ref1.modularity),
    communities=dict(initial=int(len(set(flat0))), update=int(len(set(flat1))),
                     static_initial=len(ref0), static_update=len(ref1)),
    hierarchy_levels=dict(initial=int(nlev0), update=int(nlev1),
                          sizes_initial=[int(len(set(l))) for l in lv0], sizes_update=[int(len(set(l))) for l in lv1]),
    nesting_share_initial=[nested(lv0[i], lv0[i + 1]) for i in range(len(lv0) - 1)],
    nesting_share_update=[nested(lv1[i], lv1[i + 1]) for i in range(len(lv1) - 1)],
    # does the update change the coarser levels on the version-0 nodes?
    coarse_levels_unchanged_on_old_nodes=[bool(np.array_equal(lv0[i][:n0], lv1[i][:n0])) for i in range(1, min(len(lv0), len(lv1)))],
    new_nodes_singletons_in_coarse_levels=[float(np.mean(lv1[i][n0:] == np.arange(n0, n1))) for i in range(1, len(lv1))],
)
(HERE / "b1_hitleiden_validate.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
