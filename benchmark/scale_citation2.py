"""Scalability study on ogbl-citation2 (2,927,963 papers, ~30.4M citations, publication years).

Graphs: uniform random node samples of 25% and 50% (induced subgraphs, seed 0) and the full graph.
Stream: V0 = papers up to 2016, then 2017-2019 in 3 random batches per year (10 versions, schedule seed 0);
a paper and its citations arrive together.
For every maintainer (run in its own process): wall-clock seconds per update, peak resident memory, mean
modularity and the token-level cost relative to recomputation (as in tasks/kgversion2; lambda 0.31).
usage: python scale_citation2.py prepare | run FRACTION METHOD | report
"""

import gzip
import json
import os
import resource
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
D = HERE / "data" / "citation2"
OUT = HERE / "scale.json"
BASE, LAST, LAM = 2016, 2019, 0.31
METHODS = ["recompute_leiden", "lazy", "warmstart_leiden", "frontier_leiden", "priced_warm", "priced_frontier"]


def prepare():
    z = zipfile.ZipFile(D / "citation-v2.zip")
    names = z.namelist()
    ef = [n for n in names if n.endswith("raw/edge.csv.gz")][0]
    yf = [n for n in names if n.endswith("raw/node_year.csv.gz")][0]
    import pandas as pd
    with z.open(ef) as f:
        e = pd.read_csv(gzip.open(f), header=None, dtype=np.int32).values.T
    with z.open(yf) as f:
        year = pd.read_csv(gzip.open(f), header=None).values.ravel().astype(np.int16)
    lo, hi = np.minimum(e[0], e[1]), np.maximum(e[0], e[1])
    keep = lo != hi
    key = np.unique(lo[keep].astype(np.int64) * len(year) + hi[keep])
    edges = np.stack([key // len(year), key % len(year)]).astype(np.int32)
    np.savez(D / "citation2.npz", edges=edges, year=year)
    print(f"nodes {len(year)}, undirected edges {edges.shape[1]}, years {year.min()}-{year.max()}")


def dataset(frac):
    z = np.load(D / "citation2.npz")
    edges, year = z["edges"].astype(np.int64), z["year"].astype(np.int64)
    if frac < 1:
        rng = np.random.default_rng(0)
        keep = rng.random(len(year)) < frac
        new = -np.ones(len(year), dtype=np.int64)
        new[keep] = np.arange(keep.sum())
        m = keep[edges[0]] & keep[edges[1]]
        edges = new[edges[:, m]]
        year = year[keep]
    return dict(edges=edges, node_year=year, edge_year=np.maximum(year[edges[0]], year[edges[1]]),
                feat=np.zeros((len(year), 1), np.float32), label=None, base=BASE, last=LAST, edge_mode="nodes")


def run(frac, method):
    import run_suite as rs
    import stream_eval as se
    import wd_eval as wd
    d = dataset(frac)
    node_t, edge_t, T = se.schedule(d, 0)
    path, env = rs.SPECS[method]
    os.environ.update(env)
    t0 = time.time()
    labels, times = se.replay(str(path), d, node_t, edge_t, T)
    wall = time.time() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3   # bytes on macOS
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    key = f"{frac}|{method}"
    ref = res.get(f"{frac}|_ref")
    if method == "recompute_leiden":
        ref = wd.score(labels, d, node_t, edge_t)["full"]
        res = json.loads(OUT.read_text()) if OUT.exists() else {}
        res[f"{frac}|_ref"] = ref
    r = wd.score(labels, d, node_t, edge_t, ref)
    res = json.loads(OUT.read_text()) if OUT.exists() else res
    res[f"{frac}|_ref"] = ref
    res[key] = dict(nodes=int(len(d["node_year"])), edges=int(d["edges"].shape[1]),
                    nodes_v0=int((node_t == 0).sum()), Q=r["Q"], C=r["C"], score=r["Q"] - LAM * r["C"],
                    update_seconds_median=float(np.median(times[1:])), init_seconds=float(times[0]),
                    total_seconds=float(sum(times)), wall=wall, peak_gb=peak_gb)
    OUT.write_text(json.dumps(res, indent=1))
    print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)


def report():
    res = json.loads(OUT.read_text())
    for k, v in sorted(res.items()):
        if not k.endswith("_ref"):
            print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    {"prepare": lambda: prepare(), "report": lambda: report(),
     "run": lambda: run(float(sys.argv[2]), sys.argv[3])}[sys.argv[1]]()
