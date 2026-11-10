"""Hierarchical consensus clustering with 10 / 50 / 100 runs: stability, granularity and wall-clock time.

Same setting as stability_variants.py (final graph, order 0, GraphRAG settings, every community on every level
reported, tau = 0.9): rewrite share after deleting 0.1% of the edges (3 draws) and for the real last update,
leaf/top modularity, coverage, number of reports, and seconds per consensus clustering (single process; one plain
GraphRAG hierarchical Leiden run is timed for reference).  Writes consensus_runs.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import entity_graph_eval as eg  # noqa: E402
import noise_hier as nh  # noqa: E402
import stability_variants as sv  # noqa: E402


def main():
    res = {}
    for graph in ("entity", "wikidata"):
        if graph == "wikidata":
            import stream_eval as se
            import wd_eval as wd
            d = wd.build()[0]
            node_t, edge_t, T = se.schedule(d, 0)
        else:
            d, node_t, edge_t, T = eg.stream(0)
        n, E = len(node_t), d["edges"]
        ok = (edge_t >= 0) & (edge_t < T)
        el = [(int(a), int(b)) for a, b in E[:, ok].T]
        before = [e for e, t in zip(el, edge_t[ok]) if t < T - 1]
        rng = np.random.default_rng(0)
        drops = []
        for _ in range(3):
            keep = rng.random(len(el)) >= 0.001
            drops.append([e for e, k in zip(el, keep) if k])
        t0 = time.time()
        sv.leiden(el, n)
        out = dict(plain_seconds=time.time() - t0)
        for runs in (10, 50, 100):
            t0 = time.time()
            ref = sv.consensus(el, n, runs)
            secs = time.time() - t0
            q = sv.quality(ref, el, n)
            q["seconds"] = secs
            q["drop"] = [nh.rewrite(ref, sv.consensus(x, n, runs), 0.9) for x in drops]
            q["last"] = nh.rewrite(sv.consensus(before, n, runs), ref, 0.9)
            out[str(runs)] = q
            print(graph, runs, {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in q.items()},
                  flush=True)
            res[graph] = out
            (HERE / "consensus_runs.json").write_text(json.dumps(res, indent=1, default=float))
        print(graph, "plain run seconds", round(out["plain_seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
