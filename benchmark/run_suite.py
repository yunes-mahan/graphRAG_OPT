"""Run a set of maintainers on a (dataset, base-year) stream for several schedule orders.

usage: run_suite.py DATASET BASE_YEAR SEEDS METHODS [OUTFILE]
  METHODS: comma list of names from SPECS below. Results are appended (resumable).
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402


import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import stream_eval as se  # noqa: E402

HERE = config.RESULTS
P = config.MAINTAINERS
SPECS = {
    "lazy": (P / "lazy_attach.py", {}),
    "recompute_leiden": (P / "recompute_leiden.py", {}),
    "warmstart_leiden": (P / "warmstart_leiden.py", {}),
    "frontier_leiden": (P / "frontier_leiden.py", {}),
    "local_repair": (P / "local_repair_winner.py", {}),
    "hybrid": (P / "hybrid_refresh.py", dict(HYBRID_DELTA="0.15", HYBRID_BASE="local", HYBRID_TRIGGER="drift")),
    "place_drift": (P / "hybrid_refresh.py", dict(HYBRID_DELTA="0.15", HYBRID_BASE="place", HYBRID_TRIGGER="drift")),
    "local_periodic5": (P / "hybrid_refresh.py", dict(HYBRID_BASE="local", HYBRID_TRIGGER="periodic", HYBRID_K="5")),
    "local_periodic3": (P / "hybrid_refresh.py", dict(HYBRID_BASE="local", HYBRID_TRIGGER="periodic", HYBRID_K="3")),
    "priced_warm": (P / "priced_warmstart.py", {}),
    "graphrag_delta": (P / "graphrag_delta.py", {}),
    "priced_frontier": (P / "priced_frontier.py", {}),
    "semantic_hybrid": (P / "semantic_hybrid.py", dict(HYBRID_DELTA="0.15", HYBRID_BASE="local", HYBRID_TRIGGER="drift")),
    "semantic_hybrid_c": (P / "semantic_hybrid.py", dict(HYBRID_DELTA="0.15", HYBRID_BASE="local", HYBRID_TRIGGER="drift", SEM_CENTER="1")),
}


def main():
    dataset, base, seeds, methods = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(",")
    seeds = [int(x) for x in seeds.split(",")]
    out = HERE / (sys.argv[5] if len(sys.argv) > 5 else f"suite_{dataset}_{base}.json")
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["method"], r["seed"]) for r in rows}
    d = se.load(dataset)
    for seed in seeds:
        node_t, edge_t, T = se.schedule(d, seed, base=base)
        for m in methods:
            if (m, seed) in done:
                continue
            path, env = SPECS[m]
            for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER"):
                os.environ.pop(k, None)
            os.environ.update(env)
            t0 = time.time()
            labels, times = se.replay(str(path), d, node_t, edge_t, T)
            s = se.score(labels, d, node_t, edge_t)
            s.pop("Q_curve") if seed > 2 else None
            s.update(method=m, seed=seed, dataset=dataset, base=base, versions=T,
                     seconds=float(sum(times)), update_seconds=float(sum(times[1:])))
            rows.append(s)
            out.write_text(json.dumps(rows, indent=1))
            print(f"{dataset}/{base} seed={seed} {m:17s} S={s['Q']-0.15*s['R']:.4f} Q={s['Q']:.4f} "
                  f"R={s['R']:.4f} Rc={s['R_comm']:.4f} t={sum(times):.0f}s (wall {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
