"""Evaluate the hybrid maintainer for several DELTA values, datasets and schedule seeds."""

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
PROG = str(config.MAINTAINERS / "hybrid_refresh.py")


def main():
    datasets = sys.argv[1].split(",")
    deltas = [float(x) for x in sys.argv[2].split(",")]
    seeds = [int(x) for x in sys.argv[3].split(",")]
    out = HERE / os.environ.get("HYBRID_OUT", "results_hybrid.json")
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["dataset"], r["delta"], r["seed"]) for r in rows}
    for ds in datasets:
        d = se.load(ds)
        for seed in seeds:
            node_t, edge_t, T = se.schedule(d, seed)
            for delta in deltas:
                if (ds, delta, seed) in done:
                    continue
                os.environ["HYBRID_DELTA"] = str(delta)
                t0 = time.time()
                labels, times = se.replay(PROG, d, node_t, edge_t, T)
                s = se.score(labels, d, node_t, edge_t)
                s.update(dataset=ds, delta=delta, seed=seed, method=f"hybrid_d{delta}",
                         seconds=float(sum(times)), update_seconds=float(sum(times[1:])))
                rows.append(s)
                out.write_text(json.dumps(rows, indent=1))
                print(f"{ds} seed={seed} delta={delta}: score={s['Q']-0.15*s['R']:.4f} Q={s['Q']:.4f} "
                      f"R={s['R']:.4f} finalQ={s['final_Q']:.4f} t={sum(times):.0f}s "
                      f"(wall {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
