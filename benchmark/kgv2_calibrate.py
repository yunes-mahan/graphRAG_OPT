"""Calibrate tasks/kgversion2: reference token cost (recompute Leiden), lambda (recompute ~ lazy tie) and
reference scores of the known maintainers on the two DEVELOPMENT streams (order 0). Also scores any program
on held-out orders / streams with the same metric (mode 'heldout').
usage: python kgv2_calibrate.py calibrate
       python kgv2_calibrate.py heldout PROGRAM_PATH NAME     (orders 1-9 of arxiv-15/collab-12, 0-2 of arxiv-17/collab-10)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[1] / "token_task"   # resources/ = *.npz from benchmark/prepare_data.py + calibration files
sys.path.insert(0, str(TASK))
os.environ["TASK_SOURCE_DIR"] = str(TASK)
import evaluate as ev  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
import run_suite as rs  # noqa: E402

HERE = config.RESULTS
BASELINES = ["recompute_leiden", "lazy", "warmstart_leiden", "frontier_leiden", "hybrid", "local_repair"]


def replay(path, env, stream, base, last, seed):
    for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER"):
        os.environ.pop(k, None)
    os.environ.update(env)
    d = ev.load(stream)
    node_t, edge_t, T = ev.schedule(d, base, last, seed)
    import stream_eval as se
    labels, times = se.replay(str(path), d, node_t, edge_t, T)
    return labels, times


def score_any(labels, stream, base, last, seed, ref):
    saved = dict(ev.STREAMS[stream])
    ev.STREAMS[stream] = dict(base=base, last=last, seed=seed)
    try:
        return ev.score_stream(labels, stream, ref)
    finally:
        ev.STREAMS[stream] = saved


def calibrate():
    ref, raw = {}, {}
    for stream, s in ev.STREAMS.items():
        path, env = rs.SPECS["recompute_leiden"]
        labels, _ = replay(path, env, stream, s["base"], s["last"], s["seed"])
        ref[stream] = score_any(labels, stream, s["base"], s["last"], s["seed"], None)["full_tokens_per_version"]
    (TASK / "resources" / "reference_tokens.json").write_text(json.dumps(ref))
    for m in BASELINES:
        for stream, s in ev.STREAMS.items():
            path, env = rs.SPECS[m]
            labels, times = replay(path, env, stream, s["base"], s["last"], s["seed"])
            r = score_any(labels, stream, s["base"], s["last"], s["seed"], ref[stream])
            raw[(m, stream)] = (r["mean_modularity"], r["mean_relative_token_cost"], float(sum(times)))
            print(m, stream, raw[(m, stream)], flush=True)
    dq = np.mean([raw[("recompute_leiden", s)][0] - raw[("lazy", s)][0] for s in ev.STREAMS])
    dc = np.mean([raw[("recompute_leiden", s)][1] - raw[("lazy", s)][1] for s in ev.STREAMS])
    lam = float(dq / dc)
    (TASK / "resources" / "calibration.json").write_text(json.dumps(dict(lambda_=lam, **{"lambda": lam})))
    out = {}
    for m in BASELINES:
        sc = {s: raw[(m, s)][0] - lam * raw[(m, s)][1] for s in ev.STREAMS}
        out[m] = dict(combined=float(np.mean(list(sc.values()))), **{f"{s}_score": v for s, v in sc.items()},
                      **{f"{s}_Q": raw[(m, s)][0] for s in ev.STREAMS},
                      **{f"{s}_C": raw[(m, s)][1] for s in ev.STREAMS},
                      **{f"{s}_seconds": raw[(m, s)][2] for s in ev.STREAMS})
    print(f"lambda = {lam:.4f}")
    for m, r in sorted(out.items(), key=lambda x: -x[1]["combined"]):
        print(f"{m:18s} combined={r['combined']:.4f}  arxiv Q={r['arxiv_Q']:.4f} C={r['arxiv_C']:.3f}  "
              f"collab Q={r['collab_Q']:.4f} C={r['collab_C']:.3f}")
    (HERE / "kgv2_calibration.json").write_text(json.dumps(dict(lambda_=lam, baselines=out), indent=1))


HELDOUT = [("arxiv", 2015, 2020, s) for s in range(1, 10)] + [("collab", 2012, 2017, s) for s in range(1, 10)] + \
          [("arxiv", 2017, 2020, s) for s in range(3)] + [("collab", 2010, 2017, s) for s in range(3)]


def heldout(program, name):
    """Held-out evaluation: reference tokens are recomputed per order with recompute Leiden."""
    f = HERE / os.environ.get("KGV2_OUT", "kgv2_heldout.json")
    res = json.loads(f.read_text()) if f.exists() else {}
    main_f = HERE / "kgv2_heldout.json"            # reference tokens are shared from the main file
    if f != main_f and main_f.exists():
        for k, v in json.loads(main_f.read_text()).items():
            if "_ref" in v:
                res.setdefault(k, {})["_ref"] = v["_ref"]
    path_ref, env_ref = rs.SPECS["recompute_leiden"]
    for stream, base, last, seed in HELDOUT:
        key = f"{stream}|{base}|{seed}"
        res.setdefault(key, {})
        if "_ref" not in res[key]:
            labels, _ = replay(path_ref, env_ref, stream, base, last, seed)
            r = score_any(labels, stream, base, last, seed, None)
            res[key]["_ref"] = r["full_tokens_per_version"]
            res[key]["recompute_leiden"] = None   # filled below like any program
            f.write_text(json.dumps(res))
        if res[key].get(name):
            continue
        path, env = (rs.SPECS[name] if name in rs.SPECS else (program, {}))
        labels, times = replay(path, env, stream, base, last, seed)
        r = score_any(labels, stream, base, last, seed, res[key]["_ref"])
        res[key][name] = dict(Q=r["mean_modularity"], C=r["mean_relative_token_cost"], seconds=float(sum(times)))
        f.write_text(json.dumps(res))
        print(key, name, res[key][name], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate()
    else:
        heldout(sys.argv[2], sys.argv[3])
