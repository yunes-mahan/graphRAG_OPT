"""Priced rollback at other cost weights: run priced_warm with PRICE_LAMBDA=lam on both dev streams and compare
Q - lam*C with every baseline's (Q, C) from kgv2_calibration.json at the same lam."""
import json, os, sys
import numpy as np
import kgv2_calibrate as kc
import run_suite as rs
ref = json.loads((kc.TASK / "resources" / "reference_tokens.json").read_text())
cal = json.loads((kc.HERE / "kgv2_calibration.json").read_text())["baselines"]
out = {}
f = kc.HERE / "kgv2_lambda.json"
res = json.loads(f.read_text()) if f.exists() else {}
for lam in [float(x) for x in sys.argv[1].split(",")]:
    if str(lam) in res:
        continue
    r_s = {}
    for stream, s in kc.ev.STREAMS.items():
        path, _ = rs.SPECS[sys.argv[2] if len(sys.argv) > 2 else "priced_warm"]
        labels, _ = kc.replay(path, {"PRICE_LAMBDA": str(lam)}, stream, s["base"], s["last"], s["seed"])
        r = kc.score_any(labels, stream, s["base"], s["last"], s["seed"], ref[stream])
        r_s[stream] = (r["mean_modularity"], r["mean_relative_token_cost"])
    res[str(lam)] = dict(priced=r_s)
    f.write_text(json.dumps(res, indent=1))
    sc = lambda q, c: q - lam * c
    line = {"priced": np.mean([sc(*r_s[s]) for s in r_s])}
    for m, b in cal.items():
        line[m] = np.mean([sc(b[f"{s}_Q"], b[f"{s}_C"]) for s in r_s])
    print(lam, {k: round(v, 4) for k, v in sorted(line.items(), key=lambda x: -x[1])}, flush=True)
