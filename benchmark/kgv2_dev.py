"""Score maintainers on the two development streams with the token-level metric (tasks/kgversion2)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import kgv2_calibrate as kc
import run_suite as rs
ref = json.loads((kc.TASK / "resources" / "reference_tokens.json").read_text())
lam = json.loads((kc.TASK / "resources" / "calibration.json").read_text())["lambda"]
out = {}
for name in sys.argv[1].split(","):
    sc = []
    for stream, s in kc.ev.STREAMS.items():
        path, env = rs.SPECS[name]
        t0 = time.time()
        labels, times = kc.replay(path, env, stream, s["base"], s["last"], s["seed"])
        r = kc.score_any(labels, stream, s["base"], s["last"], s["seed"], ref[stream])
        sc.append(r["mean_modularity"] - lam * r["mean_relative_token_cost"])
        print(name, stream, f"score={sc[-1]:.4f} Q={r['mean_modularity']:.4f} C={r['mean_relative_token_cost']:.4f} t={sum(times):.0f}s", flush=True)
    print(name, "combined", round(float(np.mean(sc)), 4), flush=True)
