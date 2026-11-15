"""W6 (V16): the Leiden seed as a random factor for Table 5 (held-out flat streams). Every leidenalg call of a maintainer
(find_partition seed, Optimiser rng seed) uses SEED instead of the programs' fixed seed 0; orders and reference tokens
are those of kgv2_heldout.json (seed 0 = the existing results). DF Leiden (authors' C++ code) is not seeded here.
usage: python v16/w6_seeds.py SEED MAINTAINER    -> analysis/v16/w6/seed{SEED}_{MAINTAINER}.json"""
import json
import sys
from pathlib import Path

import leidenalg as la

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A)]
SEED = int(sys.argv[1])
_fp = la.find_partition
la.find_partition = lambda *a, **k: _fp(*a, **{**k, "seed": SEED})
_srs = la.Optimiser.set_rng_seed
la.Optimiser.set_rng_seed = lambda self, s: _srs(self, SEED)
import kgv2_calibrate as kc  # noqa: E402
import run_suite as rs  # noqa: E402

name = sys.argv[2]
ref = json.loads((A / "kgv2_heldout.json").read_text())
out_f = HERE / "w6" / f"seed{SEED}_{name}.json"
res = json.loads(out_f.read_text()) if out_f.exists() else {}
path, env = rs.SPECS[name]
for stream, base, last, seed in kc.HELDOUT:
    key = f"{stream}|{base}|{seed}"
    if key in res:
        continue
    labels, times = kc.replay(path, env, stream, base, last, seed)
    r = kc.score_any(labels, stream, base, last, seed, ref[key]["_ref"])
    res[key] = dict(Q=r["mean_modularity"], C=r["mean_relative_token_cost"], seconds=float(sum(times)))
    out_f.write_text(json.dumps(res, indent=1))
    print(key, name, SEED, res[key], flush=True)
