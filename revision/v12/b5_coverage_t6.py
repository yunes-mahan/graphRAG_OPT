"""B5 (V12): coverage for Table 6 (Wikidata films, flat LLM entity graph; 8 maintainers, 5 orders): mean over versions
of the share of present nodes that lie in a community with >= 5 members (= reported), from the community sizes recorded
by v2/b3_extra_sensitivity.py (base replays, which reproduce Table 6's Q and C exactly).  Output: v12/b5_coverage_t6.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "v2"), str(A / "wikidata")]
import b3_extra_sensitivity as bx  # noqa: E402

out = {}
for g in bx.GRAPHS:
    out[g] = {}
    for m in bx.METHODS:
        covs = []
        for s in range(5):
            d, node_t, edge_t, T = bx.data(g, s)
            sz = bx.load(g, m, "base", s)
            n_present = [int(((node_t >= 0) & (node_t <= t)).sum()) for t in range(T)]
            covs.append(float(np.mean([sum(fs) / n for fs, n in zip(sz["full_sizes"], n_present)])))
        out[g][m] = dict(mean=float(np.mean(covs)), orders=covs)
        print(g, m, round(out[g][m]["mean"], 4), flush=True)
(HERE / "b5_coverage_t6.json").write_text(json.dumps(out, indent=1))
