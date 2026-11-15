"""V15/D2: did the hybrid's drift refresh fire on the QA stream?  (a) compare the stored hybrid and local-repair labels
of qa/plan.json version by version; (b) replay the hybrid (run_suite.SPECS, unchanged) on the QA stream and count
refreshes.  Output: analysis/v12/v15_hybrid_qa.json"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "programs")]
import importlib.util  # noqa: E402
import qa_plan as qp  # noqa: E402
import run_suite as rs  # noqa: E402

plan = json.loads((A / "qa" / "plan.json").read_text())
H, L = plan["methods"]["hybrid"]["labels"], plan["methods"]["local_repair"]["labels"]
same = [bool(np.array_equal(np.asarray(h), np.asarray(l))) for h, l in zip(H, L)]
d, _ = qp.build_subgraph()
node_t, edge_t, T = qp.schedule(d)
path, env = rs.SPECS["hybrid"]
os.environ.update(env)
spec = importlib.util.spec_from_file_location("hyb", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
m = mod.make_maintainer()
nodes0 = np.where(node_t == 0)[0]
E = d["edges"]
lab = m.initialize(nodes0, E[:, edge_t == 0], d["feat"][nodes0], d["node_year"][nodes0])
rep_same = []
for t in range(1, T):
    new = np.where(node_t == t)[0]
    alln = np.where((node_t >= 0) & (node_t <= t))[0]
    lab = m.update(new, E[:, edge_t == t], d["feat"][new], d["node_year"][new], alln)
    rep_same.append(bool(np.array_equal(np.asarray(lab), np.asarray(H[t]))))
out = dict(versions=len(H), hybrid_equals_local_repair=same, refreshes=int(m.refreshes),
           replay_reproduces_stored_hybrid=rep_same)
print(out)
(HERE / "v15_hybrid_qa.json").write_text(json.dumps(out, indent=1))
