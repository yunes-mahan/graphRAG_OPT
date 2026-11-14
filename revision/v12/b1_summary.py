"""B1 (V12) summary of the batch-size sweep: means over 3 seeds of the per-update rewrite cost (fraction of a full
re-index; tau = 0.9 and never), coverage and modularity, plus the mean number of new nodes per update.
Inputs: v12/b1/entity_b{1,10,100}.json (graphrag_hier.py, GraphRAG hierarchy, every community reported; coverage at the
final version), v12/b1/flat_arxiv.json (b1_flat_sweep.py + b1_df.py; coverage averaged over versions).
Output: v12/b1_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]
import b1_streams as bs  # noqa: E402

out = {"entity": {}, "arxiv": {}}
for B in (1, 10, 100):
    D = json.loads((HERE / "b1" / f"entity_b{B}.json").read_text())
    new = [float(np.mean([int((bs.entity_stream(B, 10, int(s))[1] == t).sum()) for t in range(1, 11)])) for s in D]
    rec = dict(new_nodes=float(np.mean(new)))
    for m in ("reindex+match", "warm", "graphrag_update", "zep_attach"):
        rec[m] = dict(C09=float(np.mean([D[s][m]["cost"]["0.9|1"] for s in D])),
                      Cnever=float(np.mean([D[s][m]["cost"]["0.0|1"] for s in D])),
                      coverage=float(np.mean([D[s][m]["coverage_final"]["1"] for s in D])),
                      Qleaf=float(np.mean([D[s][m]["Qleaf"] for s in D])), Qtop=float(np.mean([D[s][m]["Qtop"] for s in D])))
    out["entity"][str(B)] = rec
F = json.loads((HERE / "b1" / "flat_arxiv.json").read_text())
for B in (10, 100, 1000):
    rec = {}
    for m in ("recompute_leiden", "warmstart_leiden", "frontier_leiden", "df_leiden_sahu", "graphrag_delta"):
        rs = [F[f"{B}|{s}|{m}"] for s in (0, 1, 2) if f"{B}|{s}|{m}" in F]
        if len(rs) == 3:
            rec[m] = {k: float(np.mean([r[k] for r in rs])) for k in ("C09", "Cnever", "Q", "coverage")}
            rec["new_nodes"] = float(np.mean([r["new_nodes"] for r in rs]))
    out["arxiv"][str(B)] = rec
print(json.dumps(out, indent=1))
(HERE / "b1_summary.json").write_text(json.dumps(out, indent=1))
