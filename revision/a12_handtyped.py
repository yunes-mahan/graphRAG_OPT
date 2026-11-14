"""A12: recompute the numbers that are typed by hand in the paper text / Table 1 (no script produced them so far):
dataset sizes, number of versions, and the batch-size ranges of order 0 quoted in Section 4.
Output: analysis/v2/a12_handtyped.json   (compared against the paper by hand in CHANGES.md)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASK = Path(os.environ.get("GCM_TASK", Path(__file__).resolve().parents[2] / "tasks" / "kgversion2"))
os.environ["TASK_SOURCE_DIR"] = str(TASK)
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(HERE.parent))
import evaluate as ev  # noqa: E402
import entity_graph_eval as eg  # noqa: E402

out = {}
for name, base, last in (("arxiv", 2015, 2020), ("arxiv", 2017, 2020), ("collab", 2012, 2017), ("collab", 2010, 2017)):
    d = ev.load(name)
    node_t, edge_t, T = ev.schedule(d, base, last, 0)
    und = {tuple(sorted(map(int, e))) for e in d["edges"].T if e[0] != e[1]}
    new_nodes = [int((node_t == t).sum()) for t in range(1, T)]
    out[f"{name}-{base % 100}"] = dict(nodes=int(len(d["node_year"])), edges_array=int(d["edges"].shape[1]),
                                       edges_undirected_unique=len(und), versions=int(T),
                                       batch_new_nodes_min=min(new_nodes), batch_new_nodes_max=max(new_nodes))
d, node_t, edge_t, T = eg.stream(0)
new = [int((node_t == t).sum()) for t in range(1, T)]
out["entity"] = dict(nodes=int(len(node_t)), edges=int(d["edges"].shape[1]), versions=int(T),
                     batch_new_entities_min=min(new), batch_new_entities_max=max(new))
print(json.dumps(out, indent=1))
(HERE / "a12_handtyped.json").write_text(json.dumps(out, indent=1))
