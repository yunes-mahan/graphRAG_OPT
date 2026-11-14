"""A5 (V12): the rollback price uses (T+1)/T with T = 15 (the 16-version development streams) on every graph
(programs/priced_*.py: PRICE = lambda * 16 / 15).  This script reads the real number of versions of every stream
on which rollback is evaluated and reports the exact factor (T+1)/T and the relative deviation of the fixed 16/15.
Output: analysis/v12/a5_rollback_factor.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
TASK = Path(os.environ.get("GCM_TASK", A.parent / "tasks" / "kgversion2"))
os.environ["TASK_SOURCE_DIR"] = str(TASK)
sys.path[:0] = [str(TASK), str(A), str(A / "wikidata")]
import evaluate as ev  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402

V = {}
for name, base, last in (("arxiv", 2015, 2020), ("collab", 2012, 2017), ("arxiv", 2017, 2020), ("collab", 2010, 2017)):
    V[f"{name}-{base % 100}"] = int(ev.schedule(ev.load(name), base, last, 0)[2])
V["entity"] = int(eg.stream(0)[3])
V["wikidata"] = int(se.schedule(wd.build()[0], 0)[2])
V["wikipedia"] = len(json.loads((A / "wiki" / "results.json").read_text())["n_pages"])
fixed = 16 / 15
out = {k: dict(versions=v, updates=v - 1, exact=v / (v - 1), deviation_of_fixed=fixed / (v / (v - 1)) - 1) for k, v in V.items()}
print(json.dumps(out, indent=1))
(HERE / "a5_rollback_factor.json").write_text(json.dumps(out, indent=1))
