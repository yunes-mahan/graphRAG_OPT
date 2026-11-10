"""Does the token cost model tok(S) = 1500 + min(8000, 250|S|) rank stores like the measured LLM tokens?
For the 16 QA-stream stores (4 maintainers x 4 tolerances): modelled update cost = sum of tok(S) over all report
(re)writes after version 0; measured = input + output tokens of the same gpt-5.6-luna calls (cache).
Writes tokmodel_check.json."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import sys
sys.path.insert(0, str(HERE))
import qa_run as qr  # noqa: E402

plan = json.loads((HERE / "qa" / "plan.json").read_text())
cs = json.loads((HERE / "qa" / "cache_summaries.json").read_text())
rows = []
for m in ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"]:
    for tau, v in plan["methods"][m]["taus"].items():
        ev = [s for t, s in v["events"] if t > 0]
        model = sum(1500 + min(8000, 250 * len(s)) for s in ev)
        meas = sum(cs[qr.key_of(s)]["usage"]["inp"] + cs[qr.key_of(s)]["usage"]["out"] for s in ev)
        rows.append(dict(store=f"{m}|{tau}", model=model, measured=meas))
mo, me = np.array([r["model"] for r in rows]), np.array([r["measured"] for r in rows])
res = dict(stores=rows, spearman=float(stats.spearmanr(mo, me).statistic), pearson=float(stats.pearsonr(mo, me).statistic),
           ratio_median=float(np.median(me / mo)))
print({k: v for k, v in res.items() if k != "stores"})
(HERE / "tokmodel_check.json").write_text(json.dumps(res, indent=1))

# ---- flat LLM entity graph: the 7 maintainer stores of qa_entity.py (reports for communities >= 10 entities)
import hashlib  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import qa_entity as qe  # noqa: E402

G = HERE / "entity_graph"
r = json.loads((G / "eval.json").read_text())
cr = json.loads((G / "cache_reports.json").read_text())
_, node_t, _, _ = eg.stream(0)
erows = []
for m in qe.STORES:
    labels = [np.asarray(x) for x in r[f"{m}|labels0"]]
    ev, _ = qe.events(labels, node_t)
    ev = [s for t, s in ev if t > 0]
    model = sum(1500 + min(8000, 250 * len(s)) for s in ev)
    meas = 0
    for s in ev:
        u = cr[hashlib.sha1(("ereport|" + ",".join(map(str, s))).encode()).hexdigest()]["usage"]
        meas += u["inp"] + u["out"]
    erows.append(dict(store=m, model=model, measured=meas))
mo, me = np.array([x["model"] for x in erows]), np.array([x["measured"] for x in erows])
res["entity"] = dict(stores=erows, spearman=float(stats.spearmanr(mo, me).statistic),
                     pearson=float(stats.pearsonr(mo, me).statistic), ratio_median=float(np.median(me / mo)))
print("entity graph:", {k: v for k, v in res["entity"].items() if k != "stores"})
(HERE / "tokmodel_check.json").write_text(json.dumps(res, indent=1))
