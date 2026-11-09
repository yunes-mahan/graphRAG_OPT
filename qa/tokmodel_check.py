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
