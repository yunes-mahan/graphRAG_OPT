"""M8 (V17): (b) break-even query rates re-derived from their inputs (v16/w3 per-update GraphRAG-format report tokens,
v12/b4_tco.json measured query tokens), independently of v16/w3_summary.py; (c) virtual view for GLOBAL questions on
the QA stream: a global question reads all reports of a store, so a virtual view writes all reports of the final
fresh partition at query time (count and measured tokens per report as v16/w9 and v12/t5).
Output: analysis/v17/m8_checks.json"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
R = {}
for s in range(5):
    R.update(json.loads((A / "v16" / "w3" / f"entity_s{s}.json").read_text()))
rep = {m: float(np.mean([R[s][m]["exact_abs_per_update"]["0.9"] for s in R])) for m in ("reindex+match", "graphrag_update", "zep_attach")}
q = {m: json.loads((A / "v12" / "b4_tco.json").read_text())["maintainers"][m]["query_tokens_measured"] for m in rep}
be_match = (rep["reindex+match"] - rep["graphrag_update"]) / (q["graphrag_update"] - q["reindex+match"])
be_zep = (rep["zep_attach"] - rep["graphrag_update"]) / (q["graphrag_update"] - q["zep_attach"])
w9 = json.loads((A / "v16" / "w9_virtual_view.json").read_text())
n_fresh = json.loads((A / "v12" / "t5_reuse_checks.json").read_text())["fresh"]["written"]
per_global = n_fresh * w9["per_report_tokens"]
R9 = [r for r in json.loads((A / "qa" / "results.json").read_text())["summary"] if r["store"] not in ("FRESH|recompute all", "papers|vector")]
be_g = [(r["update_input_tokens"] + r["update_output_tokens"]) / w9["updates"] / per_global for r in R9]
out = dict(report_tokens_per_update=rep, query_tokens=q, break_even_gu_vs_match=be_match, break_even_gu_vs_zep=be_zep,
           global_reports=n_fresh, per_global_question_virtual=per_global, global_break_even_min=min(be_g), global_break_even_max=max(be_g))
print(json.dumps(out, indent=1))
(HERE / "m8_checks.json").write_text(json.dumps(out, indent=1))
