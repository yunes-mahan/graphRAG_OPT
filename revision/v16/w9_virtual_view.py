"""W9 (V16): materialised vs virtual view of community reports on the QA stream (all tokens measured, gpt-5.6-luna).
Virtual view (LazyGraphRAG-style, no reports kept): every local question writes the 6 reports it reads at query time;
cost per question = 6 x mean measured tokens (input + output) of a report of the QA stream (qa/cache_summaries.json).
Materialised view: a maintained store's measured update tokens (input + output, qa/results.json) per update (6 updates).
Break-even = materialised tokens per update / virtual tokens per question (questions per update above which keeping the
reports is cheaper).  This ignores answer tokens (equal in both) and reuse of query-time reports across questions.
Output: analysis/v16/w9_virtual_view.json"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
K = 6
c = json.loads((A / "qa" / "cache_summaries.json").read_text())
u = [x["usage"] for x in c.values() if isinstance(x, dict) and "usage" in x]
per_report = float(np.mean([x["inp"] + x["out"] for x in u]))
T = json.loads((A / "qa" / "plan.json").read_text())["T"]
R = [r for r in json.loads((A / "qa" / "results.json").read_text())["summary"] if r["store"] != "FRESH|recompute all"
     and r["store"] != "papers|vector"]
be = {r["store"]: (r["update_input_tokens"] + r["update_output_tokens"]) / (T - 1) / (K * per_report) for r in R}
out = dict(per_report_tokens=per_report, n_reports=len(u), per_question_virtual=K * per_report, updates=T - 1,
           break_even=be, break_even_min=min(be.values()), break_even_max=max(be.values()))
print({k: v for k, v in out.items() if k != "break_even"})
(HERE / "w9_virtual_view.json").write_text(json.dumps(out, indent=1))
