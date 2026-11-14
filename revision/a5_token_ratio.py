"""A5 check: exact ratio of the largest to the smallest update input tokens over the 16 QA-stream stores
(source of \\QATokRatio, computed in make_assets.qa_assets from analysis/qa/results.json).
Output: analysis/v2/a5_token_ratio.json"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE.parent / "qa" / "results.json").read_text())
tok = {x["store"]: x["update_input_tokens"] for x in r["summary"]
       if x["store"].split("|")[0] in ("recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid")}
mx, mn = max(tok, key=tok.get), min(tok, key=tok.get)
out = dict(n_stores=len(tok), max_store=mx, max_tokens=tok[mx], min_store=mn, min_tokens=tok[mn],
           ratio_exact=tok[mx] / tok[mn], ratio_of_rounded_display=round(tok[mx] / 1e3) / round(tok[mn] / 1e3))
print(json.dumps(out, indent=1))
(HERE / "a5_token_ratio.json").write_text(json.dumps(out, indent=1))
