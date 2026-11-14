"""D2 (V12): when was the TOST margin (+-0.05 recall) fixed relative to the data it was applied to?
Records file creation/modification times of the script that defines the margin (validate_score.py) and of the QA
results (120 questions: qa/results.json; 180 additional questions: qa_more.py / qa/results_more.json), and the margin
relative to the fresh-report recall.  File times are the only evidence (the git history starts later).
Output: analysis/v12/d2_margin_provenance.json
"""
import datetime as dt
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent


def times(p):
    st = os.stat(p)
    born = getattr(st, "st_birthtime", st.st_mtime)
    f = lambda x: dt.datetime.fromtimestamp(x).isoformat(timespec="minutes")  # noqa: E731
    return dict(created=f(born), modified=f(st.st_mtime))


files = {k: times(A / k) for k in ("validate_score.py", "qa/results.json", "qa_more.py", "qa/results_more.json")}
fresh = next(r for r in json.loads((A / "qa" / "results.json").read_text())["summary"] if r["store"] == "FRESH|recompute all")
margin = 0.05
out = dict(files=files, margin=margin, fresh_recall=fresh["recall"], margin_share_of_fresh=margin / fresh["recall"],
           margin_set_before_120q_results=files["validate_score.py"]["created"] < files["qa/results.json"]["created"],
           margin_set_before_180q_generated=files["validate_score.py"]["created"] < files["qa_more.py"]["created"])
print(json.dumps(out, indent=1))
(HERE / "d2_margin_provenance.json").write_text(json.dumps(out, indent=1))
