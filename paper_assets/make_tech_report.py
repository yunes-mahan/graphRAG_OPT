"""Render the artifact's technical report from tech_report_template.md and numbers.json (no hand-typed numbers)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
M = json.loads((HERE / "numbers.json").read_text())


def tex2md(v):
    return (v.replace("{,}", ",").replace("\\%", "%").replace("$<$", "<").replace("$", "").replace("{<}", "<")
             .replace("\\,", " ").replace("\\pm", "±"))


tpl = (HERE / "tech_report_template.md").read_text()
missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", tpl)) - set(M))
if missing:
    sys.exit(f"missing macros: {missing}")
out = re.sub(r"\{\{(\w+)\}\}", lambda m: tex2md(M[m.group(1)]), tpl)
dst = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "TECH_REPORT.md"
dst.write_text(out)
print("wrote", dst)
