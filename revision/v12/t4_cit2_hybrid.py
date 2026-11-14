"""T4 (V13): Hybrid on the full ogbl-citation2 stream (the one 'not run' cell of Table 7's citation2 column).
Runs scale_citation2.run(1.0, 'hybrid') unchanged, but writes to analysis/v12/t4/scale_hybrid.json, seeded with
the recompute reference of scale.json (scale.json itself is not modified)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import scale_citation2 as sc  # noqa: E402

out = HERE / "t4" / "scale_hybrid.json"
if not out.exists():
    ref = json.loads((HERE.parent / "scale.json").read_text())["1.0|_ref"]
    out.write_text(json.dumps({"1.0|_ref": ref}, indent=1))
sc.OUT = out
sc.run(1.0, "hybrid")
