"""T4 (V13): one more maintainer on the full ogbl-citation2 stream (Table 7 cells that were "not run").
usage: python v12/t4_cit2.py METHOD      (hybrid | local_repair)
Runs scale_citation2.run(1.0, METHOD) unchanged, but writes to analysis/v12/t4/scale_METHOD.json, seeded with the
recompute reference of scale.json (scale.json itself is not modified).  v12/t4_cit2_hybrid.py = this for hybrid."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import scale_citation2 as sc  # noqa: E402

m = sys.argv[1]
out = HERE / "t4" / f"scale_{m}.json"
if not out.exists():
    ref = json.loads((HERE.parent / "scale.json").read_text())["1.0|_ref"]
    out.write_text(json.dumps({"1.0|_ref": ref}, indent=1))
sc.OUT = out
sc.run(1.0, m)
