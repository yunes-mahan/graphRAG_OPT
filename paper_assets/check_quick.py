"""Compare the re-run development scores of quick_reproduce.sh with the values reported in the paper."""

import re
import sys

PAPER = {"warmstart_leiden": 0.7213, "priced_warm": 0.7428}
log = open(sys.argv[1]).read()
got = {m: float(v) for m, v in re.findall(r"^(\w+) combined ([0-9.]+)", log, re.M)}
ok = True
for m, v in PAPER.items():
    same = m in got and abs(got[m] - v) < 0.002
    ok &= same
    print(f"      {m:18s} paper {v:.4f}  reproduced {got.get(m, float('nan')):.4f}  {'OK' if same else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
