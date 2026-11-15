"""V15/D3: does "never-refreshed reports differ substantively from fresh ones for 35 % of communities" (delta_summary.py,
kind 'stale') include self-comparisons?  The stale report is the one written when the community first appeared; the
reference is the cached fresh report of its CURRENT member set.  If the member set never changed, both texts are the
same cache entry.  Recomputed from cached texts and cached judge verdicts only (no LLM calls).
Output: analysis/v12/v15_fid_stale.json"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A))
import delta_summary as ds  # noqa: E402
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

plan = json.loads((A / "qa" / "plan.json").read_text())
node_t = np.load(A / "qa" / "node_t.npy")
nodes = pd.read_csv(A / "qa" / "nodes.csv")
texts = qp.load_texts(nodes.paper_id.values)
docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
cs, cr = qr.Cache("summaries"), qr.Cache("delta_rate")
ch, same = [], []
for m in ds.METHODS:
    labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
    ev, final, current = ds.replay(labels, node_t)
    created = {}
    for t, cid, old, new in ev:
        if old is None or t == 0:
            created[cid] = sorted(new)
    for cid in final:
        if cid not in created:
            continue
        stale = qr.summarise(created[cid], docs, cs)["text"]
        full = qr.summarise(sorted(current[cid]), docs, cs)["text"]
        ch.append(ds.rate(stale, full, cr)["change"])
        same.append(stale == full)
ch, same = np.array(ch), np.array(same)
out = dict(n=int(len(ch)), self_compared=int(same.sum()), subst_all=float((ch >= 3).mean()),
           subst_other=float((ch[~same] >= 3).mean()), n_other=int((~same).sum()))
print(out)
(HERE / "v15_fid_stale.json").write_text(json.dumps(out, indent=1))
