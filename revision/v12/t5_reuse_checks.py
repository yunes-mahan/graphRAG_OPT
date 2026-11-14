"""T5 (V13): two checks behind Table 8 (topic-anchored reuse, QA stream). No LLM calls: cached outputs only.
1. Fresh reference cells: the fresh store has one report for every community (>= 5 members) of the recomputed final
   partition; "written" = number of those reports, "tokens" = their input + output tokens (cache_summaries.json).
2. Why "subst." is 0 % for Recompute/ID, below the 2 % noise floor of two independent fresh reports: the fidelity
   check of summary_reuse.py compares each final store report with the cached fresh report of the community's
   CURRENT members, and both texts come from the same cache entry (key = member set).  When a report was written for
   exactly the current members, the check compares a text with itself.  Count, per evaluated store, the share of
   final reports whose stored member set equals the current member set (input "full"; title-only reports are a
   different text by construction).
Output: analysis/v12/t5_reuse_checks.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(A))
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402
import summary_reuse as sr  # noqa: E402

QA = A / "qa"
plan = json.loads((QA / "plan.json").read_text())
node_t = np.load(QA / "node_t.npy")
E = np.load(QA / "emb_te3small.npy").astype(np.float64)
E = E - E[node_t == 0].mean(0)
E /= np.linalg.norm(E, axis=1, keepdims=True)
cs = qr.Cache("summaries")

lab = np.asarray(plan["methods"]["recompute_leiden"]["labels"][-1])
nd = np.where(node_t >= 0)[0]                 # as qa_run.py (FRESH|recompute all)
comms = {}
for v, c in zip(nd, lab):
    comms.setdefault(int(c), set()).add(int(v))
fresh = [sorted(m) for m in comms.values() if len(m) >= qp.MIN_SIZE]
hits = [cs.get(qr.key_of(m)) for m in fresh]
missing = sum(h is None for h in hits)
tok = sum(h["usage"]["inp"] + h["usage"]["out"] for h in hits if h)
out = dict(fresh=dict(written=len(fresh), tokens=int(tok), missing_in_cache=int(missing)))
print("fresh", out["fresh"])

identical = {}
for m in sr.METHODS:
    labels = [np.asarray(x) for x in plan["methods"][m]["labels"]]
    for rule in ("id", "topic:0.95"):
        ev, lib, final, current = sr.replay(labels, node_t, E, rule)
        same = [lib[j]["members"] == frozenset(current[cid]) for cid, j in final.items()]
        identical[f"{m}|{rule}|full"] = dict(final_reports=len(same), identical=int(sum(same)),
                                             share=float(np.mean(same)) if same else 0.0)
        print(m, rule, identical[f"{m}|{rule}|full"])
out["identical_member_sets"] = identical
(HERE / "t5_reuse_checks.json").write_text(json.dumps(out, indent=1))

# ---- 3. every row of Table 8 from the caches: overall "subst." (must equal the table) and "subst." over the
#         reports whose member set differs from the current one (the only pairs that are not a text vs itself)
import pandas as pd  # noqa: E402
import delta_summary as ds  # noqa: E402
import input_variants as iv  # noqa: E402
import qa_pipeline as qpl  # noqa: E402

nodes = pd.read_csv(QA / "nodes.csv")
texts = qp.load_texts(nodes.paper_id.values)
docs = {int(i): texts[int(p)] for i, p in zip(nodes.idx, nodes.paper_id)}
cv, cr = qr.Cache("variants"), qr.Cache("delta_rate")


def summ(mem, inp):
    return qr.summarise(sorted(mem), docs, cs) if inp == "full" else iv.summarise_variant(sorted(mem), docs, "titles<=300", cv)


ROWS = [("recompute_leiden", "id", "full"), ("recompute_leiden", "topic:0.95", "full"), ("warmstart_leiden", "id", "full"),
        ("warmstart_leiden", "topic:0.95", "full"), ("frontier_leiden", "topic:0.95", "titles"),
        ("hybrid", "topic:0.95", "titles"), ("priced_warm", "id", "full"), ("priced_warm", "topic:0.95", "full"),
        ("priced_warm", "topic:0.95", "titles")]
rows = {}
for m, rule, inp in ROWS:
    labels = qpl.labels_for(m) if m in qpl.NEW else [np.asarray(x) for x in plan["methods"][m]["labels"]]
    ev, lib, final, current = sr.replay(labels, node_t, E, rule)
    ch, same = [], []
    for cid, j in final.items():
        a = summ(sorted(lib[j]["members"]), inp)["text"]
        b = qr.summarise(current[cid], docs, cs)["text"]
        ch.append(ds.rate(a, b, cr)["change"])
        same.append(a == b)
    ch, same = np.array(ch), np.array(same)
    rows[f"{m}|{rule}|{inp}"] = dict(final_reports=int(len(ch)), identical_text=int(same.sum()),
                                     subst_all=float((ch >= 3).mean()),
                                     subst_nonidentical=float((ch[~same] >= 3).mean()) if (~same).any() else None,
                                     n_nonidentical=int((~same).sum()))
    print(m, rule, inp, rows[f"{m}|{rule}|{inp}"], flush=True)
out["table8"] = rows
(HERE / "t5_reuse_checks.json").write_text(json.dumps(out, indent=1))
