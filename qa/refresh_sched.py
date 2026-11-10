"""Refresh scheduling under a token budget (cf. Cho & Garcia-Molina, SIGMOD 2000).

Flat maintained partitions (warm-started Leiden, document/batch order 0) of the QA stream and of the LLM entity graph.
Reports for communities with >= MIN members; every report remembers its written member set and version. At every
update the maintainer's new communities must be reported (mandatory); after that, a budget of B report tokens
(B = f x tokens of reporting every community, f in {0.02, 0.05, 0.10, 0.20}) is spent on refreshes chosen by a
policy, highest priority first, until the budget is used:
  jaccard     : lowest Jaccard(written, current) first          (the staleness rule, budgeted)
  round_robin : fixed cyclic order over community identifiers
  oldest      : oldest report first
  largest     : largest community first
  missing     : most current members missing from the written set first (an oracle of structural staleness)
Structural staleness (per version, averaged over updates): token-weighted share of current members that are missing
from their report's written set.  Writes refresh_sched.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS


def tok(n):
    return 1500 + min(8000, 250 * n)


def run(labels, node_t, MIN, f, policy):
    store, rr = {}, []                     # cid -> (written set, version)
    stale_curve = []
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nodes, lab):
            cur.setdefault(int(c), set()).add(int(v))
        cur = {c: m for c, m in cur.items() if len(m) >= MIN}
        for c in [c for c in store if c not in cur]:
            del store[c]
        full = sum(tok(len(m)) for m in cur.values())
        for c, m in cur.items():
            if c not in store:
                store[c] = (set(m), t)                                   # mandatory new report
                if c not in rr:
                    rr.append(c)
        if t > 0:
            budget = f * full
            cand = [c for c in cur if store[c][1] < t]
            if policy == "jaccard":
                key = lambda c: len(cur[c] & store[c][0]) / len(cur[c] | store[c][0])  # noqa: E731
            elif policy == "round_robin":
                pos = {c: i for i, c in enumerate(rr)}
                start = (t * 7919) % max(len(rr), 1)
                key = lambda c: (pos[c] - start) % len(rr)  # noqa: E731
            elif policy == "oldest":
                key = lambda c: store[c][1]  # noqa: E731
            elif policy == "largest":
                key = lambda c: -len(cur[c])  # noqa: E731
            else:
                key = lambda c: -len(cur[c] - store[c][0])  # noqa: E731
            for c in sorted(cand, key=key):
                cost = tok(len(cur[c]))
                if cost > budget:
                    continue
                budget -= cost
                store[c] = (set(cur[c]), t)
            w = {c: tok(len(m)) for c, m in cur.items()}
            stale_curve.append(sum(w[c] * len(cur[c] - store[c][0]) / len(cur[c]) for c in cur) / sum(w.values()))
    return float(np.mean(stale_curve))


def main():
    out = {}
    plan = json.loads((HERE / "qa" / "plan.json").read_text())
    qa_labels = [np.asarray(x) for x in plan["methods"]["warmstart_leiden"]["labels"]]
    qa_t = np.load(HERE / "qa" / "node_t.npy")
    import entity_graph_eval as eg
    r = json.loads((HERE / "entity_graph" / "eval.json").read_text())
    eg_labels = [np.asarray(x) for x in r["warmstart_leiden|labels0"]]
    _, eg_t, _, _ = eg.stream(0)
    for name, labels, nt, MIN in (("qa_stream", qa_labels, qa_t, 5), ("entity_graph", eg_labels, eg_t, 10)):
        res = {}
        for f in (0.02, 0.05, 0.10, 0.20):
            res[str(f)] = {p: run(labels, nt, MIN, f, p) for p in ("jaccard", "round_robin", "oldest", "largest", "missing")}
            print(name, f, {k: round(v, 3) for k, v in res[str(f)].items()}, flush=True)
        out[name] = res
    (HERE / "refresh_sched.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
