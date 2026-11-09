"""QA experiment, stage 1 (no LLM calls): build the Data&Knowledge subgraph stream, run maintainers,
simulate summary stores under staleness tolerances, and count the unique summaries needed.

Subgraph: ogbn-arxiv papers in cs.IR, cs.DB, cs.SI, cs.DL (induced citation edges).
Stream: base = papers up to 2017, then 2018, 2019, 2020 in 2 random batches each (6 updates).
Summary store: communities with >= MIN_SIZE members carry a summary. At every update a community's
summary is regenerated if the community is new or Jaccard(members at last summary, members now) < tau;
otherwise the old summary (written for the old member set) is kept. tau = 0 means "never refresh"
(only brand-new communities get a summary).
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402


import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

HERE = config.RESULTS
OUT = HERE / "qa"
OUT.mkdir(exist_ok=True)
CATS = {31: "cs.IR", 37: "cs.DB", 26: "cs.SI", 38: "cs.DL"}
BASE, LAST, BATCHES, SEED = 2017, 2020, 2, 0
MIN_SIZE = 5
TAUS = [0.9, 0.7, 0.5, 0.0]
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid", "local_repair"]


def load_texts(paper_ids):
    want = set(int(p) for p in paper_ids)
    out = {}
    with gzip.open(config.TEXTS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) in want:
                out[int(parts[0])] = (parts[1].strip(), parts[2].strip())
    return out


def build_subgraph():
    full = se.load("arxiv")
    keep = np.isin(full["label"], list(CATS))
    old2new = -np.ones(len(keep), dtype=np.int64)
    old2new[keep] = np.arange(keep.sum())
    e = full["edges"]
    m = keep[e[0]] & keep[e[1]]
    edges = old2new[e[:, m]]
    year = full["node_year"][keep]
    d = dict(edges=edges, edge_year=np.maximum(year[edges[0]], year[edges[1]]), node_year=year,
             feat=full["feat"][keep], label=full["label"][keep], base=BASE, last=LAST, edge_mode="nodes")
    mp = pd.read_csv(config.OGB_ROOT / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz")
    paper_id = mp["paper id"].values[keep]
    return d, paper_id


def schedule(d):
    se_batches = se.BATCHES
    se.BATCHES = BATCHES
    try:
        return se.schedule(d, SEED)
    finally:
        se.BATCHES = se_batches


def summary_events(labels, node_t, tau):
    """Return list of (version, frozenset members) for every summary generation, and final store."""
    store = {}  # cid -> frozenset of members the current summary was written for
    events = []
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        cur = {}
        for v, c in zip(nodes, lab):
            cur.setdefault(int(c), set()).add(int(v))
        for cid, mem in cur.items():
            if len(mem) < MIN_SIZE:
                continue
            old = store.get(cid)
            regen = old is None or (t > 0 and tau > 0 and len(mem & old) / len(mem | old) < tau)
            if regen:
                fm = frozenset(mem)
                store[cid] = fm
                events.append((t, fm))
        for cid in list(store):  # communities that vanished or shrank below MIN_SIZE lose their summary
            if cid not in cur or len(cur[cid]) < MIN_SIZE:
                del store[cid]
    return events, store


def main():
    d, paper_id = build_subgraph()
    node_t, edge_t, T = schedule(d)
    print(f"subgraph: {len(paper_id)} papers, {d['edges'].shape[1]} edges, versions {T}, "
          f"base papers {(node_t == 0).sum()}")
    texts = load_texts(paper_id)
    print(f"texts: {len(texts)}/{len(paper_id)}")
    plan = dict(methods={}, T=T)
    all_sets = set()
    for m in METHODS:
        path, env = rs.SPECS[m]
        for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K"):
            os.environ.pop(k, None)
        os.environ.update(env)
        labels, _ = se.replay(str(path), d, node_t, edge_t, T)
        sc = se.score(labels, d, node_t, edge_t)
        plan["methods"][m] = dict(Q=sc["Q"], R=sc["R"], labels=[lab.tolist() for lab in labels], taus={})
        for tau in TAUS:
            ev, store = summary_events(labels, node_t, tau)
            plan["methods"][m]["taus"][str(tau)] = dict(n_generations=len(ev),
                                                        final_store=[sorted(s) for s in store.values()],
                                                        events=[(t, sorted(s)) for t, s in ev])
            all_sets.update(s for _, s in ev)
            print(f"{m:17s} tau={tau}: generations={len(ev):4d} final summaries={len(store):4d}  Q={sc['Q']:.3f} R={sc['R']:.3f}")
    # token estimate per unique member set (title + abstract, truncated to 6000 tokens + 600 instructions)
    tok = {i: (len(texts.get(int(paper_id[i]), ("", ""))[0]) + len(texts.get(int(paper_id[i]), ("", ""))[1])) / 4
           for i in range(len(paper_id))}
    inp = sum(600 + min(6000, sum(tok[v] for v in s)) for s in all_sets)
    out = 350 * len(all_sets)
    print(f"\nunique summaries to generate: {len(all_sets)}; est. input {inp/1e6:.2f}M, output {out/1e6:.2f}M tokens")
    plan["unique_sets"] = len(all_sets)
    plan["est_tokens"] = dict(input=inp, output=out)
    np.save(OUT / "node_t.npy", node_t)
    (OUT / "plan.json").write_text(json.dumps(plan))
    pd.DataFrame({"idx": range(len(paper_id)), "paper_id": paper_id, "year": d["node_year"],
                  "label": d["label"]}).to_csv(OUT / "nodes.csv", index=False)


if __name__ == "__main__":
    main()
