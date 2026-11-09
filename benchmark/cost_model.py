"""LLM summary-cost model on ogbn-arxiv using the real paper texts (no LLM calls).

For each maintainer (update order 0) we replay the stream, find every community summary that must be
(re)generated per update (tau = 0.9, communities with >= 2 members), and estimate tokens:
  input  = PROMPT_OVERHEAD + min(CONTEXT, sum of member text tokens)   (member text = title + abstract)
  output = OUTPUT_TOKENS
Tokens are approximated as characters / 4. Costs are reported for two price points (USD / 1M tokens).
All assumptions are explicit parameters; nothing here calls an LLM.
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402


import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

HERE = config.RESULTS
PROMPT_OVERHEAD = 1000
OUTPUT_TOKENS = 500
CONTEXTS = (4000, 8000, 16000)
PRICES = {"mid-tier ($2/$12 per 1M)": (2.0, 12.0), "low-cost ($0.20/$1.20 per 1M)": (0.20, 1.20)}
METHODS = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "local_repair", "hybrid", "lazy", "place_drift"]


def node_tokens():
    mp = pd.read_csv(config.OGB_ROOT / "ogbn_arxiv" / "mapping" / "nodeidx2paperid.csv.gz")
    with gzip.open(config.TEXTS, "rt", encoding="utf-8", errors="replace") as f:
        chars = {}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].isdigit():
                chars[int(parts[0])] = len(parts[1]) + len(parts[2])
    tok = np.array([chars.get(int(p), 0) / 4.0 for p in mp["paper id"].values])
    print(f"texts matched: {(tok > 0).mean():.3%}, mean tokens/paper: {tok[tok > 0].mean():.0f}")
    return tok


def regenerated(prev, cur, min_size=2):
    """IDs of communities (size >= min_size) whose summary must be regenerated (tau = 0.9)."""
    out = []
    for k, mem in cur.items():
        if len(mem) < min_size:
            continue
        old = prev.get(k)
        if old is None or len(mem & old) / len(mem | old) < 0.9:
            out.append(k)
    return out


def main():
    tok = node_tokens()
    d = se.load("arxiv")
    node_t, edge_t, T = se.schedule(d, 0)
    res = {}
    for m in METHODS:
        path, env = rs.SPECS[m]
        import os
        for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K"):
            os.environ.pop(k, None)
        os.environ.update(env)
        labels, _ = se.replay(str(path), d, node_t, edge_t, T)
        prev = None
        MINS = (2, 5, 10, 20)
        acc = {ms: dict(inp=[], out=[], n=[], cov=[], ncomm=[]) for ms in MINS}
        for t, lab in enumerate(labels):
            nodes = np.where((node_t >= 0) & (node_t <= t))[0]
            cur = {}
            for v, c in zip(nodes, lab):
                cur.setdefault(int(c), set()).add(int(v))
            if prev is not None:
                for ms in MINS:
                    regen = regenerated(prev, cur, ms)
                    mem_tok = [tok[list(cur[k])].sum() for k in regen]
                    acc[ms]["inp"].append(sum(PROMPT_OVERHEAD + min(8000, x) for x in mem_tok))
                    acc[ms]["out"].append(OUTPUT_TOKENS * len(regen))
                    acc[ms]["n"].append(len(regen))
                    acc[ms]["cov"].append(sum(len(m_) for m_ in cur.values() if len(m_) >= ms) / len(nodes))
                    acc[ms]["ncomm"].append(sum(1 for m_ in cur.values() if len(m_) >= ms))
            prev = cur
        r = {}
        for ms in MINS:
            inp, out = float(np.mean(acc[ms]["inp"])), float(np.mean(acc[ms]["out"]))
            r[f"min{ms}"] = dict(summaries_per_update=float(np.mean(acc[ms]["n"])), input_tokens=inp, output_tokens=out,
                                 coverage=float(np.mean(acc[ms]["cov"])), communities=float(np.mean(acc[ms]["ncomm"])),
                                 usd={p_: float(inp / 1e6 * a + out / 1e6 * b) for p_, (a, b) in PRICES.items()})
        res[m] = r
        print(m, " | ".join(f"min{ms}: n={r[f'min{ms}']['summaries_per_update']:.0f} "
              f"${r[f'min{ms}']['usd']['mid-tier ($2/$12 per 1M)']:.2f} cov={r[f'min{ms}']['coverage']:.2f} "
              f"comms={r[f'min{ms}']['communities']:.0f}" for ms in (2, 5, 10, 20)), flush=True)
    (HERE / "cost_model.json").write_text(json.dumps(dict(
        assumptions=dict(prompt_overhead=PROMPT_OVERHEAD, output_tokens=OUTPUT_TOKENS, contexts=CONTEXTS,
                         tokens_per_char=0.25, tau=0.9, min_size=2, prices=PRICES, dataset="ogbn-arxiv",
                         order=0), results=res), indent=1))


if __name__ == "__main__":
    main()
