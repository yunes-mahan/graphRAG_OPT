"""B5: Pareto view of the flat maintainers on the held-out streams (main flat result; the Q - 0.31 C score is secondary).

Per held-out group (arXiv-15, collab-12, arXiv-17, collab-10) and maintainer: mean and 95% t-interval over orders of
  Q   mean modularity over versions                          (stored Table 4 inputs, analysis/kgv2_heldout*.json)
  C   mean relative report-token cost per update             (same)
  cov mean share of present nodes in a reported community (>= 5 members), over versions
      (from the community sizes recorded by v2/b3_flat_sensitivity.py base; Sahu's DF Leiden from v2/dfleiden_work)
Pareto-optimal maintainers (no other maintainer has >= Q (resp. cov) and <= C with one strict), and the ranking by the
held-out score for lambda in [0, 1.5]; the rollback variants stay priced at lambda = 0.31 (only the scoring lambda varies).
Output: analysis/v2/b5_pareto.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "programs"))
import kgv2_calibrate as kc  # noqa: E402
from hybrid_refresh import map_to_existing_ids  # noqa: E402
import b3_flat_sensitivity as b3  # noqa: E402

GROUPS = [("arxiv", 2015, "arXiv-15"), ("collab", 2012, "collab-12"), ("arxiv", 2017, "arXiv-17"), ("collab", 2010, "collab-10")]
MS = b3.BASE_M + ["df_leiden_sahu"]


def stored():
    H = {}
    for hf in sorted(HERE.parent.glob("kgv2_heldout*.json")) + [HERE / "kgv2_heldout_df_leiden_sahu.json"]:
        if "agent" in hf.name:
            continue
        for k, v in json.loads(hf.read_text()).items():
            for m, x in v.items():
                if m != "_ref" and x:
                    H.setdefault(k, {})[m] = x
    return H


def sahu_sizes(order):
    stream, base, last, seed = order
    d = kc.ev.load(stream)
    node_t, edge_t, T = kc.ev.schedule(d, base, last, seed)
    raw = [np.array(l.split(), dtype=np.int64) for l in (HERE / "dfleiden_work" / f"{stream}_{base}_{seed}.out").read_text().splitlines()]
    label, nxt, labels = {}, 0, []
    for t in range(T):
        nodes = [int(v) for v in np.where((node_t >= 0) & (node_t <= t))[0]]
        memb = raw[t][nodes]
        if t == 0:
            _, comp = np.unique(memb, return_inverse=True)
            label, nxt = {v: int(c) for v, c in zip(nodes, comp)}, int(comp.max()) + 1
        else:
            mp, nxt = map_to_existing_ids(nodes, memb.tolist(), label, nxt)
            label = {v: mp[int(c)] for v, c in zip(nodes, memb)}
        labels.append(np.asarray([label[v] for v in nodes], dtype=np.int64))
    return b3.sizes_of(labels, d, node_t, edge_t)


def coverage(sz, order):
    stream, base, last, seed = order
    d = kc.ev.load(stream)
    node_t, _, T = kc.ev.schedule(d, base, last, seed)
    n_present = [int(((node_t >= 0) & (node_t <= t)).sum()) for t in range(T)]
    return float(np.mean([sum(fs) / n for fs, n in zip(sz["full_sizes"], n_present)]))


def ci(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def pareto(pts):
    """pts: {m: (gain, cost)}; maximise gain, minimise cost."""
    return sorted(m for m, (g, c) in pts.items()
                  if not any((g2 >= g and c2 <= c and (g2 > g or c2 < c)) for m2, (g2, c2) in pts.items() if m2 != m))


if __name__ == "__main__":
    H = stored()
    lam_cal = json.loads((HERE.parent / "kgv2_calibration.json").read_text())["lambda_"]
    res = dict(groups={}, lambda_calibrated=lam_cal)
    covc = {}
    for s, b, name in GROUPS:
        orders = [o for o in kc.HELDOUT if o[0] == s and o[1] == b]
        g = {}
        for m in MS:
            keys = [f"{o[0]}|{o[1]}|{o[3]}" for o in orders]
            Q = [H[k][m]["Q"] for k in keys]
            C = [H[k][m]["C"] for k in keys]
            cov = [coverage(sahu_sizes(o) if m == "df_leiden_sahu" else b3.load(m, "base", o), o) for o in orders]
            g[m] = dict(Q=ci(Q), C=ci(C), cov=ci(cov), n=len(orders), Q_orders=Q, C_orders=C, cov_orders=cov)
        g_par_q = pareto({m: (g[m]["Q"][0], g[m]["C"][0]) for m in MS})
        g_par_c = pareto({m: (g[m]["cov"][0], g[m]["C"][0]) for m in MS})
        res["groups"][name] = dict(maintainers=g, pareto_Q_C=g_par_q, pareto_cov_C=g_par_c)
        print(name, "Pareto(Q,C):", g_par_q, "Pareto(cov,C):", g_par_c, flush=True)
    lams = [round(x, 2) for x in np.arange(0, 1.501, 0.01)]
    rk = {}
    for l in lams:
        sc = {m: float(np.mean([res["groups"][n]["maintainers"][m]["Q"][0] - l * res["groups"][n]["maintainers"][m]["C"][0]
                                for _, _, n in GROUPS])) for m in MS}
        rk[str(l)] = dict(scores=sc, ranking=sorted(MS, key=lambda m: -sc[m]))
    res["lambda_sweep"] = rk
    winners = {}
    for l in lams:
        w = rk[str(l)]["ranking"][0]
        winners.setdefault(w, []).append(l)
    res["winner_intervals"] = {m: [min(v), max(v)] for m, v in winners.items()}
    per_group_w = {}
    for _, _, n in GROUPS:
        ww = {}
        for l in lams:
            sc = {m: res["groups"][n]["maintainers"][m]["Q"][0] - l * res["groups"][n]["maintainers"][m]["C"][0] for m in MS}
            ww.setdefault(max(sc, key=sc.get), []).append(l)
        per_group_w[n] = {m: [min(v), max(v)] for m, v in ww.items()}
    res["winner_intervals_per_group"] = per_group_w
    print("winners over lambda (held-out mean):", res["winner_intervals"])
    print("per group:", per_group_w)
    (HERE / "b5_pareto.json").write_text(json.dumps(res, indent=1))
