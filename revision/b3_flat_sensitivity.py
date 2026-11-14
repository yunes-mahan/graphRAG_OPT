"""B3 (flat part, Table 4): sensitivity of the rankings to the report-cost model tok(S) = a + min(c, b|S|),
a = 1500, b = 250, c = 8000.  Each constant is changed by -50% and +50%, one at a time (6 variants).

Phase "base":   replay every Table 4 maintainer on the 2 development and 24 held-out orders (kgv2_calibrate replay,
                identical schedules) and store, per version, Q and the SIZES of all reported communities (>= 5 members)
                and, per update, the sizes of the communities whose report is rewritten (tau = 0.9 rule of
                tasks/kgversion2/evaluate.score_stream).  Any cost variant is then a function of these sizes.
                Check: under the base constants the recomputed Q and C equal the stored Table 4 inputs.
Phase "priced": the two rollback maintainers price reports with tok() inside the algorithm, so they are re-run per
                variant (env TOK_OVERHEAD/TOK_PER/TOK_CAP) with PRICE_LAMBDA = the variant's recalibrated lambda
                (mode "recal") and with 0.31 (mode "fixed").
Phase "analyze": lambda_v recalibrated exactly as kgv2_calibrate.calibrate (recompute and lazy tie on the two
                development orders); scores Q - lambda C per held-out group; ranking by the mean over the four
                held-out groups (Table 4 "mean"); Spearman and Kendall correlation with the base ranking.
Outputs: analysis/v2/b3/  (sizes per job) and analysis/v2/b3_flat_sensitivity.json
usage: python v2/b3_flat_sensitivity.py base|priced|analyze
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import igraph as ig
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import kgv2_calibrate as kc  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402

OUT = HERE / "b3"
OUT.mkdir(exist_ok=True)
DEV = [("arxiv", 2015, 2020, 0), ("collab", 2012, 2017, 0)]
ORDERS = DEV + list(kc.HELDOUT)
BASE_M = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "lazy", "hybrid", "local_repair", "graphrag_delta",
          "priced_warm", "priced_frontier"]
PRICED = ["priced_warm", "priced_frontier"]
BASE = (1500.0, 250.0, 8000.0)
VARIANTS = {"base": BASE, "a-50": (750.0, 250.0, 8000.0), "a+50": (2250.0, 250.0, 8000.0),
            "b-50": (1500.0, 125.0, 8000.0), "b+50": (1500.0, 375.0, 8000.0),
            "c-50": (1500.0, 250.0, 4000.0), "c+50": (1500.0, 250.0, 12000.0)}
MIN_SIZE, JACCARD_KEEP = 5, 0.9
ENVK = ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "TOK_OVERHEAD", "TOK_PER", "TOK_CAP",
        "PRICE_LAMBDA")


def sizes_of(labels, d, node_t, edge_t):
    """Same computation as tasks/kgversion2/evaluate.score_stream, keeping the community sizes."""
    edges, n_total = d["edges"], len(d["node_year"])
    SENT = np.iinfo(np.int64).min
    Q, full_sizes, regen_sizes = [], [], []
    prev_full = None
    for t, lab in enumerate(labels):
        nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        E = edges[:, edge_t <= t]
        remap = -np.ones(n_total, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        G = ig.Graph(n=len(nodes), edges=np.stack([remap[E[0]], remap[E[1]]], 1).tolist())
        uniq, inv_new = np.unique(lab, return_inverse=True)
        Q.append(float(G.modularity(inv_new.tolist())))
        size_new = np.bincount(inv_new)
        big = size_new >= MIN_SIZE
        full_sizes.append(size_new[big].tolist())
        full = np.full(n_total, SENT, dtype=np.int64)
        full[nodes] = lab
        if prev_full is not None:
            old_mask = prev_full != SENT
            ids_old = prev_full[old_mask]
            pos = np.clip(np.searchsorted(uniq, ids_old), 0, len(uniq) - 1)
            hit = uniq[pos] == ids_old
            size_old = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(size_old, pos[hit], 1)
            both = old_mask & (prev_full == full)
            inter = np.zeros(len(uniq), dtype=np.int64)
            np.add.at(inter, np.searchsorted(uniq, full[both]), 1)
            jac = inter / np.maximum(size_old + size_new - inter, 1)
            regen = big & ((size_old < MIN_SIZE) | (jac < JACCARD_KEEP))
            regen_sizes.append(size_new[regen].tolist())
        prev_full = full
    return dict(Q=Q, full_sizes=full_sizes, regen_sizes=regen_sizes)


def job(args):
    m, order, tag, env = args
    f = OUT / f"{m}__{tag}__{order[0]}_{order[1]}_{order[3]}.json"
    if f.exists():
        return str(f)
    stream, base, last, seed = order
    path, spec_env = rs.SPECS[m]
    for k in ENVK:
        os.environ.pop(k, None)
    os.environ.update(spec_env)
    os.environ.update(env)
    d = kc.ev.load(stream)
    node_t, edge_t, T = kc.ev.schedule(d, base, last, seed)
    labels, times = se.replay(str(path), d, node_t, edge_t, T)
    out = sizes_of(labels, d, node_t, edge_t)
    out["seconds"] = float(sum(times))
    f.write_text(json.dumps(out))
    return str(f)


def tok(sizes, v):
    a, b, c = v
    s = np.asarray(sizes, dtype=float)
    return float((a + np.minimum(c, b * s)).sum())


def load(m, tag, order):
    return json.loads((OUT / f"{m}__{tag}__{order[0]}_{order[1]}_{order[3]}.json").read_text())


def QC(m, tag, order, v):
    r, ref = load(m, tag, order), load("recompute_leiden", "base", order)
    C = [tok(rs_, v) / tok(fs, v) for rs_, fs in zip(r["regen_sizes"], ref["full_sizes"][1:])]
    return float(np.mean(r["Q"])), float(np.mean(C))


def lam_of(v):
    dq = np.mean([QC("recompute_leiden", "base", o, v)[0] - QC("lazy", "base", o, v)[0] for o in DEV])
    dc = np.mean([QC("recompute_leiden", "base", o, v)[1] - QC("lazy", "base", o, v)[1] for o in DEV])
    return float(dq / dc)


if __name__ == "__main__":
    mode = sys.argv[1]
    nproc = int(os.environ.get("NPROC", "14"))
    if mode == "base":
        jobs = [(m, o, "base", {}) for o in ORDERS for m in BASE_M]
        jobs.sort(key=lambda j: j[0] not in PRICED)        # long jobs first
        with Pool(nproc) as p:
            for i, f in enumerate(p.imap_unordered(job, jobs)):
                print(i, f, flush=True)
    elif mode == "priced":
        jobs = []
        for vn, v in VARIANTS.items():
            if vn == "base":
                continue
            env = dict(TOK_OVERHEAD=str(v[0]), TOK_PER=str(v[1]), TOK_CAP=str(v[2]))
            for lam_mode, lam in (("recal", lam_of(v)),):     # fixed-lambda check re-scores the base runs (below)
                for m in PRICED:
                    for o in ORDERS:
                        jobs.append((m, o, f"{vn}_{lam_mode}", dict(env, PRICE_LAMBDA=repr(lam))))
        with Pool(nproc) as p:
            for i, f in enumerate(p.imap_unordered(job, jobs)):
                print(i, f, flush=True)
    else:
        from scipy import stats
        # 1. exact reproduction of the stored Table 4 inputs under the base constants
        cal = json.loads((HERE.parent / "kgv2_calibration.json").read_text())
        H = {}
        for hf in sorted((HERE.parent).glob("kgv2_heldout*.json")):
            if "agent" in hf.name:
                continue
            for k, v in json.loads(hf.read_text()).items():
                for m, x in v.items():
                    if m != "_ref" and x:
                        H.setdefault(k, {})[m] = x
        maxdiff = 0.0
        for o in kc.HELDOUT:
            key = f"{o[0]}|{o[1]}|{o[3]}"
            for m in BASE_M:
                q, c = QC(m, "base", o, BASE)
                maxdiff = max(maxdiff, abs(q - H[key][m]["Q"]), abs(c - H[key][m]["C"]))
        lam0 = lam_of(BASE)
        groups = [("arxiv", 2015), ("collab", 2012), ("arxiv", 2017), ("collab", 2010)]
        res = dict(base_check=dict(max_abs_diff_QC_vs_stored=maxdiff, lambda_recomputed=lam0, lambda_stored=cal["lambda_"]),
                   variants={})

        def table(v, lam, tagf):
            sc = {}
            for m in BASE_M:
                g = []
                for s, b in groups:
                    os_ = [o for o in kc.HELDOUT if o[0] == s and o[1] == b]
                    g.append(np.mean([QC(m, tagf(m), o, v)[0] - lam * QC(m, tagf(m), o, v)[1] for o in os_]))
                sc[m] = dict(groups=[float(x) for x in g], mean=float(np.mean(g)))
            return sc

        base_tab = table(BASE, cal["lambda_"], lambda m: "base")
        order0 = sorted(BASE_M, key=lambda m: -base_tab[m]["mean"])
        res["base_ranking"] = order0
        res["base_scores"] = base_tab
        for vn, v in VARIANTS.items():
            if vn == "base":
                continue
            out = {}
            for lam_mode, lam in (("recal", lam_of(v)), ("fixed", 0.31)):
                # recal: rollback re-run with tok_v and lambda_v; fixed: lambda = 0.31 and the rollback runs priced with
                # the base cost model, re-scored under tok_v (pricing not re-run; labelled in the output)
                tab = table(v, lam, lambda m, vn=vn, lm=lam_mode: f"{vn}_{lm}" if (m in PRICED and lm == "recal") else "base")
                rank = sorted(BASE_M, key=lambda m: -tab[m]["mean"])
                x = [base_tab[m]["mean"] for m in BASE_M]
                y = [tab[m]["mean"] for m in BASE_M]
                per_group = []
                for gi in range(len(groups)):
                    xg = [base_tab[m]["groups"][gi] for m in BASE_M]
                    yg = [tab[m]["groups"][gi] for m in BASE_M]
                    per_group.append(float(stats.kendalltau(xg, yg).statistic))
                out[lam_mode] = dict(lambda_=lam, ranking=rank, scores=tab,
                                     rollback_pricing=("re-run with tok_v, lambda_v" if lam_mode == "recal" else "base pricing, re-scored"),
                                     spearman=float(stats.spearmanr(x, y).statistic),
                                     kendall=float(stats.kendalltau(x, y).statistic),
                                     kendall_per_group=per_group,
                                     top1_same=rank[0] == order0[0], top3_same=set(rank[:3]) == set(order0[:3]))
            res["variants"][vn] = out
            print(vn, {k: (round(o["lambda_"], 3), round(o["spearman"], 3), round(o["kendall"], 3), o["ranking"][:3])
                       for k, o in out.items()}, flush=True)
        print("base check", res["base_check"])
        (HERE / "b3_flat_sensitivity.json").write_text(json.dumps(res, indent=1))
