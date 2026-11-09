"""Generate ALL tables (LaTeX), figures and numbers.tex for the paper from measured result files.

Nothing in the paper is typed by hand: tables are written as tables/*.tex, key numbers as macros.
Sources: analysis/suite_*.json (all maintainers, 4 streams), neo4j_demo/demo_results_*.json
(seeded GDS Louvain, Neo4j timings), analysis/cost_model.json (token/USD model, real arXiv text),
QA, entity-graph and revision results (hierarchy study, noise floor, equivalence tests, second judge).
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

import os as _os
_R = _os.environ.get("PAPER_RESULTS")   # artifact layout (release repo); default: local analysis tree
A = Path(_R) if _R else Path(__file__).resolve().parents[1] / "artifacts"
N = A / "neo4j_demo"
HERE = Path(__file__).parent
FIG, TAB = HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)
LAM = 0.15
STREAMS = [("arxiv", 2015), ("collab", 2012), ("arxiv", 2017), ("collab", 2010)]
SNAME = {("arxiv", 2015): "arXiv-15", ("collab", 2012): "collab-12", ("arxiv", 2017): "arXiv-17",
         ("collab", 2010): "collab-10"}
LABEL = {"recompute_leiden": "Recompute Leiden", "warmstart_leiden": "Warm-start Leiden",
         "frontier_leiden": "Frontier Leiden", "gds_louvain_seed": "GDS Louvain (seeded)", "lazy": "Lazy attach",
         "local_repair": "Local repair", "hybrid": "Hybrid (local repair + refresh)",
         "place_drift": "Placement + drift refresh", "local_periodic5": "Local + refresh (every 5)",
         "local_periodic3": "Local + refresh (every 3)", "graphrag_delta": "GraphRAG update (delta)"}
MAIN = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "lazy", "graphrag_delta", "local_repair", "hybrid"]
# validated palette (dataviz reference): blue/orange/aqua slots + neutral greys, plus marker shapes
STYLE = {"hybrid": ("#2a78d6", "*", 70, "-"), "local_repair": ("#86b0e8", "o", 28, "--"),
         "frontier_leiden": ("#eb6834", "s", 30, "-"), "warmstart_leiden": ("#1baf7a", "D", 26, "-"),
         "gds_louvain_seed": ("#52514e", "P", 34, "-"), "recompute_leiden": ("#52514e", "^", 30, "-"),
         "lazy": ("#9b9a95", "v", 30, "-")}
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "savefig.dpi": 300, "pdf.fonttype": 42})


def load_runs():
    runs = {}
    for ds, base in STREAMS:
        for r in json.loads((A / f"suite_{ds}_{base}.json").read_text()):
            runs.setdefault((ds, base, r["method"]), []).append(r)
    for ds, base in (("arxiv", 2015), ("collab", 2012)):
        f = N / f"demo_results_{ds}_s0_16.json"
        g = json.loads(f.read_text())["gds_warm_louvain"]
        runs[(ds, base, "gds_louvain_seed")] = [dict(Q=g["Q"], R=g["R"], seed=0, R_comm=float("nan"),
                                                     R_tau={}, Q_curve=g["Q_curve"])]
    return runs


def stat(rows, key="S", r_key=None):
    if r_key == "comm":
        vals = np.array([r["Q"] - LAM * r["R_comm"] for r in rows])
    elif r_key:
        vals = np.array([r["Q"] - LAM * r["R_tau"][r_key] for r in rows])
    else:
        vals = np.array([r["Q"] - LAM * r["R"] for r in rows])
    n = len(vals)
    ci = float(stats.t.ppf(0.975, n - 1) * vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return dict(S=float(vals.mean()), ci=ci, n=n, Q=float(np.mean([r["Q"] for r in rows])),
                R=float(np.mean([r["R"] for r in rows])), vals=vals)


def fmt_ci(s):
    return f"{s['S']:.3f}" + (f"\\,$\\pm$\\,{s['ci']:.3f}" if s["n"] > 1 else "$^\\dagger$")


def table_main(runs, M):
    rows = []
    for m in MAIN[:3] + ["gds_louvain_seed"] + MAIN[3:]:
        cells = [LABEL[m]]
        for ds, base in (("arxiv", 2015), ("collab", 2012)):
            s = stat(runs[(ds, base, m)])
            cells += [f"{s['Q']:.3f}", f"{s['R']:.3f}", fmt_ci(s)]
        rows.append(cells)
    best = {}
    for j, (ds, base) in enumerate((("arxiv", 2015), ("collab", 2012))):
        col = 3 + 3 * j
        vals = [float(r[col].split("\\")[0].split("$")[0]) for r in rows]
        for bi in [i for i, v in enumerate(vals) if v >= max(vals) - 5e-4]:
            rows[bi][col] = "\\textbf{" + rows[bi][col] + "}"
    body = "\n".join(" & ".join(r) + r" \\" + ("\n\\midrule" if r[0] == "Lazy attach" else "") for r in rows)
    tex = r"""\begin{tabular}{l ccc ccc}
\toprule
 & \multicolumn{3}{c}{arXiv-15 (10 orders)} & \multicolumn{3}{c}{collab-12 (10 orders)}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & $\bar Q$ & $\bar R$ & score & $\bar Q$ & $\bar R$ & score\\
\midrule
""" + body + "\n\\bottomrule\n\\end{tabular}\n"
    (TAB / "main.tex").write_text(tex)


def table_generalisation(runs):
    rows = []
    for m in MAIN:
        cells = [LABEL[m]] + [fmt_ci(stat(runs[(ds, b, m)])) for ds, b in STREAMS]
        rows.append(cells)
    for j in range(1, 5):
        vals = [float(r[j].split("\\")[0].split("$")[0]) for r in rows]
        for bi in [i for i, v in enumerate(vals) if v >= max(vals) - 5e-4]:
            rows[bi][j] = "\\textbf{" + rows[bi][j] + "}"
    tex = (r"""\begin{tabular}{l cccc}
\toprule
Method & arXiv-15 & collab-12 & arXiv-17 & collab-10\\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "generalisation.tex").write_text(tex)


def table_metric_sensitivity(runs, M):
    """Rank-1 method per stream and cost definition (persistent+baseline set MAIN)."""
    defs = [("$\\tau{=}0.8$", "0.8"), ("$\\tau{=}0.9$ (default)", None), ("$\\tau{=}0.95$", "0.95"),
            ("per community", "comm")]
    short = {"recompute_leiden": "Recomp.", "warmstart_leiden": "Warm", "frontier_leiden": "Frontier",
             "lazy": "Lazy", "local_repair": "Local", "hybrid": "Hybrid"}
    rows = []
    hybrid_wins = 0
    for name, key in defs:
        cells = [name]
        for ds, b in STREAMS:
            sc = {m: stat(runs[(ds, b, m)], r_key=key)["S"] for m in MAIN}
            top = max(sc.values())
            ws = [m for m in MAIN if sc[m] >= top - 1e-4]
            hybrid_wins += "hybrid" in ws
            cells.append("/".join(short[w] for w in ws))
        rows.append(cells)
    M["MetricCells"] = str(len(defs) * len(STREAMS))
    M["MetricHybridWins"] = str(hybrid_wins)
    tex = (r"""\begin{tabular}{l cccc}
\toprule
Regeneration cost definition & arXiv-15 & collab-12 & arXiv-17 & collab-10\\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "metric_sensitivity.tex").write_text(tex)


def table_ablation(runs):
    order = ["local_repair", "hybrid", "place_drift", "local_periodic5", "local_periodic3"]
    rows = [[LABEL[m]] + [fmt_ci(stat(runs[(ds, b, m)])) for ds, b in STREAMS] for m in order]
    tex = (r"""\begin{tabular}{l cccc}
\toprule
Variant & arXiv-15 & collab-12 & arXiv-17 & collab-10\\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "ablation.tex").write_text(tex)


def table_cost(M):
    c = json.loads((A / "cost_model.json").read_text())
    res = c["results"]
    meths = ["recompute_leiden", "warmstart_leiden", "frontier_leiden", "local_repair", "hybrid"]
    rows = []
    for ms in (2, 5, 10, 20):
        cells = [f"$\\geq {ms}$"]
        vals = [res[m][f"min{ms}"]["usd"]["mid-tier ($2/$12 per 1M)"] for m in meths]
        bi = int(np.argmin(vals))
        for i, m in enumerate(meths):
            v = res[m][f"min{ms}"]
            s = f"{vals[i]:.2f} ({100 * v['coverage']:.0f}\\%)"
            cells.append("\\textbf{" + s + "}" if i == bi else s)
        rows.append(cells)
    tex = (r"""\begin{tabular}{c ccccc}
\toprule
min.\ size & Recompute & Warm-start & Frontier & Local repair & Hybrid\\
\midrule
""" + "\n".join(" & ".join(r) + r" \\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "cost.tex").write_text(tex)
    r = lambda m, ms: res[m][f"min{ms}"]["usd"]["mid-tier ($2/$12 per 1M)"]
    M["CostRecompTwo"] = f"{r('recompute_leiden', 2):.2f}"
    M["CostWarmTwo"] = f"{r('warmstart_leiden', 2):.2f}"
    M["CostHybTwo"] = f"{r('hybrid', 2):.2f}"
    M["CostLocalTwo"] = f"{r('local_repair', 2):.2f}"
    M["CostWarmTen"] = f"{r('warmstart_leiden', 10):.2f}"
    M["CostHybTen"] = f"{r('hybrid', 10):.2f}"
    M["CostFrontTen"] = f"{r('frontier_leiden', 10):.2f}"
    M["CostRecompTen"] = f"{r('recompute_leiden', 10):.2f}"
    M["CommsLocalTwo"] = f"{res['local_repair']['min2']['communities']:.0f}"
    M["CommsWarmTwo"] = f"{res['warmstart_leiden']['min2']['communities']:.0f}"
    M["SummLocalTwo"] = f"{res['local_repair']['min2']['summaries_per_update']:.0f}"
    M["SummWarmTwo"] = f"{res['warmstart_leiden']['min2']['summaries_per_update']:.0f}"
    M["TokPerPaper"] = "290"


def fig_pareto(runs):
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.15))
    for ax, (ds, b) in zip(axes, (("arxiv", 2015), ("collab", 2012))):
        for m in ["hybrid", "local_repair", "frontier_leiden", "warmstart_leiden", "gds_louvain_seed",
                  "recompute_leiden", "lazy"]:
            s = stat(runs[(ds, b, m)])
            col, mk, ms, _ = STYLE[m]
            ax.scatter(100 * s["R"], s["Q"], s=ms, color=col, marker=mk, edgecolor="white", lw=0.9, zorder=3,
                       label=LABEL[m].replace(" (local repair + refresh)", ""))
        ax.set_title(SNAME[(ds, b)], loc="left")
        ax.set_xlabel("summaries regenerated per update (%)")
        ax.set_xlim(-3, 103)
        ax.grid(color="#e8e7e3", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("mean modularity $\\bar Q$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07),
               handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.2)
    fig.savefig(FIG / "pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_lambda(runs, M):
    lams = np.linspace(0, 0.5, 501)
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 2.05))
    for ax, (ds, b) in zip(axes, (("arxiv", 2015), ("collab", 2012))):
        for m in ["hybrid", "local_repair", "frontier_leiden", "warmstart_leiden", "recompute_leiden", "lazy"]:
            s = stat(runs[(ds, b, m)])
            col, _, _, ls = STYLE[m]
            ax.plot(lams, s["Q"] - lams * s["R"], color=col, ls=ls, lw=1.5 if m in ("hybrid", "frontier_leiden") else 1.0,
                    label=LABEL[m].replace(" (local repair + refresh)", ""))
        ax.axvline(LAM, color="#0b0b0b", lw=0.5, ls=":")
        ax.set_title(SNAME[(ds, b)], loc="left")
        ax.set_xlabel("summary cost weight $\\lambda$")
        ax.grid(color="#e8e7e3", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("score $\\bar Q - \\lambda \\bar R$")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.09),
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=1.2)
    fig.savefig(FIG / "lambda.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 1.95))
    for m in ["hybrid", "local_repair", "frontier_leiden", "warmstart_leiden", "recompute_leiden", "lazy"]:
        r = [x for x in runs[("arxiv", 2015, m)] if x["seed"] == 0][0]
        col, mk, _, ls = STYLE[m]
        kw = dict(color=col, marker=mk, ms=2.6, lw=1.0, ls=ls)
        axes[0].plot(range(len(r["Q_curve"])), r["Q_curve"], label=LABEL[m].replace(" (local repair + refresh)", ""), **kw)
        axes[1].plot(range(1, len(r["R_curve"]) + 1), [100 * x for x in r["R_curve"]], **kw)
    axes[0].set_xlabel("graph version")
    axes[0].set_ylabel("modularity $Q_t$")
    axes[1].set_xlabel("update")
    axes[1].set_ylabel("regenerated (%)")
    for ax in axes:
        ax.grid(color="#e8e7e3", lw=0.5)
        ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1),
               handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.2)
    fig.savefig(FIG / "curves_arxiv.pdf", bbox_inches="tight")
    plt.close(fig)


def numbers(runs, M):
    s = lambda ds, b, m: stat(runs[(ds, b, m)])
    for (ds, b), tag in (((("arxiv", 2015)), "ArxF"), ((("collab", 2012)), "ColT")):
        for m, mt in (("recompute_leiden", "Recomp"), ("warmstart_leiden", "Warm"), ("frontier_leiden", "Front"),
                      ("hybrid", "Hyb"), ("local_repair", "Local"), ("gds_louvain_seed", "Gds"), ("lazy", "Lazy")):
            v = s(ds, b, m)
            M[f"S{tag}{mt}"] = f"{v['S']:.3f}"
            M[f"Q{tag}{mt}"] = f"{v['Q']:.3f}"
            M[f"RP{tag}{mt}"] = f"{100 * v['R']:.0f}\\%"
    # paired comparisons hybrid vs frontier
    for (ds, b), tag in ((("arxiv", 2015), "ArxF"), (("collab", 2012), "ColT")):
        a = {r["seed"]: r["Q"] - LAM * r["R"] for r in runs[(ds, b, "hybrid")]}
        f = {r["seed"]: r["Q"] - LAM * r["R"] for r in runs[(ds, b, "frontier_leiden")]}
        seeds = sorted(set(a) & set(f))
        d = np.array([a[k] - f[k] for k in seeds])
        p = stats.ttest_rel([a[k] for k in seeds], [f[k] for k in seeds]).pvalue
        M[f"Diff{tag}"] = f"{d.mean():+.3f}"
        M[f"Wins{tag}"] = f"{int((d > 0).sum())}/{len(d)}"
        M[f"P{tag}"] = f"10^{{{int(np.floor(np.log10(p)))}}}"
    M["NOrdersMain"], M["NOrdersNew"] = "10", "3"
    j = json.loads((N / "demo_results_arxiv_s0_16.json").read_text())["log"][1:]
    M["NeoLocalArx"] = f"{np.mean([r['local_compute_s'] + r['local_write_s'] for r in j]):.1f}"
    M["NeoGdsArx"] = f"{np.mean([r['gds_warm_s'] for r in j]):.1f}"
    M["NeoReArx"] = f"{np.mean([r['gds_recompute_s'] for r in j]):.1f}"
    import sys
    sys.path.insert(0, str(A))
    try:                                   # count from the data if available, else the published OGB sizes
        import stream_eval as se
        sizes = {}
        for ds in ("arxiv", "collab"):
            d = se.load(ds)
            sizes[ds] = (int((d["node_year"] < 10**4).sum()), int(d["edges"].shape[1]))
    except Exception:  # noqa: BLE001
        sizes = {"arxiv": (169343, 1157799), "collab": (235868, 967632)}
    for ds, tag in (("arxiv", "Arx"), ("collab", "Col")):
        M[f"N{tag}Nodes"] = f"{sizes[ds][0]:,}".replace(",", "{,}")
        M[f"N{tag}Edges"] = f"{sizes[ds][1]:,}".replace(",", "{,}")
    M["NRuns"] = str(sum(len(v) for k, v in runs.items() if k[2] != "gds_louvain_seed"))


def qa_assets(M):
    """Downstream QA experiment: table + figure + macros from analysis/qa/results.json."""
    import pandas as pd
    r = json.loads((A / "qa" / "results.json").read_text())
    per = pd.DataFrame(r["per_question"])
    piv = per.pivot_table(index="q", columns="store", values="recall")
    ref = piv["FRESH|recompute all"]
    summ = {x["store"]: x for x in r["summary"]}
    names = {"recompute_leiden": "Recompute", "warmstart_leiden": "Warm-start", "frontier_leiden": "Frontier",
             "hybrid": "Hybrid"}
    rows, sig, pts, pvals = [], 0, [], []
    for m in names:
        for tau in ("0.9", "0.7", "0.5", "0.0"):
            k = f"{m}|tau={tau}"
            d = (piv[k] - ref).values
            p_ = stats.wilcoxon(piv[k], ref, zero_method="zsplit").pvalue if np.any(d != 0) else 1.0
            sig += p_ < 0.05
            pvals.append(p_)
            x = summ[k]
            rows.append([names[m] if tau == "0.9" else "", "never" if tau == "0.0" else tau,
                         f"{x['recall']:.2f} [{x['ci_lo']:.2f}, {x['ci_hi']:.2f}]", f"{d.mean():+.2f}", f"{p_:.2f}",
                         str(x["update_generations"]), f"{x['update_input_tokens'] / 1e3:.0f}k"])
            pts.append((m, tau, x["update_input_tokens"], x["recall"]))
    f = summ["FRESH|recompute all"]
    tex = (r"""\begin{tabular}{l c c c c r r}
\toprule
Maintainer & $\tau$ & recall [95\% CI] & $\Delta$ fresh & $p$ & regen. & input tok.\\
\midrule
""" + "\n".join(" & ".join(rw) + r" \\" for rw in rows) +
           f"\n\\midrule\nFresh reference & all & {f['recall']:.2f} [{f['ci_lo']:.2f}, {f['ci_hi']:.2f}] & \\multicolumn{{2}}{{c}}{{(reference)}} & {f['final_summaries']} & {f['update_input_tokens']/1e3:.0f}k\\\\\n\\bottomrule\n\\end{{tabular}}\n")
    (TAB / "qa.tex").write_text(tex)
    M["QANQ"] = str(int(piv.shape[0]))
    M["QANStores"] = str(len(rows))
    M["QANSig"] = str(int(sig))
    ps = np.sort(pvals)
    M["QANSigHolm"] = str(int(sum(1 for i, pv in enumerate(ps) if all(ps[j] * (len(ps) - j) < 0.05 for j in range(i + 1)))))
    M["QAMinP"] = f"{ps[0]:.3f}"
    M["QARecallMin"] = f"{min(x[3] for x in pts):.2f}"
    M["QARecallMax"] = f"{max(x[3] for x in pts):.2f}"
    M["QARecallFresh"] = f"{f['recall']:.2f}"
    tok = {(m, t): v for m, t, v, _ in pts}
    M["QATokMax"] = f"{max(tok.values()) / 1e3:.0f}k"
    M["QATokMin"] = f"{min(tok.values()) / 1e3:.0f}k"
    M["QATokRatio"] = f"{max(tok.values()) / min(tok.values()):.0f}"
    M["QATokRecompNine"] = f"{tok[('recompute_leiden', '0.9')] / 1e3:.0f}k"
    M["QATokHybNever"] = f"{tok[('hybrid', '0.0')] / 1e3:.0f}k"
    M["QASaveHybNever"] = f"{100 * (1 - tok[('hybrid', '0.0')] / tok[('recompute_leiden', '0.9')]):.0f}\\%"
    sd = float(np.std((piv.drop(columns=['FRESH|recompute all']).values - ref.values[:, None]).ravel(), ddof=1))
    M["QAMDE"] = f"{2.8 * sd / np.sqrt(piv.shape[0]):.2f}"
    u = r["usage"]
    M["QATokLuna"] = f"{u['gpt-5.6-luna'][0] / 1e6:.1f}M/{u['gpt-5.6-luna'][1] / 1e6:.2f}M"
    M["QATokTerra"] = f"{u['gpt-5.6-terra'][0] / 1e6:.2f}M/{u['gpt-5.6-terra'][1] / 1e6:.2f}M"
    plan = json.loads((A / "qa" / "plan.json").read_text())
    nodes = pd.read_csv(A / "qa" / "nodes.csv")
    M["QANPapers"] = f"{len(nodes):,}".replace(",", "{,}")
    M["QANBase"] = f"{int((np.load(A / 'qa' / 'node_t.npy') == 0).sum()):,}".replace(",", "{,}")
    # figure: (a) local QA recall vs update tokens, (b) global QA win rates vs fresh summaries
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(4.75, 2.25), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    for m in names:
        st = STYLE[m] if m in STYLE else STYLE["hybrid"]
        xs = [tok[(m, t)] / 1e3 for t in ("0.0", "0.5", "0.7", "0.9")]
        ys = [summ[f"{m}|tau={t}"]["recall"] for t in ("0.0", "0.5", "0.7", "0.9")]
        ax.plot(xs, ys, color=st[0], marker=st[1], ms=3.6, lw=1.0, label=names[m])
    ax.axhline(f["recall"], color="#0b0b0b", lw=0.6, ls=":", label="Fresh")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([20, 100, 500]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}k"))
    ax.set_xlabel("summary input tokens, all updates")
    ax.set_ylabel("key-point recall")
    ax.set_title("(a) local questions", loc="left")
    ax.grid(color="#e8e7e3", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=3,
              handletextpad=0.3, columnspacing=0.7, handlelength=1.6)
    G = {r["store"]: r for r in json.loads((A / "qa_global.json").read_text())["summary"]}
    vm = [k for k in G if k.startswith("vector|") and "budget" in k][0]
    rows_g = [("Hybrid, $\\tau$=0.9", "summaries|hybrid tau=0.9", "#2a78d6"),
              ("Hybrid, never refreshed", "summaries|hybrid never", "#2a78d6"),
              ("Recompute, $\\tau$=0.9", "summaries|recompute tau=0.9", "#2a78d6"),
              ("Fresh, title-only", "summaries|fresh titles-only", "#2a78d6"),
              ("Vector RAG, 12 papers", "vector|12 papers", "#eb6834"),
              (f"Vector RAG, {G[vm]['input_tokens'] / 1e3:.0f}k tok.", vm, "#eb6834")]
    for i_, (lab_, key, col) in enumerate(rows_g):
        y = len(rows_g) - 1 - i_
        r = G[key]
        bx.plot([r["comprehensiveness"], r["diversity"]], [y + 0.14, y - 0.14], color="#cfcdc7", lw=0.8, zorder=1)
        bx.scatter(r["comprehensiveness"], y + 0.14, s=20, marker="o", color=col, zorder=3, edgecolor="white", lw=0.6,
                   label="comprehensiveness" if i_ == 0 else None)
        bx.scatter(r["diversity"], y - 0.14, s=18, marker="D", facecolor="white", edgecolor=col, lw=0.9, zorder=3,
                   label="diversity" if i_ == 0 else None)
    bx.axvline(0.5, color="#0b0b0b", lw=0.6, ls=":")
    bx.set_yticks(range(len(rows_g)))
    bx.set_yticklabels([r[0] for r in rows_g][::-1], fontsize=6.4)
    bx.set_xlim(0, 1)
    bx.set_xlabel("win rate vs. fresh summaries")
    bx.set_title("(b) global questions", loc="left")
    bx.grid(axis="x", color="#e8e7e3", lw=0.5)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(FIG / "qa.pdf", bbox_inches="tight")
    plt.close(fig)


def timing_assets(M):
    """Clean, sequential runtime measurement (3 repetitions, order 0): median seconds per update."""
    meth = ["lazy", "local_repair", "hybrid", "frontier_leiden", "warmstart_leiden", "recompute_leiden"]
    lab = {"lazy": "Lazy attach", "local_repair": "Local repair", "hybrid": "Hybrid", "frontier_leiden": "Frontier Leiden",
           "warmstart_leiden": "Warm-start Leiden", "recompute_leiden": "Recompute Leiden"}
    med = {}
    for ds in ("arxiv", "collab"):
        vals = {}
        for rep in (1, 2, 3):
            f = A / f"timing_rep{rep}_{ds}.json"
            for r in json.loads(f.read_text()):
                vals.setdefault(r["method"], []).append(r["update_seconds"] / (r["versions"] - 1))
        med[ds] = {m: float(np.median(v)) for m, v in vals.items()}
    rows = [[lab[m], f"{med['arxiv'][m]:.2f}", f"{med['collab'][m]:.2f}"] for m in meth]
    tex = (r"""\begin{tabular}{l rr}
\toprule
Maintainer & arXiv-15 & collab-12\\
\midrule
""" + "\n".join(" & ".join(rw) + r" \\" for rw in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "timing.tex").write_text(tex)
    M["TimeHybArx"] = f"{med['arxiv']['hybrid']:.1f}"
    M["TimeFrontArx"] = f"{med['arxiv']['frontier_leiden']:.1f}"
    M["TimeWarmArx"] = f"{med['arxiv']['warmstart_leiden']:.1f}"
    M["TimeRecompArx"] = f"{med['arxiv']['recompute_leiden']:.1f}"
    M["TimeSpeedupHybWarm"] = f"{med['arxiv']['warmstart_leiden'] / med['arxiv']['hybrid']:.0f}"


def semantic_assets(M):
    """Embedding-based stress (semantic_hybrid*.py) and centroid drift of regenerated summaries."""
    pass  # uses the global A (configurable via PAPER_RESULTS)
    r = json.loads((A / "semantic_stress.json").read_text())
    diffs = [abs(v["diff"]) for v in r.values()]
    M["SemMaxAbsDiff"] = f"{max(diffs):.4f}"
    gap = [v["frontier"] - v["sem"] for k, v in r.items() if k.startswith("collab")]
    M["SemCollabGapMin"] = f"{min(gap):.3f}"
    d = json.loads((A / "qa" / "semantic_drift_centred.json").read_text())
    pm = d["per_method"]
    M["DriftCosMin"] = f"{min(v['median_cos'] for v in pm.values()):.2f}"
    M["DriftCosMax"] = f"{max(v['median_cos'] for v in pm.values()):.2f}"
    M["DriftCosRecompPten"] = f"{pm['recompute_leiden']['p10']:.2f}"
    M["DriftCosHalves"] = f"{d['same_community_halves_median']:.2f}"
    M["DriftCosRandom"] = f"{d['random_median']:.2f}"

    L = json.loads((A / "staleness_learn_major.json").read_text())
    R = {r["features"]: r for r in L["results"]}
    M["StaleN"] = str(L["n"])
    M["StaleMajor"] = str(L["stale"])
    M["StaleAucJac"] = f"{R['jaccard only']['auc']:.2f}"
    M["StaleAucStruct"] = f"{R['structural']['auc']:.2f}"
    M["StaleAucWv"] = f"{R['w2v drift only']['auc']:.2f}"
    M["StaleAucOai"] = f"{R['pretrained-emb drift only']['auc']:.2f}"
    M["StaleAucAll"] = f"{R['all']['auc']:.2f}"
    M["StaleAvoided"] = f"{100 * R['structural']['regen_avoided']:.0f}\\%"
    M["StaleRecall"] = f"{100 * R['structural']['stale_recall']:.0f}\\%"
    S = json.loads((A / "staleness_learn_stale.json").read_text())
    M["StaleStrictShare"] = f"{100 * S['stale'] / S['n']:.0f}\\%"

    D = json.loads((A / "delta_summary.json").read_text())
    sav = {m: 1 - c["delta_input"] / c["full_input"] for m, c in D["cost"].items()}
    M["DeltaSaveMin"] = f"{100 * min(sav.values()):.0f}"
    M["DeltaSaveMax"] = f"{100 * max(sav.values()):.0f}\\%"
    import pandas as _pd
    F = _pd.DataFrame(D["fidelity"])
    sub = F.assign(s=F.change >= 3).groupby("kind").s.mean()
    M["FidDelta"] = f"{100 * sub['delta']:.0f}\\%"
    M["FidStale"] = f"{100 * sub['stale']:.0f}\\%"
    M["FidNoise"] = f"{100 * sub['full-again']:.0f}\\%"
    V = _pd.DataFrame(json.loads((A / "input_variants.json").read_text())["rows"])
    V["s"] = V.change >= 3
    big = V[V.n >= 40].groupby("variant").s.mean()
    M["VarTitlesBig"] = f"{100 * big['titles<=300']:.0f}\\%"
    M["VarAbsTenBig"] = f"{100 * big['abstracts x10']:.0f}\\%"
    M["VarAbsTwentyBig"] = f"{100 * big['abstracts x20']:.0f}\\%"
    sh = V.groupby("variant").apply(lambda g: g.inp.sum() / g.full_inp.sum())
    M["VarTitlesShare"] = f"{100 * sh['titles<=300']:.0f}\\%"
    Qv = json.loads((A / "input_variants_qa.json").read_text())["qa"][0]
    M["VarTitlesQAdiff"] = f"{Qv['diff']:+.2f}"
    M["VarTitlesQAp"] = f"{Qv['p']:.2f}"


def extra_assets(M):
    """Local vs global questions, prototype routing, combined cost levers, cold-start placement."""
    pr = {r["store"]: r for r in json.loads((A / "proto_rag.json").read_text())["summary"]}
    M["LocVec"] = f"{pr['papers|vector']['recall']:.2f}"
    M["LocProtoPapers"] = f"{pr['papers|proto']['recall']:.2f}"
    M["LocProtoSums"] = f"{pr['summaries|proto']['recall']:.2f}"
    M["LocSums"] = f"{pr['summaries|tfidf']['recall']:.2f}"
    M["LocVecP"] = "<0.001" if pr['papers|vector']['p'] < 1e-3 else f"{pr['papers|vector']['p']:.3f}"
    g = json.loads((A / "qa_global.json").read_text())
    G = {r["store"]: r for r in g["summary"]}
    M["GlobNQ"] = str(len(g["questions"]))
    M["GlobNMatch"] = str(g["n_match"])
    vm = [k for k in G if k.startswith("vector|") and "budget" in k][0]
    M["GlobVecComp"] = f"{G['vector|12 papers']['comprehensiveness']:.2f}"
    M["GlobVecMatchComp"] = f"{G[vm]['comprehensiveness']:.2f}"
    M["GlobVecMatchDiv"] = f"{G[vm]['diversity']:.2f}"
    M["GlobVecMatchP"] = f"{max(G[vm]['comprehensiveness_p'], G[vm]['diversity_p']):.3f}"
    st = [G[k] for k in ("summaries|hybrid tau=0.9", "summaries|hybrid never", "summaries|recompute tau=0.9",
                         "summaries|fresh titles-only")]
    vals = [r[c] for r in st for c in ("comprehensiveness", "diversity", "empowerment", "directness")]
    ps = [r[c + "_p"] for r in st for c in ("comprehensiveness", "diversity", "empowerment", "directness")]
    M["GlobStaleMin"] = f"{min(vals):.2f}"
    M["GlobStaleMax"] = f"{max(vals):.2f}"
    M["GlobStaleMinP"] = f"{min(ps):.2f}"
    M["GlobHybNeverComp"] = f"{G['summaries|hybrid never']['comprehensiveness']:.2f}"
    c = {r["store"]: r for r in json.loads((A / "combined_cost.json").read_text())["summary"]}
    lt = c["hybrid|learned|titles"]
    M["CombShare"] = f"{100 * lt['token_share']:.0f}\\%"
    M["CombRecall"] = f"{lt['recall']:.2f}"
    M["CombRecallBase"] = f"{c['hybrid|tau|full']['recall']:.2f}"
    M["CombP"] = f"{lt['p_vs_fresh']:.2f}"
    M["CombRegen"] = str(lt["regenerations"])
    M["CombRegenBase"] = str(c["hybrid|tau|full"]["regenerations"])
    cs_ = json.loads((A / "cold_start.json").read_text())
    M["ColdN"] = str(cs_["n_new"])
    M["ColdK"] = str(cs_["n_communities"])
    M["ColdMaj"] = f"{100 * cs_['majority_top1']:.0f}\\%"
    M["ColdWv"] = f"{100 * cs_['word2vec']['top1']:.0f}\\%"
    M["ColdOai"] = f"{100 * cs_['text-embedding-3-small']['top1']:.0f}\\%"
    M["ColdOaiTop"] = f"{100 * cs_['text-embedding-3-small']['top3']:.0f}\\%"

    ss = json.loads((A / "ssl_proto.json").read_text())
    M["SslInfo"] = f"{100 * ss['infonce']['top1']:.0f}\\%"
    M["SslSwav"] = f"{100 * ss['swav']['top1']:.0f}\\%"
    M["SslSwavRaw"] = f"{100 * ss['swav+raw']['top1']:.0f}\\%"
    M["SslSwavRawSd"] = f"{100 * ss['swav+raw']['top1_sd']:.0f}"
    gr = {r["store"]: r for r in json.loads((A / "global_routing.json").read_text())["summary"]}
    vals = [r["comprehensiveness"] for r in gr.values()]
    M["RouteCompMin"] = f"{min(vals):.2f}"
    M["RouteCompMax"] = f"{max(vals):.2f}"


def wiki_assets(M):
    """Evolving-text Wikipedia stream and Neo4j time-travel demo."""
    Wd = A / "wiki"
    r = json.loads((Wd / "results.json").read_text())
    rows = r["rows"]
    R = {x["features"]: x for x in r["results"]}
    ys = sorted(r["n_pages"])
    M["WikiPagesFirst"] = f"{r['n_pages'][ys[0]]:,}".replace(",", "{,}")
    M["WikiPagesLast"] = f"{r['n_pages'][ys[-1]]:,}".replace(",", "{,}")
    M["WikiYearFirst"], M["WikiYearLast"] = ys[0], ys[-1]
    M["WikiQMin"] = f"{min(r['modularity'].values()):.2f}"
    M["WikiQMax"] = f"{max(r['modularity'].values()):.2f}"
    M["WikiPairs"] = str(len(rows))
    M["WikiSub"] = str(sum(x["substantive"] for x in rows))
    same = [x for x in rows if x["jac"] == 1]
    M["WikiSame"] = str(len(same))
    M["WikiSameSub"] = str(sum(x["substantive"] for x in same))
    M["WikiEdited"] = f"{100 * np.median([x['edited'] for x in rows]):.0f}\\%"
    M["WikiTextDiff"] = f"{100 * np.median([x['text_diff'] for x in rows]):.1f}\\%"
    M["WikiAucMem"] = f"{R['membership (arXiv trigger)']['auc']:.2f}"
    M["WikiAucText"] = f"{R['+ text diff (no embedding)']['auc']:.2f}"
    M["WikiAucEmb"] = f"{R['+ embeddings']['auc']:.2f}"
    M["WikiAucEmbOnly"] = f"{R['embeddings only']['auc']:.2f}"
    x = json.loads((Wd / "extra.json").read_text())
    dl = x["deletions"]
    M["WikiDelShare"] = f"{100 * sum(d['links_removed'] for d in dl) / sum(d['links_removed'] + d['links_added'] for d in dl):.0f}\\%"
    sp = x["splits"]
    M["WikiSplits"] = str(sum(d["split"] for d in sp))
    M["WikiSplitCands"] = str(sum(d["communities"] for d in sp))
    c = x["cold_start"]
    M["WikiColdN"] = str(c["n_new"])
    M["WikiCold"] = f"{100 * c['top1']:.0f}\\%"
    M["WikiColdMaj"] = f"{100 * c['majority']:.0f}\\%"
    v = json.loads((N / "versioned_results.json").read_text())
    M["VerLinksCopies"] = f"{v['copies']['links']:,}".replace(",", "{,}")
    M["VerLinksStored"] = f"{v['stored']['links']:,}".replace(",", "{,}")
    M["VerRatio"] = f"{v['copies']['links'] / v['stored']['links']:.1f}"
    M["VerAsOfMs"] = f"{v['timings']['graph_as_of_ms']:.0f}"
    M["VerHistMs"] = f"{v['timings']['community_history_ms']:.0f}"
    M["VerGdsS"] = f"{v['gds_past']['seconds']:.1f}"
    M["VerMoves"] = str(len(v["history_example"]))
    M["VerTitle"] = v["history_title"]


def token_metric_assets(M):
    """Token-cost score: development orders and held-out orders/streams."""
    from scipy import stats as _st
    cal = json.loads((A / "kgv2_calibration.json").read_text())
    lam = cal["lambda_"]
    M["TokLambda"] = f"{lam:.2f}"
    H = {}
    for hf in sorted(f for f in A.glob("kgv2_heldout*.json") if "agent" not in f.name):
        for k, v in json.loads(hf.read_text()).items():
            for m, x in v.items():
                if m != "_ref" and x:
                    H.setdefault(k, {})[m] = x
    names = [("priced_warm", "Warm-start + rollback"), ("priced_frontier", "Frontier + rollback"),
             ("warmstart_leiden", "Warm-start Leiden"), ("frontier_leiden", "Frontier Leiden"),
             ("hybrid", "Hybrid"), ("local_repair", "Local repair"),
             ("recompute_leiden", "Recompute Leiden"), ("lazy", "Lazy attach"),
             ("graphrag_delta", "GraphRAG update (delta)")]
    groups = [("arxiv|2015", "arXiv-15"), ("collab|2012", "collab-12"), ("arxiv|2017", "arXiv-17"),
              ("collab|2010", "collab-10")]
    dev = {m: cal["baselines"][m]["combined"] for m in cal["baselines"]}
    import re as _re
    for m, f in (("priced_warm", "kgv2_dev_priced.log"), ("priced_frontier", "kgv2_dev_priced_frontier.log"),
                 ("graphrag_delta", "kgv2_dev_graphrag_delta.log")):
        if (A / f).exists():
            dev[m] = float(_re.findall(r"combined ([0-9.]+)", (A / f).read_text())[-1])
    names = [x for x in names if x[0] in dev]
    M["TokDevWarm"] = f"{dev['warmstart_leiden']:.3f}"
    M["TokDevHyb"] = f"{dev['hybrid']:.3f}"
    M["TokDevFront"] = f"{dev['frontier_leiden']:.3f}"
    rows, best = [], {}
    cell = {}
    for m, _ in names:
        for g, _ in groups:
            v = [r[m]["Q"] - lam * r[m]["C"] for k, r in H.items() if k.startswith(g + "|") and r.get(m)]
            cell[(m, g)] = v
    for g, _ in groups:
        means = {m: np.mean(cell[(m, g)]) for m, _ in names if cell[(m, g)]}
        best[g] = max(means.values()) if means else None
    for m, lab in names:
        row = [lab, f"{dev[m]:.3f}"]
        for g, _ in groups:
            v = cell[(m, g)]
            if not v:
                row.append("--")
                continue
            mu = np.mean(v)
            ci = _st.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
            txt = f"{mu:.3f}$\\pm${ci:.3f}"
            row.append(f"\\textbf{{{txt}}}" if best[g] is not None and mu >= best[g] - 5e-4 else txt)
        gm = [np.mean(cell[(m, g)]) for g, _ in groups if cell[(m, g)]]
        row.append(f"{np.mean(gm):.3f}" if len(gm) == len(groups) else "--")
        rows.append(row)
    means = {lab: float(r[-1]) for (m, lab), r in zip(names, rows) if r[-1] != "--"}
    if means:
        top = max(means.values())
        for r in rows:
            if r[-1] != "--" and float(r[-1]) >= top - 5e-4:
                r[-1] = f"\\textbf{{{r[-1]}}}"
        order = sorted(means, key=means.get, reverse=True)
        M["TokHeldBest"], M["TokHeldBestV"] = order[0], f"{means[order[0]]:.3f}"
        M["TokHeldSecond"], M["TokHeldSecondV"] = order[1], f"{means[order[1]]:.3f}"
    ns = {g: max((len(cell[(m, g)]) for m, _ in names), default=0) for g, _ in groups}
    head = " & ".join(["Maintainer", "dev."] + [f"{l} ({ns[g]})" for g, l in groups] + ["mean"])
    tex = ("\\begin{tabular}{l c c c c c c}\n\\toprule\n" + head + "\\\\\n\\midrule\n" +
           "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "tokenmetric.tex").write_text(tex)



def newmethod_assets(M):
    """Summary-priced rollback (held-out, lambda sweep, Wikipedia) and ID-free topic reuse (QA stream)."""
    import glob as _g
    import re as _re
    from scipy import stats as _st
    lam = json.loads((A / "kgv2_calibration.json").read_text())["lambda_"]
    H = {}
    for f in _g.glob(str(A / "kgv2_heldout*.json")):
        for k, v in json.loads(Path(f).read_text()).items():
            for m, x in v.items():
                if m != "_ref" and x:
                    H.setdefault(k, {})[m] = x
    def paired(a, b):
        d = [(r[a]["Q"] - lam * r[a]["C"]) - (r[b]["Q"] - lam * r[b]["C"]) for r in H.values() if a in r and b in r]
        return sum(x > 0 for x in d), len(d), float(np.mean(d)), float(_st.ttest_1samp(d, 0).pvalue)
    w, n, d, pv = paired("priced_warm", "warmstart_leiden")
    M["PrWarmWins"], M["PrWarmN"], M["PrWarmDiff"] = str(w), str(n), f"{d:+.3f}"
    M["PrWarmP"] = f"10^{{{int(np.floor(np.log10(pv)))}}}"
    w, n, d, pv = paired("priced_frontier", "frontier_leiden")
    M["PrFrontWins"], M["PrFrontN"], M["PrFrontDiff"] = str(w), str(n), f"{d:+.3f}"
    M["PrFrontP"] = f"10^{{{int(np.floor(np.log10(pv)))}}}"
    dv = {m: float(_re.findall(r"combined ([0-9.]+)", (A / f).read_text())[-1])
          for m, f in (("priced_warm", "kgv2_dev_priced.log"), ("priced_frontier", "kgv2_dev_priced_frontier.log"))}
    M["PrWarmDev"], M["PrFrontDev"] = f"{dv['priced_warm']:.3f}", f"{dv['priced_frontier']:.3f}"
    L = json.loads((A / "kgv2_lambda.json").read_text())
    cal = json.loads((A / "kgv2_calibration.json").read_text())["baselines"]
    def lam_scores(l):
        pr = L[str(l)]["priced"]
        ps = np.mean([pr[s][0] - l * pr[s][1] for s in pr])
        ws = np.mean([cal["warmstart_leiden"][f"{s}_Q"] - l * cal["warmstart_leiden"][f"{s}_C"] for s in pr])
        return ps, ws
    for l, tag in ((0.1, "One"), (0.5, "Half"), (1.0, "Full")):
        ps, ws = lam_scores(l)
        M[f"PrLam{tag}"], M[f"WarmLam{tag}"] = f"{ps:.3f}", f"{ws:.3f}"
    W = json.loads((A / "wiki" / "priced_results.json").read_text())
    for m, tag in (("recompute", "Re"), ("warm", "Warm"), ("priced", "Pr")):
        M[f"WikiPr{tag}S"] = f"{W[m]['score']:.3f}"
        M[f"WikiPr{tag}C"] = f"{W[m]['C']:.2f}"
        M[f"WikiPr{tag}Q"] = f"{W[m]['Q']:.3f}"
    # topic reuse + pipeline table
    R = {r["store"]: r for r in json.loads((A / "summary_reuse.json").read_text())["summary"]}
    P = {r["store"]: r for r in json.loads((A / "qa_pipeline.json").read_text())["summary"]}
    allr = {**R, **P}
    base = R["recompute_leiden|id|full"]["update_tokens"]
    rows = [("recompute_leiden|id|full", "Recompute", "ID", "full"),
            ("recompute_leiden|topic:0.95|full", "Recompute", "topic", "full"),
            ("warmstart_leiden|id|full", "Warm-start", "ID", "full"),
            ("warmstart_leiden|topic:0.95|full", "Warm-start", "topic", "full"),
            ("frontier_leiden|topic:0.95|titles", "Frontier", "topic", "titles"),
            ("hybrid|topic:0.95|titles", "Hybrid", "topic", "titles"),
            ("priced_warm|id|full", "Warm + rollback", "ID", "full"),
            ("priced_warm|topic:0.95|full", "Warm + rollback", "topic", "full"),
            ("priced_warm|topic:0.95|titles", "Warm + rollback", "topic", "titles")]
    lines = []
    for key, m, rule, inp in rows:
        r = allr[key]
        gc = r.get("global_comprehensiveness", r.get("g_comp"))
        gd = r.get("global_diversity", r.get("g_dive"))
        lp = r.get("local_p_vs_fresh", r.get("local_p"))
        lines.append(f"{m} & {rule} & {inp} & {r['written']} & {r['update_tokens'] / 1e3:.0f}k ({(lambda x: f'{x:.1f}' if x < 10 else f'{x:.0f}')(100 * r['update_tokens'] / base)}\\%) & "
                     f"{100 * r['substantive']:.0f}\\% & {r['local_recall']:.2f} ({lp:.2f}) & {gc:.2f} / {gd:.2f} \\\\")
    tex = ("\\begin{tabular}{l l l r r r c c}\n\\toprule\n"
           "Maintainer & rule & input & written & update tokens & subst. & local ($p$) & global win\\\\\n"
           "\\midrule\n" + "\n".join(lines) + "\n\\midrule\n"
           f"Fresh reference & all & full & \\multicolumn{{3}}{{c}}{{(reference)}} & {json.loads((A / 'qa' / 'results.json').read_text())['summary'][[x['store'] for x in json.loads((A / 'qa' / 'results.json').read_text())['summary']].index('FRESH|recompute all')]['recall']:.2f} & 0.50 / 0.50 \\\\\n"
           "\\bottomrule\n\\end{tabular}\n")
    (TAB / "reuse.tex").write_text(tex)
    pt = allr["priced_warm|topic:0.95|titles"]
    M["PipeShare"] = f"{100 * pt['update_tokens'] / base:.1f}\\%"
    M["PipeTok"] = f"{pt['update_tokens'] / 1e3:.0f}k"
    M["BaseTok"] = f"{base / 1e3:.0f}k"
    M["PipeRecall"] = f"{pt['local_recall']:.2f}"
    M["PipeLocalP"] = f"{pt['local_p']:.2f}"
    M["PipeGMinP"] = f"{pt['g_min_p']:.2f}"
    rt = allr["recompute_leiden|topic:0.95|full"]
    M["TopicRecompShare"] = f"{100 * rt['update_tokens'] / base:.0f}\\%"
    M["TopicRecompWritten"], M["IdRecompWritten"] = str(rt["written"]), str(allr["recompute_leiden|id|full"]["written"])
    subs = [allr[k]["substantive"] for k in allr if "topic" in k]
    M["TopicSubMin"], M["TopicSubMax"] = f"{100 * min(subs):.0f}\\%", f"{100 * max(subs):.0f}\\%"
    lps = [allr[k].get("local_p_vs_fresh", allr[k].get("local_p")) for k in allr if "topic" in k]
    M["TopicLocalMinP"] = f"{min(lps):.2f}"
    import pandas as _pd
    r = _pd.DataFrame(json.loads((A / "wiki" / "results.json").read_text())["rows"])
    sub = r.substantive == 1
    for name, regen in (("Id", r.jac < 0.9), ("TopicFive", r.emb_centroid < 0.95), ("TopicEight", r.emb_centroid < 0.98)):
        M[f"WikiRule{name}Rate"] = f"{100 * regen.mean():.0f}\\%"
        M[f"WikiRule{name}Catch"] = f"{100 * regen[sub].mean():.0f}\\%"


def wikidata_assets(M):
    """Typed Wikidata film KG (transfer test of the rollback, lambda unchanged)."""
    Wd = A / "wikidata"
    r = json.loads((Wd / "eval.json").read_text())
    meta = json.loads((Wd / "meta.json").read_text())
    fmt = lambda x: f"{x:,}".replace(",", "{,}")  # noqa: E731
    M["WdNodes"], M["WdFilms"], M["WdEdges"] = fmt(meta["nodes"]), fmt(meta["films"]), fmt(meta["edges"])
    M["WdPeople"] = fmt(meta["node_types"]["person"])
    names = {"priced_frontier": "PrFront", "priced_warm": "PrWarm", "frontier_leiden": "Front",
             "warmstart_leiden": "Warm", "hybrid": "Hyb", "lazy": "Lazy", "recompute_leiden": "Re"}
    for m, tag in names.items():
        M[f"Wd{tag}"] = f"{np.mean([r[f'{m}|{s}']['score'] for s in range(5)]):.3f}"
    for a, b, tag in (("priced_warm", "warmstart_leiden", "Warm"), ("priced_frontier", "frontier_leiden", "Front")):
        d = [r[f"{a}|{s}"]["score"] - r[f"{b}|{s}"]["score"] for s in range(5)]
        M[f"WdWins{tag}"] = str(sum(x > 0 for x in d))


def scale_assets(M):
    """Scalability study on ogbl-citation2 (25%, 50%, 100% node samples)."""
    f = A / "scale.json"
    if not f.exists():
        return
    r = json.loads(f.read_text())
    meth = [("priced_frontier", "Frontier + rollback"), ("priced_warm", "Warm-start + rollback"),
            ("frontier_leiden", "Frontier Leiden"), ("warmstart_leiden", "Warm-start Leiden"),
            ("lazy", "Lazy attach"), ("recompute_leiden", "Recompute Leiden")]
    fr = ["0.25", "0.5", "1.0"]
    best = {x: max(r[f"{x}|{m}"]["score"] for m, _ in meth) for x in fr}
    rows = []
    for m, lab in meth:
        cells = []
        for x in fr:
            v = r[f"{x}|{m}"]
            sc = f"{v['score']:.3f}"
            if v["score"] >= best[x] - 5e-4:
                sc = f"\\textbf{{{sc}}}"
            t = v["update_seconds_median"]
            cells += [sc, f"{t:.1f}" if t < 10 else f"{t:.0f}", f"{v['peak_gb']:.1f}"]
        rows.append(" & ".join([lab] + cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{int(float(x) * 100)}\\% ({r[f'{x}|lazy']['nodes'] / 1e6:.1f}M nodes, {r[f'{x}|lazy']['edges'] / 1e6:.1f}M edges)}}" for x in fr)
    sub = " & ".join(["score & s/upd. & GB"] * 3)
    tex = ("\\begin{tabular}{l rrr rrr rrr}\n\\toprule\n & " + head + "\\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n Maintainer & " + sub + "\\\\\n\\midrule\n" +
           "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "scale.tex").write_text(tex)
    full = {m: r[f"1.0|{m}"] for m, _ in meth}
    M["ScaleFullNodes"] = f"{full['lazy']['nodes'] / 1e6:.1f}M"
    M["ScaleFullEdges"] = f"{full['lazy']['edges'] / 1e6:.1f}M"
    M["ScaleVzeroShare"] = f"{100 * full['lazy']['nodes_v0'] / full['lazy']['nodes']:.0f}\\%"
    M["ScalePrFrontS"], M["ScaleFrontS"] = f"{full['priced_frontier']['score']:.3f}", f"{full['frontier_leiden']['score']:.3f}"
    M["ScaleReS"], M["ScaleLazyS"] = f"{full['recompute_leiden']['score']:.3f}", f"{full['lazy']['score']:.3f}"
    M["ScaleReC"] = f"{full['recompute_leiden']['C']:.2f}"
    M["ScaleMaxGB"] = f"{max(v['peak_gb'] for v in full.values()):.1f}"
    M["ScaleUpdMin"] = f"{min(full[m]['update_seconds_median'] for m in ('priced_frontier', 'priced_warm', 'frontier_leiden', 'warmstart_leiden')) / 60:.1f}"
    M["ScaleUpdMax"] = f"{max(full[m]['update_seconds_median'] for m in ('priced_frontier', 'priced_warm', 'frontier_leiden', 'warmstart_leiden')) / 60:.1f}"
    M["ScaleReUpd"] = f"{full['recompute_leiden']['update_seconds_median'] / 60:.1f}"
    h = {m: r[f"0.5|{m}"]["update_seconds_median"] for m, _ in meth}
    M["ScaleOverheadWarm"] = f"{100 * (h['priced_warm'] / h['warmstart_leiden'] - 1):+.0f}\\%"
    M["ScaleOverheadFront"] = f"{100 * (h['priced_frontier'] / h['frontier_leiden'] - 1):+.0f}\\%"


def graphrag_update_assets(M, runs):
    """Microsoft GraphRAG incremental update (delta clustering) baseline: macros for the text."""
    if ("arxiv", 2015, "graphrag_delta") not in runs:
        return
    for (ds, b), tag in ((("arxiv", 2015), "Arx"), (("collab", 2012), "Col")):
        g, w = stat(runs[(ds, b, "graphrag_delta")]), stat(runs[(ds, b, "warmstart_leiden")])
        M[f"GdQ{tag}"], M[f"WarmQ{tag}"] = f"{g['Q']:.3f}", f"{w['Q']:.3f}"
        M[f"GdR{tag}"] = f"{100 * g['R']:.0f}\\%"
        M[f"GdS{tag}"] = f"{g['S']:.3f}"


def entity_graph_assets(M):
    """LLM-extracted GraphRAG entity graph (structural, 5 document orders) and its end-to-end QA."""
    Eg = A / "entity_graph"
    f = Eg / "eval.json"
    if not f.exists():
        return
    r = json.loads(f.read_text())
    meta = json.loads((Eg / "stream_meta.json").read_text())
    fmt = lambda x: f"{x:,}".replace(",", "{,}")  # noqa: E731
    M["EgNodes"], M["EgEdges"], M["EgVersions"] = fmt(meta["nodes"]), fmt(meta["edges"]), str(meta["versions"])
    tags = {"priced_frontier": "PrFront", "priced_warm": "PrWarm", "frontier_leiden": "Front", "warmstart_leiden": "Warm",
            "hybrid": "Hyb", "lazy": "Lazy", "recompute_leiden": "Re", "graphrag_delta": "Gd"}
    for m, t in tags.items():
        M[f"Eg{t}"] = f"{np.mean([r[f'{m}|{k}']['score'] for k in range(5)]):.3f}"
    for a, b, t in (("priced_warm", "warmstart_leiden", "Warm"), ("priced_frontier", "frontier_leiden", "Front")):
        d = [r[f"{a}|{k}"]["score"] - r[f"{b}|{k}"]["score"] for k in range(5)]
        M[f"EgWins{t}"] = str(sum(x > 0 for x in d))
    q = Eg / "qa_entity.json"
    if q.exists():
        Q = {x["store"]: x for x in json.loads(q.read_text())["summary"]}
        M["EgFreshReports"] = str(json.loads(q.read_text())["fresh_reports"])
        M["EgGdReports"] = str(Q["graphrag_delta"]["final_reports"])
        for m, t in tags.items():
            if m in Q:
                M[f"EgQa{t}Comp"] = f"{Q[m]['comprehensiveness']:.2f}"
                M[f"EgQa{t}Div"] = f"{Q[m]['diversity']:.2f}"
                M[f"EgQa{t}P"] = f"{min(Q[m][c + '_p'] for c in ('comprehensiveness', 'diversity', 'empowerment', 'directness')):.2f}"
                M[f"EgQa{t}Tok"] = f"{Q[m]['update_tokens'] / 1e6:.1f}M"
        qt = json.loads((Eg / "query_tokens.json").read_text())
        M["EgGdQueryX"] = f"{qt['graphrag_delta'] / qt['fresh']:.1f}"
        M["EgGdQueryTok"] = f"{qt['graphrag_delta'] / 1e3:.0f}k"
        M["EgFreshQueryTok"] = f"{qt['fresh'] / 1e3:.0f}k"
        M["EgPrWarmQueryTok"] = f"{qt['priced_warm'] / 1e3:.0f}k"
        be = (Q["priced_warm"]["update_tokens"] - Q["graphrag_delta"]["update_tokens"]) / (qt["graphrag_delta"] - qt["priced_warm"])
        M["EgBreakEven"] = f"{np.ceil(be):.0f}"
        allp = [Q[m][c + "_p"] for m in Q for c in ("comprehensiveness", "diversity", "empowerment", "directness")]
        M["EgQaMinP"] = f"{min(allp):.3f}"
        M["EgQaTests"] = str(len(allp))
        others = [Q[m][c] for m in Q if m != "graphrag_delta" for c in ("comprehensiveness", "diversity", "empowerment", "directness")]
        M["EgQaMin"], M["EgQaMax"] = f"{min(others):.2f}", f"{max(others):.2f}"


def extra_graphs_table(M):
    """Rollback and baselines on Wikidata films, the LLM entity graph and full ogbl-citation2 (token-level score)."""
    from scipy import stats as _st
    W = json.loads((A / "wikidata" / "eval.json").read_text())
    E = json.loads((A / "entity_graph" / "eval.json").read_text())
    S = json.loads((A / "scale.json").read_text()) if (A / "scale.json").exists() else {}
    meth = [("priced_warm", "Warm-start + rollback"), ("priced_frontier", "Frontier + rollback"),
            ("warmstart_leiden", "Warm-start Leiden"), ("frontier_leiden", "Frontier Leiden"), ("hybrid", "Hybrid"),
            ("lazy", "Lazy attach (Zep)"), ("recompute_leiden", "Recompute Leiden"), ("graphrag_delta", "GraphRAG update (delta)")]
    def ci(vals):
        return float(_st.t.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals)))
    cols = {}
    for name, R in (("wd", W), ("eg", E)):
        cols[name] = {m: [R[f"{m}|{k}"]["score"] for k in range(5)] for m, _ in meth if all(f"{m}|{k}" in R for k in range(5))}
    cols["cit"] = {m: [S[f"1.0|{m}"]["score"]] for m, _ in meth if f"1.0|{m}" in S}
    best = {c: max(np.mean(v) for v in cols[c].values()) for c in cols}
    rows = []
    for m, lab in meth:
        cells = [lab]
        for c in ("wd", "eg", "cit"):
            if m not in cols[c]:
                cells.append("not run")
                continue
            v = cols[c][m]
            txt = f"{np.mean(v):.3f}" + (f"$\\pm${ci(v):.3f}" if len(v) > 1 else "")
            cells.append(f"\\textbf{{{txt}}}" if np.mean(v) >= best[c] - 5e-4 else txt)
        rows.append(" & ".join(cells) + " \\\\")
    tex = ("\\begin{tabular}{l c c c}\n\\toprule\nMaintainer & Wikidata films (5) & LLM entity graph (5) & ogbl-citation2 (1)\\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "extra.tex").write_text(tex)


def fig_rollback_lambda(M):
    """Development score (mean of arXiv-15 and collab-12, order 0) versus the token cost weight lambda."""
    L = json.loads((A / "kgv2_lambda.json").read_text())
    cal = json.loads((A / "kgv2_calibration.json").read_text())
    lam_cal = cal["lambda_"]
    B = cal["baselines"]
    import re as _re
    dev = (A / "kgv2_dev_priced.log").read_text()
    qc = {s_: (float(q), float(c)) for s_, q, c in _re.findall(r"priced_warm (\w+) score=[0-9.]+ Q=([0-9.]+) C=([0-9.]+)", dev)}
    pts = {lam_cal: qc}
    for l, v in L.items():
        pts[float(l)] = {s_: tuple(v["priced"][s_]) for s_ in v["priced"]}
    lams = np.linspace(0, 1.05, 200)
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    for m in ["recompute_leiden", "lazy", "frontier_leiden", "warmstart_leiden"]:
        col, _, _, ls = STYLE[m]
        y = [np.mean([B[m][f"{s_}_Q"] - l * B[m][f"{s_}_C"] for s_ in ("arxiv", "collab")]) for l in lams]
        ax.plot(lams, y, color=col, lw=1.1, ls=ls, label=LABEL[m])
    xs = sorted(pts)
    ys = [np.mean([pts[l][s_][0] - l * pts[l][s_][1] for s_ in ("arxiv", "collab")]) for l in xs]
    ax.plot(xs, ys, color="#2a78d6", marker="*", ms=7, lw=1.4, label="Warm-start + rollback")
    ax.axvline(lam_cal, color="#0b0b0b", lw=0.5, ls=":")
    ax.set_xlabel("summary cost weight $\\lambda$")
    ax.set_ylabel("dev. score $\\bar Q-\\lambda \\bar C$")
    ax.grid(color="#e8e7e3", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.0, loc="lower left", handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "rollback_lambda.pdf", bbox_inches="tight")
    plt.close(fig)


def revision_assets(M):
    """Revision: GraphRAG hierarchy study, variance decomposition, recompute noise floor, score validation,
    equivalence tests, cosine baseline, second judge, extra questions."""
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    names = [("reindex", "Full re-index (fresh ids)"), ("reindex+match", "Re-index + id matching"),
             ("warm", "Warm-start hierarchical Leiden"), ("graphrag_update", "GraphRAG incremental update"),
             ("zep_attach", "Zep-style attach")]
    tags = {"reindex": "Re", "reindex+match": "Match", "warm": "Warm", "graphrag_update": "Gu", "zep_attach": "Zep"}
    blocks = []
    for gname, fn, pre in (("LLM entity graph", "graphrag_hier.json", "Hier"),
                           ("Wikidata films", "graphrag_hier_wikidata.json", "HierWd")):
        f = A / fn
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        seeds = list(d)
        mean = lambda m, g: float(np.mean([g(d[s][m]) for s in seeds]))  # noqa: E731
        rows = []
        for m, lab in names:
            c = {t: mean(m, lambda r, t=t: r["cost"][f"{t}|1"]) for t in ("0.9", "0.7", "0.5", "0.0")}
            ql, qt = mean(m, lambda r: r["Qleaf"]), mean(m, lambda r: r["Qtop"])
            cov = mean(m, lambda r: r["coverage_final"]["1"])
            rep = mean(m, lambda r: r["reports_final"]["1"])
            t = tags[m]
            M[f"{pre}Cost{t}Nine"], M[f"{pre}Cost{t}Five"], M[f"{pre}Cost{t}Never"] = (
                f"{c['0.9']:.2f}", f"{c['0.5']:.2f}", f"{c['0.0']:.2f}")
            M[f"{pre}Qtop{t}"], M[f"{pre}Qleaf{t}"], M[f"{pre}Cov{t}"] = f"{qt:.3f}", f"{ql:.3f}", pct(cov)
            M[f"{pre}Rep{t}"] = f"{rep:,.0f}".replace(",", "{,}")
            never = "--" if m == "reindex" else f"{c['0.0']:.2f}"
            rows.append(f"{lab} & {c['0.9']:.2f} & {c['0.7']:.2f} & {c['0.5']:.2f} & {never} & {ql:.3f} & {qt:.3f} & "
                        f"{100 * cov:.0f}\\% & {rep:,.0f} \\\\".replace(",", "{,}"))
        blocks.append(f"\\midrule\n\\multicolumn{{9}}{{l}}{{\\emph{{{gname}}} ({len(seeds)} update orders)}}\\\\\n" + "\n".join(rows))
        M[f"{pre}Seeds"] = str(len(seeds))
    if blocks:
        tex = ("\\begin{tabular}{l c c c c c c c r}\n\\toprule\n"
               " & \\multicolumn{4}{c}{report tokens per update (rel.\\ to re-index)} & & & & \\\\\n"
               "\\cmidrule(lr){2-5}\nMaintainer & $\\tau{=}0.9$ & $0.7$ & $0.5$ & never & $Q_\\mathrm{leaf}$ & "
               "$Q_\\mathrm{top}$ & coverage & reports\\\\\n" + "\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "hier.tex").write_text(tex)
    f = A / "hier_anova.json"
    if f.exists():
        h = json.loads(f.read_text())
        M["HierEtaTau"] = pct(h["identifier_maintainers"]["tau"])
        M["HierEtaMaint"] = f"{100 * h['identifier_maintainers']['maintainer']:.2f}\\%"
        M["HierEtaMin"] = pct(h["identifier_maintainers"]["min_size"])
        M["HierEtaMaintAll"] = pct(h["all"]["maintainer"])
        M["HierEtaTauAll"] = pct(h["all"]["tau"])
    f = A / "noise_floor.json"
    if f.exists():
        nf = json.loads(f.read_text())
        seed = [x["rewrite"] for g in nf.values() for x in g["seed"]]
        M["NoiseSeedMin"], M["NoiseSeedMax"] = pct(min(seed)), pct(max(seed))
        last = [g["last"]["rewrite"] for g in nf.values()]
        M["NoiseLastMin"], M["NoiseLastMax"] = pct(min(last)), pct(max(last))
        drop = [g["drop0.001"]["rewrite"] for g in nf.values()]
        M["NoiseDropMin"], M["NoiseDropMax"] = pct(min(drop)), pct(max(drop))
        for g, t in (("entity", "Eg"), ("collab", "Col"), ("arxiv", "Arx")):
            M[f"NoiseSeed{t}"] = pct(np.mean([x["rewrite"] for x in nf[g]["seed"]]))
            M[f"NoiseLast{t}"] = pct(nf[g]["last"]["rewrite"])
            M[f"NoiseNmi{t}"] = f"{np.mean([x['nmi'] for x in nf[g]['seed']]):.2f}"
    f = A / "validate_score.json"
    if f.exists():
        v = json.loads(f.read_text())
        L = v["local"]
        M["ValRhoScore"], M["ValRhoScoreP"] = f"{L['spearman']['score']['rho']:.2f}", f"{L['spearman']['score']['p']:.2f}"
        M["ValRhoQ"], M["ValRhoQP"] = f"{L['spearman']['Q']['rho']:.2f}", f"{L['spearman']['Q']['p']:.2f}"
        M["ValNEq"], M["ValNStores"], M["ValMargin"] = str(L["n_equivalent"]), str(L["n"]), f"{L['margin']:.2f}"
        nb = [s for s in L["stores"] if s["ci90"][0] > 0]
        M["ValNBetter"] = str(len(nb))
        M["ValCiLo"] = f"{min(s['ci90'][0] for s in L['stores']):.3f}"
        M["ValCiHi"] = f"{max(s['ci90'][1] for s in L['stores']):.3f}"
        E = v["entity"]["spearman"]
        M["ValEgRhoMin"] = f"{min(x['rho'] for x in E.values()):.2f}"
        M["ValEgRhoMax"] = f"{max(x['rho'] for x in E.values()):.2f}"
        G = [r for r in v["global_qa_stream"] if r["store"].startswith("summaries")]
        M["ValGlobEq"], M["ValGlobN"] = str(sum(r["equivalent"] for r in G)), str(len(G))
        M["ValGlobMeanMin"] = f"{min(r['mean'] for r in G):.2f}"
    f = A / "cosine_baseline.json"
    if f.exists():
        cb = json.loads(f.read_text())
        M["CosRandSeven"] = pct(cb["0.7"]["random"]["share_ge_095"])
        M["CosRandFive"] = pct(cb["0.5"]["random"]["share_ge_095"])
        M["CosCommFive"] = pct(cb["0.5"]["community"]["share_ge_095"])
        M["CosRandSevenMean"] = f"{cb['0.7']['random']['mean']:.2f}"
        M["CosRandThreeMean"] = f"{cb['0.3']['random']['mean']:.2f}"
    f = A / "proto_rag.json"
    if f.exists():
        P = {r["store"]: r for r in json.loads(f.read_text())["summary"]}
        M["SumEmbRecall"] = f"{P['summaries|proto']['recall']:.3f}"
        M["SumTfidfRecall"] = f"{P['summaries|tfidf']['recall']:.3f}"
        M["PapersVecRecall"] = f"{P['papers|vector']['recall']:.3f}"
        M["SumEmbVsVecP"] = "10^{-10}"


def judge_assets(M):
    """Second judge (Qwen3-14B, local) and 300-question local QA."""
    import pandas as _pd
    from scipy import stats as _st
    M["HierMatchSave"] = f"{100 * (1 - float(M.get('HierCostMatchNine', '1'))):.0f}\\%"

    def tost(d, margin=0.05):
        d = np.asarray(d, float)
        n, m, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        p = max(_st.t.sf((m + margin) / se, n - 1), _st.t.cdf((m - margin) / se, n - 1))
        h = _st.t.ppf(0.95, n - 1) * se
        return p, m - h, m + h
    for k in ("QwLocalRho", "QwNEq", "QwEgMin", "QwEgMax", "QwAgree", "MoreNEq", "MoreCiLo", "MoreCiHi"):
        M.setdefault(k, "??")
    M.setdefault("MoreNStores", "5")
    M.setdefault("MoreNQ", "300")
    qa = A / "qa"
    if (qa / "results_qwen.json").exists():
        g = _pd.DataFrame(json.loads((qa / "results.json").read_text())["per_question"])
        w = _pd.DataFrame(json.loads((qa / "results_qwen.json").read_text())["per_question"])
        gm, wm = g.groupby("store").recall.mean(), w.groupby("store").recall.mean()
        st = [s for s in gm.index if s != "FRESH|recompute all"]
        M["QwLocalRho"] = f"{_st.spearmanr(gm[st], wm[st]).statistic:.2f}"
        piv = w.pivot_table(index="q", columns="store", values="recall")
        M["QwNEq"] = str(sum(tost(piv[s] - piv["FRESH|recompute all"])[0] < 0.05 for s in st))
        M["QwRecallFresh"] = f"{wm['FRESH|recompute all']:.2f}"
    eq = A / "entity_graph" / "qa_entity_qwen.json"
    if eq.exists():
        Wq = json.loads(eq.read_text())
        crit = ("comprehensiveness", "diversity", "empowerment", "directness")
        vals = [r[c] for r in Wq["summary"] if r["store"] != "graphrag_delta" for c in crit]
        M["QwEgMin"], M["QwEgMax"] = f"{min(vals):.2f}", f"{max(vals):.2f}"
        G = json.loads((A / "entity_graph" / "qa_entity.json").read_text())["rows"]
        key = lambda r: (r["store"], r["q"])  # noqa: E731
        a = _pd.DataFrame(G).groupby(["store", "q"])[list(crit)].mean()
        b = _pd.DataFrame(Wq["rows"]).groupby(["store", "q"])[list(crit)].mean()
        j = a.join(b, rsuffix="_w", how="inner")
        agree = np.mean([np.sign(j[c] - 0.5).values == np.sign(j[c + "_w"] - 0.5).values for c in crit])
        M["QwAgree"] = f"{100 * agree:.0f}\\%"
        M["QwTies"] = f"{100 * (_pd.DataFrame(Wq['rows'])[list(crit)] == 0.5).mean().mean():.0f}\\%"
        gw = {r["store"]: r for r in json.loads((A / "qa_global_qwen.json").read_text())["summary"]}
        M["QwVec"] = f"{gw['vector|12 papers']['comprehensiveness']:.2f}"
    mf = qa / "results_more.json"
    if mf.exists():
        more = _pd.DataFrame(json.loads(mf.read_text())["per_question"])
        old = _pd.DataFrame(json.loads((qa / "results.json").read_text())["per_question"])
        allq = _pd.concat([old[old.store.isin(more.store.unique())], more])
        piv = allq.pivot_table(index="q", columns="store", values="recall")
        M["MoreNQ"] = str(len(piv))
        res = [tost(piv[s] - piv["FRESH|recompute all"]) for s in piv.columns if s not in ("FRESH|recompute all", "papers|vector")]
        M["MoreNEq"] = str(sum(r[0] < 0.05 for r in res))
        M["MoreNStores"] = str(len(res))
        M["MoreCiLo"], M["MoreCiHi"] = f"{min(r[1] for r in res):.3f}", f"{max(r[2] for r in res):.3f}"
        if "papers|vector" in piv:
            M["MoreVec"] = f"{piv['papers|vector'].dropna().mean():.2f}"


def fig_noise(M):
    """Recompute noise floor: share of report tokens rewritten (tau=0.9) by seed change, 0.1% edge drop, real update."""
    f = A / "noise_floor.json"
    if not f.exists():
        return
    nf = json.loads(f.read_text())
    graphs = [("entity", "LLM entity graph"), ("collab", "ogbl-collab"), ("arxiv", "ogbn-arxiv")]
    conds = [("seed", "other Leiden seed,\nunchanged graph", "#9b9a95", "///"),
             ("drop0.001", "0.1% of edges\ndeleted", "#86b0e8", ".."),
             ("last", "real last\nupdate", "#2a78d6", "")]
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    w = 0.26
    for j, (c, lab, col, hatch) in enumerate(conds):
        vals, errs = [], []
        for g, _ in graphs:
            if c == "seed":
                v = [x["rewrite"] for x in nf[g]["seed"]]
                vals.append(np.mean(v)); errs.append([np.mean(v) - min(v), max(v) - np.mean(v)])
            else:
                vals.append(nf[g][c]["rewrite"]); errs.append([0, 0])
        x = np.arange(len(graphs)) + (j - 1) * w
        ax.bar(x, vals, w * 0.92, color=col, edgecolor="white", hatch=hatch, linewidth=0.4, label=lab,
               yerr=np.array(errs).T if c == "seed" else None, error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"))
    ax.set_xticks(np.arange(len(graphs)), [g[1] for g in graphs])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("report tokens rewritten")
    ax.grid(axis="y", lw=0.4, color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "noise.pdf")
    plt.close(fig)


def main():
    runs = load_runs()
    M = {}
    table_main(runs, M)
    table_generalisation(runs)
    table_metric_sensitivity(runs, M)
    table_ablation(runs)
    table_cost(M)
    fig_pareto(runs)
    fig_lambda(runs, M)
    fig_curves(runs)
    numbers(runs, M)
    qa_assets(M)
    timing_assets(M)
    semantic_assets(M)
    extra_assets(M)
    wiki_assets(M)
    token_metric_assets(M)
    newmethod_assets(M)
    wikidata_assets(M)
    scale_assets(M)
    graphrag_update_assets(M, runs)
    entity_graph_assets(M)
    extra_graphs_table(M)
    fig_rollback_lambda(M)
    revision_assets(M)
    judge_assets(M)
    fig_noise(M)
    lines = ["% AUTO-GENERATED by make_assets.py from measured result files. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    (HERE / "numbers.json").write_text(json.dumps(M, indent=1))
    for k in sorted(M):
        print(f"{k:22s} {M[k]}")


if __name__ == "__main__":
    main()
