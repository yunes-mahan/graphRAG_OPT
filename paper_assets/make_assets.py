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
         "frontier_leiden": "Frontier-restr. Leiden", "gds_louvain_seed": "GDS Louvain (seeded)", "lazy": "Lazy attach",
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


B5_STYLE = {  # (colour, marker, size, filled) - families share a hue, markers distinguish (B5, VLDB revision)
    "priced_warm": ("#1baf7a", "P", 34, True), "warmstart_leiden": ("#1baf7a", "D", 22, True),
    "priced_frontier": ("#eb6834", "X", 34, True), "frontier_leiden": ("#eb6834", "s", 24, True),
    "df_leiden_sahu": ("#eb6834", "s", 24, False), "hybrid": ("#2a78d6", "*", 60, True),
    "local_repair": ("#86b0e8", "o", 22, True), "lazy": ("#9b9a95", "v", 26, True),
    "recompute_leiden": ("#52514e", "^", 26, True), "graphrag_delta": ("#52514e", "x", 26, True)}
B5_LABEL = {"priced_warm": "Warm-start + rollback", "warmstart_leiden": "Warm-start Leiden",
            "priced_frontier": "Frontier-restr. + rollback", "frontier_leiden": "Frontier-restr. Leiden (ours)",
            "df_leiden_sahu": "DF Leiden (Sahu's code)", "hybrid": "Hybrid", "local_repair": "Local repair",
            "lazy": "Lazy attach", "recompute_leiden": "Recompute Leiden", "graphrag_delta": "GraphRAG update (delta)"}


def fig_pareto_heldout(M):
    """B5: Q vs C and coverage vs C on the four held-out groups, and the held-out ranking over lambda."""
    f = A / "v2" / "b5_pareto.json"
    if not f.exists():
        return
    P = json.loads(f.read_text())
    groups = list(P["groups"])
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.73), sharex="col")
    offscale = {}                                   # V17f: off-scale GraphRAG values go below the legend, not into panels
    for j, gname in enumerate(groups):
        G = P["groups"][gname]
        for i, (key, par) in enumerate((("Q", "pareto_Q_C"), ("cov", "pareto_cov_C"))):
            ax = axes[i, j]
            for m, st in B5_STYLE.items():
                r = G["maintainers"][m]
                col, mk, ms, fill = st
                ax.errorbar(r["C"][0], r[key][0], xerr=r["C"][1], yerr=r[key][1], fmt="none", ecolor=col, lw=0.6, zorder=2)
                ax.scatter(r["C"][0], r[key][0], s=ms, marker=mk, zorder=3, linewidths=0.9 if fill else 1.0,
                           facecolors=col if fill else "white", edgecolors=(col if not fill else "white") if mk != "x" else None,
                           label=B5_LABEL[m] if (i == 0 and j == 0) else None)
            # GraphRAG's delta update is far below all others in Q/coverage: fit the axis to the rest, note it as text
            ys = [G["maintainers"][m][key][0] for m in B5_STYLE if m != "graphrag_delta"]
            yr = max(ys) - min(ys)
            lo, hi = min(ys) - 0.15 * yr, max(ys) + 0.10 * yr
            gd = G["maintainers"]["graphrag_delta"]
            if gd[key][0] < lo:
                ax.set_ylim(lo, hi)
                v = f"{gd[key][0]:.2f}" if key == "Q" else f"{100 * gd[key][0]:.0f}%"
                offscale.setdefault(gname, {"C": f"{gd['C'][0]:.2f}"})[key] = v
            pts = sorted((G["maintainers"][m]["C"][0], G["maintainers"][m][key][0]) for m in G[par])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#0b0b0b", lw=0.5, ls=":", zorder=1)
            ax.grid(color="#e8e7e3", lw=0.5)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(gname, loc="left")
            if i == 1:
                ax.set_xlabel("report-token cost $\\bar C$")
        axes[0, j].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        axes[1, j].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    axes[0, 0].set_ylabel("modularity $\\bar Q$")
    axes[1, 0].set_ylabel("coverage")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06), handletextpad=0.2,
               columnspacing=0.9, fontsize=6.3)
    if offscale:
        parts = [f"{g}: " + ", ".join(x for x in ((f"Q {o['Q']}" if "Q" in o else ""),
                                                  (f"coverage {o['cov']}" if "cov" in o else ""), f"C {o['C']}") if x)
                 for g, o in offscale.items()]
        half = (len(parts) + 1) // 2
        note = "GraphRAG update (delta), off scale:  " + ";  ".join(parts[:half]) + "\n" + ";  ".join(parts[half:])
        fig.text(0.5, -0.095, note, ha="center", va="top", fontsize=6.3, color="#52514e", linespacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8, h_pad=0.6)
    fig.savefig(FIG / "pareto_heldout.pdf", bbox_inches="tight")
    plt.close(fig)
    # held-out mean score over lambda (rollback variants stay priced at the calibrated lambda)
    lams = sorted(float(x) for x in P["lambda_sweep"])
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m, (col, mk, ms, fill) in B5_STYLE.items():
        y = [P["lambda_sweep"][str(round(l, 2)) if str(round(l, 2)) in P["lambda_sweep"] else str(l)]["scores"][m] for l in lams]
        ax.plot(lams, y, color=col, lw=1.0, ls="-" if fill else "--", marker=mk, markevery=25, ms=3.5,
                mfc=col if fill else "white", mec=col, label=B5_LABEL[m])
    ax.axvline(P["lambda_calibrated"], color="#0b0b0b", lw=0.5, ls=":")
    ax.set_xlabel("report cost weight $\\lambda$")
    ax.set_ylabel("held-out mean $\\bar Q-\\lambda\\bar C$")
    ax.grid(color="#e8e7e3", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_position([0.15, 0.40, 0.82, 0.57])
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=5.6, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               handlelength=1.8, columnspacing=0.8)
    fig.savefig(FIG / "lambda_heldout.pdf", bbox_inches="tight")
    plt.close(fig)
    # macros: Pareto sets and winners over lambda
    name = {m: B5_LABEL[m] for m in B5_LABEL}
    wi = P["winner_intervals"]
    order = sorted(wi, key=lambda m: wi[m][0])
    M["BfWinners"] = "; ".join(f"{name[m]} for " + (f"$\\lambda={wi[m][0]:.2f}$" if wi[m][0] == wi[m][1] else
                                                      f"$\\lambda\\in[{wi[m][0]:.2f},{wi[m][1]:.2f}]$") for m in order)
    for gname, t in zip(groups, ("ArxF", "ColT", "ArxS", "ColTen")):
        M[f"BfPar{t}"] = str(len(P["groups"][gname]["pareto_Q_C"]))
    allpar = set.intersection(*[set(P["groups"][g]["pareto_Q_C"]) for g in groups])
    M["BfParAll"] = ", ".join(name[m] for m in sorted(allpar)) if allpar else "none"
    anypar = set.union(*[set(P["groups"][g]["pareto_Q_C"]) for g in groups])
    never = [m for m in B5_LABEL if m not in anypar]
    M["BfParNever"] = ", ".join(name[m] for m in never[:-1]) + (" and " if len(never) > 1 else "") + name[never[-1]] if never else "none"
    pw = P["winner_intervals"].get("priced_warm")
    M["BfRollFrom"] = f"{pw[0]:.2f}" if pw else "--"
    M["BfLamMax"] = f"{max(float(x) for x in P['lambda_sweep']):.1f}"


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
    names = {"recompute_leiden": "Recompute", "warmstart_leiden": "Warm-start", "frontier_leiden": "Frontier-restr.",
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
    M["QATokRatio"] = f"{max(tok.values()) / min(tok.values()):.1f}"   # A5: one decimal (34.5), was rounded to 35
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
              (f"Vector RAG, {vm.split('|')[1].split()[0]} papers (char.-matched)", vm, "#eb6834")]
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
    lab = {"lazy": "Lazy attach", "local_repair": "Local repair", "hybrid": "Hybrid", "frontier_leiden": "Frontier-restr.\\ Leiden",
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
    # B2b (VLDB revision): Sahu's own DF Leiden (v2/b2b_dfleiden.py) is read from analysis/v2
    for hf in sorted(f for f in list(A.glob("kgv2_heldout*.json")) + list((A / "v2").glob("kgv2_heldout*.json"))
                     if "agent" not in f.name):
        for k, v in json.loads(hf.read_text()).items():
            for m, x in v.items():
                if m != "_ref" and x:
                    H.setdefault(k, {})[m] = x
    names = [("priced_warm", "Warm-start + rollback"), ("priced_frontier", "Frontier-restr.\\ + rollback"),
             ("warmstart_leiden", "Warm-start Leiden"), ("frontier_leiden", "Frontier-restr.\\ Leiden (ours)"),
             ("df_leiden_sahu", "DF Leiden (Sahu's code)"),
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
    if (A / "v2" / "kgv2_dev_df_leiden_sahu.json").exists():
        dev["df_leiden_sahu"] = json.loads((A / "v2" / "kgv2_dev_df_leiden_sahu.json").read_text())["combined"]
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
            txt = f"{mu:.3f}$\\pm${ci:.3f}" if ci >= 5e-4 else f"{mu:.3f}$\\pm${{<}}0.001$"   # T7 (V13): 3 decimals
            row.append(f"\\textbf{{{txt}}}" if best[g] is not None and mu >= best[g] - 5e-4 else txt)
        gm = [np.mean(cell[(m, g)]) for g, _ in groups if cell[(m, g)]]
        row.append(f"{np.mean(gm):.3f}" if len(gm) == len(groups) else "--")
        rows.append(row)
    means = {lab: float(r[-1]) for (m, lab), r in zip(names, rows) if r[-1] != "--"}
    # A2 (VLDB revision): rows sorted by the mean over the four held-out groups (descending); rows without a
    # mean go last.  The caption states the order.
    rows.sort(key=lambda r: -float(r[-1]) if r[-1] != "--" else float("inf"))
    if means:
        top = max(means.values())
        for r in rows:
            if r[-1] != "--" and float(r[-1]) >= top - 5e-4:
                r[-1] = f"\\textbf{{{r[-1]}}}"
        order = sorted(means, key=means.get, reverse=True)
        M["TokHeldBest"], M["TokHeldBestV"] = order[0], f"{means[order[0]]:.3f}"
        M["TokHeldSecond"], M["TokHeldSecondV"] = order[1], f"{means[order[1]]:.3f}"
        M["PrHeldGain"] = f"{means['Warm-start + rollback'] - means['Warm-start Leiden']:+.3f}"   # D4 (V12)
    ns = {g: max((len(cell[(m, g)]) for m, _ in names), default=0) for g, _ in groups}
    head = " & ".join(["Maintainer", "dev."] + [f"{l} ({ns[g]})" for g, l in groups] + ["mean"])
    # B5 (V12): coverage column = mean over the four held-out groups of the mean share of present nodes in a reported
    # community (>= 5 members), from analysis/v2/b5_pareto.json
    bp = A / "v2" / "b5_pareto.json"
    if bp.exists():
        BP = json.loads(bp.read_text())
        lab2m = {lab: m for m, lab in names}
        for r in rows:
            m = lab2m[r[0]]
            cv = [BP["groups"][gg]["maintainers"][m]["cov"][0] for gg in BP["groups"] if m in BP["groups"][gg]["maintainers"]]
            r.append(f"{100 * np.mean(cv):.1f}\\%" if len(cv) == len(BP["groups"]) else "--")
        head += " & cov."
    ncols = len(head.split(" & "))
    tex = ("\\begin{tabular}{l " + "c " * (ncols - 1) + "}\n\\toprule\n" + head + "\\\\\n\\midrule\n" +
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
            ("frontier_leiden|topic:0.95|titles", "Frontier-restr.", "topic", "titles"),
            ("hybrid|topic:0.95|titles", "Hybrid", "topic", "titles"),
            ("priced_warm|id|full", "Warm + rollback", "ID", "full"),
            ("priced_warm|topic:0.95|full", "Warm + rollback", "topic", "full"),
            ("priced_warm|topic:0.95|titles", "Warm + rollback", "topic", "titles")]
    t5f = A / "v12" / "t5_reuse_checks.json"   # T5 (V13): subst. also over pairs that are not a text vs itself
    T5R = json.loads(t5f.read_text()) if t5f.exists() else None
    if T5R:
        r_ = T5R["table8"]["recompute_leiden|id|full"]
        M["SubSelfRecId"], M["SubNRecId"] = str(r_["identical_text"]), str(r_["final_reports"])
    lines = []
    for key, m, rule, inp in rows:
        r = allr[key]
        gc = r.get("global_comprehensiveness", r.get("g_comp"))
        gd = r.get("global_diversity", r.get("g_dive"))
        lp = r.get("local_p_vs_fresh", r.get("local_p"))
        lines.append(f"{m} & {rule} & {inp} & {r['written']} & {r['update_tokens'] / 1e3:.0f}k ({(lambda x: f'{x:.1f}' if x < 10 else f'{x:.0f}')(100 * r['update_tokens'] / base)}\\%) & "
                     f"{100 * r['substantive']:.0f}" + (f" / {100 * T5R['table8'][key]['subst_nonidentical']:.0f}" if T5R else "") +
                     f"\\% & {r['local_recall']:.2f} ({lp:.2f}) & {gc:.2f} / {gd:.2f} \\\\")
    tex = ("\\begin{tabular}{l l l r r r c c}\n\\toprule\n"
           "Maintainer & rule & input & written & update tokens & subst. & local ($p$) & global win\\\\\n"
           "\\midrule\n" + "\n".join(lines) + "\n\\midrule\n" +
           (f"Fresh reference & all & full & {T5R['fresh']['written']} & {T5R['fresh']['tokens'] / 1e3:.0f}k$^{{*}}$ & -- & " if T5R else "Fresh reference & all & full & \\multicolumn{3}{c}{(reference)} & ") + f"{json.loads((A / 'qa' / 'results.json').read_text())['summary'][[x['store'] for x in json.loads((A / 'qa' / 'results.json').read_text())['summary']].index('FRESH|recompute all')]['recall']:.2f} & 0.50 / 0.50 \\\\\n"
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
    meth = [("priced_frontier", "Frontier-restr.\\ + rollback"), ("priced_warm", "Warm-start + rollback"),
            ("frontier_leiden", "Frontier-restr.\\ Leiden"), ("warmstart_leiden", "Warm-start Leiden"),
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
                M[f"EgQa{t}CompP"] = f"{Q[m]['comprehensiveness_p']:.2f}"
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
    meth = [("priced_warm", "Warm-start + rollback"), ("priced_frontier", "Frontier-restr.\\ + rollback"),
            ("warmstart_leiden", "Warm-start Leiden"), ("frontier_leiden", "Frontier-restr.\\ Leiden"), ("local_repair", "Local repair (= Hybrid)"),
            ("df_leiden_sahu", "DF Leiden (Sahu's code)"), ("lazy", "Lazy attach (Zep)"), ("recompute_leiden", "Recompute Leiden"),
            ("graphrag_delta", "GraphRAG update (delta)")]
    xf = A / "v12" / "t4" / "extra_rows.json"           # T4 (V13): DF Leiden and local repair on Wikidata / entity graph
    X = json.loads(xf.read_text()) if xf.exists() else {}
    for sf in sorted((A / "v12" / "t4").glob("scale_*.json")):   # T4 (V13): more citation2 cells
        if sf.exists():
            S = {**S, **{k: v for k, v in json.loads(sf.read_text()).items() if not k.endswith("_ref")}}
    ct = A / "v12" / "b5_coverage_t6.json"
    covt6 = json.loads(ct.read_text()) if ct.exists() else None

    def ci(vals):
        return float(_st.t.ppf(0.975, len(vals) - 1) * np.std(vals, ddof=1) / np.sqrt(len(vals)))
    cols = {}
    for name, R in (("wd", W), ("eg", E)):
        cols[name] = {m: [R[f"{m}|{k}"]["score"] for k in range(5)] for m, _ in meth if all(f"{m}|{k}" in R for k in range(5))}
    xcov = {}
    for name, g in (("wd", "wikidata"), ("eg", "entity")):
        for m, _ in meth:
            if m not in cols[name] and all(f"{g}|{m}|{k}" in X for k in range(5)):
                cols[name][m] = [X[f"{g}|{m}|{k}"]["score"] for k in range(5)]
                xcov[(name, m)] = float(np.mean([X[f"{g}|{m}|{k}"]["coverage"] for k in range(5)]))
    cols["cit"] = {m: [S[f"1.0|{m}"]["score"]] for m, _ in meth if f"1.0|{m}" in S}
    # V15/D2: the "Local repair (= Hybrid)" row is valid only if both give identical results on every graph
    if X:
        for name, R, g in (("wd", W, "wikidata"), ("eg", E, "entity")):
            assert all(abs(R[f"hybrid|{k}"]["score"] - X[f"{g}|local_repair|{k}"]["score"]) < 1e-12 for k in range(5)), name
        if "1.0|hybrid" in S and "1.0|local_repair" in S:
            assert abs(S["1.0|hybrid"]["score"] - S["1.0|local_repair"]["score"]) < 1e-12
    # B6 (V12): Wikipedia stream (one real order with deletions; 5 Leiden seeds), wiki/wiki_priced.py, lambda = 0.31
    WP = json.loads((A / "wiki" / "priced_results.json").read_text())
    wmap = {"recompute_leiden": "recompute", "warmstart_leiden": "warm", "priced_warm": "priced"}
    cols["wiki"] = {m: [r[0] - 0.31 * r[1] for r in WP[w]["runs"]] for m, w in wmap.items()}
    wf = A / "v12" / "t4_wiki.json"                      # T4 (V13): all maintainers via the snapshot adapter
    if wf.exists():
        TW = json.loads(wf.read_text())
        for m in TW:
            v = [q - 0.31 * c for q, c in TW[m]["runs"]]
            if m in cols["wiki"]:
                assert np.allclose(v, cols["wiki"][m]), m        # the adapter reproduces wiki_priced.py exactly
            cols["wiki"][m] = v
        assert np.allclose([q - 0.31 * c for q, c in TW["hybrid"]["runs"]], [q - 0.31 * c for q, c in TW["local_repair"]["runs"]])
    best = {c: max(np.mean(v) for v in cols[c].values()) for c in cols}
    rows = []
    for m, lab in meth:
        cells = [lab]
        for c in ("wd", "eg", "wiki", "cit"):
            if m not in cols[c]:
                cells.append("not run")
                if c in ("wd", "eg") and covt6:
                    cells.append("--")
                continue
            v = cols[c][m]
            txt = f"{np.mean(v):.3f}" + ((f"$\\pm${ci(v):.3f}" if ci(v) >= 5e-4 else "$\\pm{<}0.001$") if len(v) > 1 else "")
            cells.append(f"\\textbf{{{txt}}}" if np.mean(v) >= best[c] - 5e-4 else txt)
            if c in ("wd", "eg") and covt6:                     # B5 (V12): coverage column
                cv_ = xcov.get((c, m), covt6['wikidata' if c == 'wd' else 'entity'].get(m, {}).get('mean'))
                cells.append(f"{100 * cv_:.1f}")
        rows.append(" & ".join(cells) + " \\\\")
    tex = (("\\begin{tabular}{l c c c c c c}\n\\toprule\nMaintainer & Wikidata (5) & cov.\\,\\% & entity gr.\\ (5) & cov.\\,\\% & Wikipedia & citation2\\\\\n")
           if covt6 else "\\begin{tabular}{l c c c}\n\\toprule\nMaintainer & Wikidata films (5) & LLM entity graph (5) & ogbl-citation2 (1)\\\\\n")
    tex = (tex +
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
    tags = {"reindex": "Re", "reindex+match": "Match", "warm": "Warm", "graphrag_update": "Gu", "zep_attach": "Zep",
            "hier_frontier": "Hf"}
    blocks = []
    for gname, fn, pre in (("LLM entity graph", "graphrag_hier.json", "Hier"),
                           ("Wikidata films", "graphrag_hier_wikidata.json", "HierWd")):
        f = A / fn
        # B6 (VLDB revision): 5 document orders (orders 0-2 identical to the stored file, 3-4 new; v2/b6_hier5.py)
        f5 = A / "v2" / ("graphrag_hier_entity_5orders.json" if pre == "Hier" else "graphrag_hier_wikidata_5orders.json")
        if f5.exists():
            f = f5
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        w4 = [A / "v16" / "w4" / f"{'entity' if pre == 'Hier' else 'wikidata'}_s{x}.json" for x in range(5)]
        if all(x.exists() for x in w4):          # V16/W4: same runs (costs identical) + Q on the LCC and content coverage
            d = {}
            for x in w4:
                d.update(json.loads(x.read_text()))
        w1 = [A / "v16" / "w1" / f"{'entity' if pre == 'Hier' else 'wikidata'}_s{x}.json" for x in range(5)]
        names_g = list(names)
        if all(x.exists() for x in w1) and all(x.exists() for x in w4):   # V16/W1: hierarchical frontier maintainer (ours)
            for x in w1:
                for s_, r_ in json.loads(x.read_text()).items():
                    d[s_]["hier_frontier"] = r_["hier_frontier"]
            names_g = names + [("hier_frontier", "Hier.\\ frontier-restr.\\ (ours)")]
        seeds = list(d)
        mean = lambda m, g: float(np.mean([g(d[s][m]) for s in seeds]))  # noqa: E731
        lccq = all("Qleaf_lcc" in d[s][m] for s in seeds for m, _ in names_g)
        rows = []
        for m, lab in names_g:
            c = {t: mean(m, lambda r, t=t: r["cost"][f"{t}|1"]) for t in ("0.9", "0.7", "0.5", "0.0")}
            ql, qt = mean(m, lambda r: r["Qleaf"]), mean(m, lambda r: r["Qtop"])
            if lccq:                              # V16/W4: modularity on the LCC of G_t, unclustered nodes as singletons
                M[f"{pre}QleafClu{tags[m]}"], M[f"{pre}QtopClu{tags[m]}"] = f"{ql:.3f}", f"{qt:.3f}"
                ql, qt = mean(m, lambda r: r["Qleaf_lcc"]), mean(m, lambda r: r["Qtop_lcc"])
            ccn = mean(m, lambda r: r["content_coverage_final"]["0.0"]) if lccq else None
            if ccn is not None:
                M[f"{pre}CcovNever{tags[m]}"] = pct(ccn)
            cov = mean(m, lambda r: r["coverage_final"]["1"])
            rep = mean(m, lambda r: r["reports_final"]["1"])
            t = tags[m]
            M[f"{pre}Cost{t}Nine"], M[f"{pre}Cost{t}Five"], M[f"{pre}Cost{t}Never"] = (
                f"{c['0.9']:.2f}", f"{c['0.5']:.2f}", f"{c['0.0']:.2f}")
            M[f"{pre}Qtop{t}"], M[f"{pre}Qleaf{t}"], M[f"{pre}Cov{t}"] = f"{qt:.3f}", f"{ql:.3f}", pct(cov)
            M[f"{pre}Rep{t}"] = f"{rep:,.0f}".replace(",", "{,}")
            never = f"{c['0.0']:.2f}"            # T1 (V13): re-index rewrites everything also under "never"
            rows.append(f"{lab} & {c['0.9']:.2f} & {c['0.7']:.2f} & {c['0.5']:.2f} & {never} & {ql:.3f} & {qt:.3f} & "
                        f"{100 * cov:.0f}\\%" + (f" & {100 * ccn:.0f}\\%" if ccn is not None else "") + f" & {rep:,.0f} \\\\".replace(",", "{,}"))
        blocks.append(f"\\midrule\n\\multicolumn{{{10 if lccq else 9}}}{{l}}{{\\emph{{{gname}}} ({len(seeds)} update orders)}}\\\\\n" + "\n".join(rows))
        M[f"{pre}Seeds"] = str(len(seeds))
        M[f"{pre}ZepNeverMax"] = f"{max(d[s_]['zep_attach']['cost']['0.0|1'] for s_ in seeds):.2f}"
    lf = A / "v12" / "t1_lcc.json"
    if lf.exists():                         # T1 (V13): GraphRAG clusters only the largest connected component
        L = json.loads(lf.read_text())
        for g, t in (("entity", "Eg"), ("wikidata", "Wd")):
            M[f"Lcc{t}N"], M[f"Lcc{t}All"] = f"{L[g]['lcc']:,}".replace(",", "{,}"), f"{L[g]['entities']:,}".replace(",", "{,}")
    if blocks:
        cc_ = all("& " in b for b in blocks) and "CcovNeverRe" in "".join(M)
        tex = ("\\begin{tabular}{l c c c c c c c " + ("c " if cc_ else "") + "r}\n\\toprule\n"
               " & \\multicolumn{4}{c}{report tokens per update (rel.\\ to re-index)} & & & & " + ("& " if cc_ else "") + "\\\\\n"
               "\\cmidrule(lr){2-5}\nMaintainer & $\\tau{=}0.9$ & $0.7$ & $0.5$ & never & $Q_\\mathrm{leaf}$ & "
               "$Q_\\mathrm{top}$ & cov. & " + ("content cov.\\ (never) & " if cc_ else "") + "reports\\\\\n" + "\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "hier.tex").write_text(tex)
    f = A / "hier_anova.json"
    if f.exists() and "identifier_maintainers" in json.loads(f.read_text()):
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
        from scipy import stats as _st2                # V17c: are the two hybrid stores significantly above fresh?
        for st_, tg in (("hybrid|tau=0.9", "Nine"), ("hybrid|tau=0.0", "Never")):
            dd = (piv[st_] - piv["FRESH|recompute all"]).dropna()
            M[f"MoreHyb{tg}Diff"] = f"{dd.mean():+.3f}"
            pv_ = _st2.wilcoxon(dd, zero_method='zsplit').pvalue
            M[f"MoreHyb{tg}P"] = "{<}0.001" if pv_ < 0.001 else f"={pv_:.2f}"
        if "papers|vector" in piv:
            M["MoreVec"] = f"{piv['papers|vector'].dropna().mean():.2f}"


def fig_noise(M):
    """Instability of GraphRAG's hierarchical clustering: report tokens (tau=0.9) rewritten after re-clustering."""
    f = A / "noise_hier.json"
    if not f.exists():
        return
    nh = json.loads(f.read_text())
    graphs = [("entity", "LLM entity graph"), ("wikidata", "Wikidata films")]
    conds = [("drop0.001", "0.1% of edges deleted", "#86b0e8", ".."),
             ("drop0.01", "1% of edges deleted", "#4a90e2", "\\\\"),
             ("last", "real last update", "#1f5fae", ""),
             ("seed", "other seed, same graph", "#9b9a95", "///")]
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    w = 0.2
    for j, (c, lab, col, hatch) in enumerate(conds):
        vals, errs = [], []
        for g, _ in graphs:
            v = nh[g][c]["0.9"]
            v = v if isinstance(v, list) else [v]
            vals.append(np.mean(v))
            errs.append([np.mean(v) - min(v), max(v) - np.mean(v)])
        x = np.arange(len(graphs)) + (j - 1.5) * w
        ax.bar(x, vals, w * 0.92, color=col, edgecolor="white", hatch=hatch, linewidth=0.4, label=lab,
               yerr=np.array(errs).T, error_kw=dict(lw=0.6, capsize=1.5, ecolor="#52514e"))
    ax.set_xticks(np.arange(len(graphs)), [g[1] for g in graphs])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("report tokens rewritten")
    ax.grid(axis="y", lw=0.4, color="#e4e3df")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "noise.pdf")
    plt.close(fig)

def revision2_assets(M):
    """Second revision: all-maintainer variance decomposition per graph, hierarchical instability, hierarchical QA,
    own-partition fresh control, score-validation range, rollback TOST on the entity graph."""
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    f = A / "hier_anova.json"
    if (A / "v2" / "hier_anova_5orders.json").exists():      # B6: variance decomposition over the 5-order design
        f = A / "v2" / "hier_anova_5orders.json"
    if f.exists():
        h = json.loads(f.read_text())
        if "entity" in h:
            for g, t in (("entity", ""), ("wikidata", "Wd")):
                if g in h:
                    M[f"Hier{t}EtaMaintAll"] = pct(h[g]["all"]["maintainer"])
                    M[f"Hier{t}Split"] = pct(h[g].get("recluster_split", h[g]["all"]["maintainer"]))
                    M[f"Hier{t}EtaTauAll"] = pct(h[g]["all"]["tau"])
                    M[f"Hier{t}EtaTau"] = pct(h[g]["reclustering"]["tau"])
                    M[f"Hier{t}EtaMin"] = pct(h[g]["reclustering"]["min_size"])
                    M[f"Hier{t}EtaMaint"] = "$<$0.1\\%" if h[g]['reclustering']['maintainer'] < 0.001 else f"{100 * h[g]['reclustering']['maintainer']:.1f}\\%"
    M["HierWdMatchSave"] = f"{100 * (1 - float(M.get('HierWdCostMatchNine', '1'))):.0f}\\%"
    f = A / "noise_hier.json"
    if f.exists():
        nh = json.loads(f.read_text())
        for g, t in (("entity", "Eg"), ("wikidata", "Wd")):
            if g not in nh:
                continue
            r = nh[g]
            M[f"NhDrop{t}"] = pct(np.mean(r["drop0.001"]["0.9"]))
            M[f"NhDropFive{t}"] = pct(np.mean(r["drop0.001"]["0.5"]))
            M[f"NhDropOne{t}"] = pct(np.mean(r["drop0.01"]["0.9"]))
            M[f"NhSeed{t}"] = pct(np.mean(r["seed"]["0.9"]))
            M[f"NhLast{t}"] = pct(r["last"]["0.9"])
            M[f"NhLastFive{t}"] = pct(r["last"]["0.5"])
    f = A / "entity_graph" / "hier_qa.json"
    if f.exists():
        H = {r["store"]: r for r in json.loads(f.read_text())["summary"]}
        crit = ("comprehensiveness", "diversity", "empowerment", "directness")
        for s, t in (("warm", "Warm"), ("graphrag_update", "Gu"), ("zep_attach", "Zep")):
            M[f"HqaComp{t}"] = f"{H[s]['comprehensiveness']:.2f}"
            M[f"HqaMin{t}"] = f"{min(H[s][c] for c in crit):.2f}"
            M[f"HqaMax{t}"] = f"{max(H[s][c] for c in crit):.2f}"
            M[f"HqaP{t}"] = f"{min(H[s][c + '_p'] for c in crit):.3f}"
        for s, t in (("fresh", "Fresh"), ("warm", "Warm"), ("graphrag_update", "Gu"), ("zep_attach", "Zep")):
            M[f"HqaRep{t}"] = str(H[s]["reports"])
            M[f"HqaQtok{t}"] = f"{H[s]['query_tokens'] / 1e3:.0f}k"
        M["HqaQtokX"] = f"{H['graphrag_update']['query_tokens'] / H['fresh']['query_tokens']:.1f}"
    f = A / "qa" / "own_fresh.json"
    if f.exists():
        import pandas as _pd
        from scipy import stats as _st
        own = _pd.DataFrame(json.loads(f.read_text())["per_question"])
        old = _pd.DataFrame(json.loads((A / "qa" / "results.json").read_text())["per_question"])
        more = _pd.DataFrame(json.loads((A / "qa" / "results_more.json").read_text())["per_question"])
        allq = _pd.concat([old, more, own]).pivot_table(index="q", columns="store", values="recall")
        allq["OWNFRESH|recompute_leiden"] = allq["FRESH|recompute all"]
        res_own = []
        for m in ("recompute_leiden", "warmstart_leiden", "frontier_leiden", "hybrid"):
            e = (allq[f"{m}|tau=0.9"] - allq[f"OWNFRESH|{m}"]).dropna()
            n_, mu, se = len(e), e.mean(), e.std(ddof=1) / np.sqrt(len(e))
            pt = max(_st.t.sf((mu + 0.05) / se, n_ - 1), _st.t.cdf((mu - 0.05) / se, n_ - 1))
            h = _st.t.ppf(0.95, n_ - 1) * se
            res_own.append((mu, mu - h, mu + h, pt))
        M["OwnNEq"] = str(sum(r[3] < 0.05 for r in res_own))
        M["OwnCiLo"], M["OwnCiHi"] = f"{min(r[1] for r in res_own):.3f}", f"{max(r[2] for r in res_own):.3f}"
        M["OwnDiffMin"], M["OwnDiffMax"] = f"{min(r[0] for r in res_own):+.3f}", f"{max(r[0] for r in res_own):+.3f}"
        for m, t in (("hybrid", "Hyb"), ("warmstart_leiden", "Warm"), ("frontier_leiden", "Front")):
            a = allq[f"OWNFRESH|{m}"]
            M[f"Own{t}"] = f"{a.mean():.3f}"
            d = (a - allq["FRESH|recompute all"]).dropna()
            M[f"Own{t}VsFreshP"] = f"{_st.wilcoxon(d, zero_method='zsplit').pvalue:.3f}"
            M[f"Own{t}VsFresh"] = f"{d.mean():+.3f}"
            mt = f"{m}|tau=0.9"
            e = (allq[mt] - a).dropna()
            M[f"Own{t}Stale"] = f"{allq[mt].mean():.3f}"
            M[f"Own{t}StaleDiff"] = f"{e.mean():+.3f}"
            M[f"Own{t}StaleP"] = f"{_st.wilcoxon(e, zero_method='zsplit').pvalue:.2f}"
        M["OwnFresh"] = f"{allq['FRESH|recompute all'].mean():.3f}"
        plan = json.loads((A / "qa" / "plan.json").read_text())
        nt = np.load(A / "qa" / "node_t.npy")
        pres = np.where(nt >= 0)[0]
        for m, t in (("hybrid", "OwnHybCount"), ("recompute_leiden", "OwnFreshCount")):
            cnt = {}
            for v, c in zip(pres, plan["methods"][m]["labels"][-1]):
                cnt[int(c)] = cnt.get(int(c), 0) + 1
            M[t] = str(sum(1 for x in cnt.values() if x >= 5))
        summ = {r["store"]: r for r in json.loads((A / "qa" / "results.json").read_text())["summary"]}
        M["QATokWarmNine"] = f"{summ['warmstart_leiden|tau=0.9']['update_input_tokens'] / 1e3:.0f}k"
    v = json.loads((A / "validate_score.json").read_text())
    qs = sorted({round(s["Q"], 3) for s in v["local"]["stores"]})
    M["ValQMin"], M["ValQMax"], M["ValQDistinct"] = f"{qs[0]:.3f}", f"{qs[-1]:.3f}", str(len(qs))
    pw = [r for r in v["entity"]["stores"] if r["store"] == "priced_warm"][0]
    M["EgPrWarmTostP"] = f"{pw['comprehensiveness_tost_p']:.3f}"
    hs = A / "hier_sensitivity.json"
    if hs.exists():
        S = json.loads(hs.read_text())
        big = [v for k, v in S.items() if not k.endswith("|10")]
        M["SensSplitMin"], M["SensSplitMax"] = pct(min(v["split"] for v in big)), pct(max(v["split"] for v in big))
        M["SensTauMin"] = pct(min(v["eta_tau"] for v in big))
        M["SensDropMin"], M["SensDropMax"] = pct(min(v["drop"] for v in big)), pct(max(v["drop"] for v in big))
        M["SensLastMin"], M["SensLastMax"] = pct(min(v["last"] for v in big)), pct(max(v["last"] for v in big))
        M["SensMatchMin"] = f"{min(v['cost09']['reindex+match'] for v in big):.2f}"
        M["SensMatchMax"] = f"{max(v['cost09']['reindex+match'] for v in big):.2f}"
    pf = A / "probes.json"
    if pf.exists():
        import pandas as _pd
        from scipy import stats as _st
        P = json.loads(pf.read_text())
        df = _pd.DataFrame(P["rows"])
        for probe, mine, other, t in (("staleness", "stale", "own_fresh", "Stale"), ("coverage", "maintained", "fresh", "Cov")):
            x = df[df.probe == probe].pivot_table(index=["maintainer", "q"], columns="store", values="recall").dropna()
            d = x[mine] - x[other]
            p_ = _st.wilcoxon(d, zero_method="zsplit").pvalue
            M[f"Probe{t}N"] = str(len(d))
            M[f"Probe{t}Mine"], M[f"Probe{t}Ref"] = f"{x[mine].mean():.2f}", f"{x[other].mean():.2f}"
            M[f"Probe{t}Diff"] = f"{d.mean():+.2f}"
            M[f"Probe{t}Loss"] = f"{-d.mean():.2f}"
            M[f"Probe{t}P"] = "<0.001" if p_ < 0.001 else f"={p_:.3f}"
        S = {(r["probe"], r["maintainer"]): r for r in P["summary"]}
        M["ProbeStaleSig"] = str(sum(1 for (pr, m), r in S.items() if pr == "staleness" and r["p"] < 0.05))
        M["ProbeStaleMinDiff"] = f"{max(r['diff'] for (pr, m), r in S.items() if pr == 'staleness'):+.2f}"
        M["ProbeStaleMaxDiff"] = f"{min(r['diff'] for (pr, m), r in S.items() if pr == 'staleness'):+.2f}"
        ps = sorted(r["p"] for (pr, m), r in S.items() if pr == "staleness")
        holm = [min(1, p_ * (len(ps) - i)) for i, p_ in enumerate(ps)]
        holm = [max(holm[:i + 1]) for i in range(len(holm))]
        M["ProbeStaleSigHolm"] = str(sum(h < 0.05 for h in holm))
        M["ProbeCovGuP"] = f"{S[('coverage', 'graphrag_update')]['p']:.3f}"
        M["ProbeCovZepP"] = f"{S[('coverage', 'zep_attach')]['p']:.3f}"
        M["ProbeCovGu"] = f"{S[('coverage', 'graphrag_update')]['diff']:+.3f}"
        M["ProbeCovZep"] = f"{S[('coverage', 'zep_attach')]['diff']:+.3f}"
    dl = A / "deletion_leak.json"
    if dl.exists():
        DL = json.loads(dl.read_text())
        f1 = lambda x: f"{100 * x:.1f}\\%"  # noqa: E731
        for pp, t in (("0.01", "One"), ("0.05", "Five")):
            r = DL[pp]
            M[f"DelGu{t}"], M[f"DelZep{t}"] = f1(r["graphrag_update"]["leak_share"]), f1(r["zep_attach"]["leak_share"])
            M[f"DelRe{t}"] = f1(max(r["reindex+match"]["leak_share"], r["warm"]["leak_share"]))
            M[f"DelLinGu{t}"] = f1(r["graphrag_update"]["lineage_cost"])
            M[f"DelLinZep{t}"] = f1(r["zep_attach"]["lineage_cost"])
            M[f"DelReCost{t}"] = pct(min(r["reindex+match"]["cost"], r["warm"]["cost"]))
            M[f"DelTxtGu{t}"] = pct(r["graphrag_update"]["txt"][0] / max(r["graphrag_update"]["txt"][1], 1))
            M[f"DelTxtZep{t}"] = pct(r["zep_attach"]["txt"][0] / max(r["zep_attach"]["txt"][1], 1))
            M[f"DelRelGu{t}"] = f1(r["graphrag_update"]["rel_leak_share"])
            M[f"DelRelZep{t}"] = f1(r["zep_attach"]["rel_leak_share"])
            M[f"DelEntGu{t}"] = f1(r["graphrag_update"]["ent_leak_share"])
            M[f"DelEntZep{t}"] = f1(r["zep_attach"]["ent_leak_share"])
            M[f"DelDocGu{t}"] = f1(r["graphrag_update"]["doc_lineage_cost"])
            M[f"DelDocZep{t}"] = f1(r["zep_attach"]["doc_lineage_cost"])
    qrf = A / "qa" / "results_qwenreader.json"
    if qrf.exists():
        import pandas as _pd
        from scipy import stats as _st
        w = _pd.DataFrame(json.loads(qrf.read_text())["per_question"]).pivot_table(index="q", columns="store", values="recall")
        res = []
        for c in w.columns:
            if c == "FRESH|recompute all":
                continue
            d = (w[c] - w["FRESH|recompute all"]).dropna()
            n_, mu, se = len(d), d.mean(), d.std(ddof=1) / np.sqrt(len(d))
            pt = max(_st.t.sf((mu + 0.05) / se, n_ - 1), _st.t.cdf((mu - 0.05) / se, n_ - 1))
            h = _st.t.ppf(0.95, n_ - 1) * se
            res.append((mu, mu - h, mu + h, pt))
        M["QrNEq"], M["QrN"] = str(sum(r[3] < 0.05 for r in res)), str(len(res))
        M["QrCiLo"], M["QrCiHi"] = f"{min(r[1] for r in res):.3f}", f"{max(r[2] for r in res):.3f}"
        M["QrMin"], M["QrMax"] = f"{w.mean().min():.2f}", f"{w.mean().max():.2f}"
        M["QrFresh"] = f"{w['FRESH|recompute all'].mean():.2f}"
    ex = A / "tau09_exposure.json"
    if ex.exists():
        X = json.loads(ex.read_text())
        M["ExpReports"], M["ExpGained"], M["ExpGainedFour"] = str(X["reports"]), str(X["gained"]), str(X["gained4"])
    for fn, t in (("probes_qwen_never.json", "QpNever"), ("probes_qwen_tau09.json", "QpNine")):
        f = A / fn
        if f.exists():
            import pandas as _pd
            from scipy import stats as _st
            x = _pd.DataFrame(json.loads(f.read_text())["rows"])
            pv = x.pivot_table(index=["maintainer", "q"], columns="store", values="recall").dropna()
            d = pv["stale"] - pv["own_fresh"]
            p_ = _st.wilcoxon(d, zero_method="zsplit").pvalue
            M[f"{t}N"], M[f"{t}Diff"] = str(len(d)), f"{d.mean():+.2f}"
            M[f"{t}Stale"], M[f"{t}Ref"] = f"{pv['stale'].mean():.2f}", f"{pv['own_fresh'].mean():.2f}"
            M[f"{t}P"] = "<0.001" if p_ < 0.001 else f"={p_:.2f}"
    dc = A / "dilution_check.json"
    if dc.exists():
        D = json.loads(dc.read_text())
        M["DilNineSize"], M["DilNeverSize"] = f"{D['tau09']['median_size']:.0f}", f"{D['never']['median_size']:.0f}"
        M["DilNineJoined"], M["DilNeverJoined"] = pct(D["tau09"]["joined_share"]), pct(D["never"]["joined_share"])
        M["DilNineInput"], M["DilNeverInput"] = pct(D["tau09"]["input_share"]), pct(D["never"]["input_share"])
    rs = A / "refresh_sched.json"
    if rs.exists():
        RS = json.loads(rs.read_text())
        for g, t in (("qa_stream", "Qa"), ("entity_graph", "Eg")):
            for f_, ft in (("0.1", "Ten"), ("0.2", "Twenty")):
                r = RS[g][f_]
                for pol, pt in (("jaccard", "Jac"), ("round_robin", "Rr"), ("oldest", "Old"), ("largest", "Big")):
                    M[f"Rs{t}{ft}{pt}"] = pct(r[pol])
    sv = A / "stability_variants.json"
    if sv.exists():
        SV = json.loads(sv.read_text())
        CRf = A / "consensus_runs.json"
        CR = json.loads(CRf.read_text()) if CRf.exists() else {}
        TS = json.loads((A / "table3_structure.json").read_text()) if (A / "table3_structure.json").exists() else {}

        def stream_cost(g, mode):
            if mode is None:
                fn, key = A / ("graphrag_hier.json" if g == "entity" else "graphrag_hier_wikidata.json"), "reindex+match"
            elif mode == "threshold":
                fn, key = A / f"graphrag_hier_{g}_consensus.json", "consensus"
            else:
                key, D = "consensus", {}
                for f_ in [A / f"graphrag_hier_{g}_consensus_{mode}.json"] + [A / f"graphrag_hier_{g}_consensus_{mode}_s{sd}.json" for sd in (0, 1, 2)]:
                    if f_.exists():
                        D.update(json.loads(f_.read_text()))
                return float(np.mean([D[x][key]["cost"]["0.9|1"] for x in D])) if D else None
            if not fn.exists():
                return None
            D = json.loads(fn.read_text())
            return float(np.mean([D[x][key]["cost"]["0.9|1"] for x in D]))
        def t2_stream(g, key, files):
            """T2 (V13): per-update stream cost (tau 0.9) over orders 0-2, merged from several result files."""
            D = {}
            for f_ in files:
                if f_.exists():
                    D.update({x: v for x, v in json.loads(f_.read_text()).items() if key in v})
            return float(np.mean([D[x][key]["cost"]["0.9|1"] for x in ("0", "1", "2")])) if all(x in D for x in ("0", "1", "2")) else None
        T2D = A / "v12" / "t2"
        blocks = []
        for g, gname in (("entity", "LLM entity graph"), ("wikidata", "Wikidata films")):
            t = {"entity": "", "wikidata": "Wd"}[g]
            fxf = A / f"consensus_fix_{g}.json"
            FX = json.loads(fxf.read_text()) if fxf.exists() else {}
            rows_src = [("GraphRAG Leiden", "yes", FX.get("leiden", SV[g]["leiden"]), stream_cost(g, None))]
            for k in ("10", "50"):
                if f"{k}r" in CR.get(g, {}):
                    r = dict(CR[g][f"{k}r"])
                    r.update(TS.get(g, {}).get(f"threshold{k}", {}))
                    rows_src.append((f"Consensus, {k} runs", "yes", r, stream_cost(g, f"r{k}")))   # T2 (V13)
            if "threshold" in FX:
                rows_src.append(("Consensus, 100 runs", "yes", FX["threshold"], stream_cost(g, "threshold")))
            if "soft" in FX:
                rows_src.append(("Soft consensus, 100", "yes", FX["soft"], stream_cost(g, "soft")))
            if "leafup" in FX:
                rows_src.append(("Cons.\\ leaves + Leiden", "yes", FX["leafup"], stream_cost(g, "leafup")))
            kc = dict(SV[g]["kcore"]); kc.update(TS.get(g, {}).get("kcore", {}))
            wm = dict(SV[g]["warm"]); wm.update(TS.get(g, {}).get("warm", {}))
            # A4: warm start's structure columns are those of the GraphRAG reference it starts from (see caption)
            rows_src += [("k-core, our BFS stand-in", "yes", kc, t2_stream(g, "kcore", [T2D / f"kcore_{g}_-_s{x}.json" for x in range(3)]))]
            # B2 (VLDB revision): the authors' own k-core hierarchy (Hossain & Sariyuce), v2/b2_kcore_authors.py
            KA = json.loads((A / "v2" / "b2_kcore_authors.json").read_text()) if (A / "v2" / "b2_kcore_authors.json").exists() else {}
            for ct in ("RkH", "M2hC", "MRC"):
                if ct in KA.get(g, {}):
                    sc_ = t2_stream(g, "kcore_authors", [A / "v2" / "hier5" / f"kcore_authors_{g}_{ct}.json"] +
                                    [T2D / f"kcore_authors_{g}_{ct}_s{x}.json" for x in range(3)])
                    rows_src.append((f"k-core, authors' {ct}", "yes", KA[g][ct], sc_))
            # A4: warm start's structure columns are those of the GraphRAG reference it starts from (see caption)
            rows_src += [("Warm start$^{\\dagger}$", "no", wm,
                          t2_stream(g, "warm", [A / ("graphrag_hier.json" if g == "entity" else "graphrag_hier_wikidata.json")]))]
            rows = []
            for lab_, rep_, r, sc in rows_src:
                sc_ = "--" if sc is None else f"{sc:.2f}"
                nt_ = r.get("n_top"); lt_ = r.get("largest_top")
                rows.append(f"{lab_} & {rep_} & {100 * np.mean(r['drop']):.0f}\\% & {100 * r['last']:.0f}\\% & {sc_} & "
                            f"{r.get('levels', '--')} & {nt_:,} & {lt_:,} & "
                            f"{r['leaf']:.3f} & {r['top']:.3f} & {100 * r['coverage']:.0f}\\% & {r['reports']:,} \\\\".replace(",", "{,}"))
            for k in ("leiden", "consensus", "kcore", "warm"):
                r = SV[g][k]
                M[f"Sv{t}{k.capitalize()}Drop"] = pct(np.mean(r["drop"]))
                M[f"Sv{t}{k.capitalize()}Last"] = pct(r["last"])
                M[f"Sv{t}{k.capitalize()}Top"] = f"{r['top']:.3f}"
                M[f"Sv{t}{k.capitalize()}Leaf"] = f"{r['leaf']:.3f}"
                M[f"Sv{t}{k.capitalize()}Rep"] = f"{r['reports']:,}".replace(",", "{,}")
                M[f"Sv{t}{k.capitalize()}Cov"] = pct(r["coverage"])
            for name_, tag in (("leiden", "Ld"), ("threshold", "Th"), ("soft", "Soft"), ("leafup", "Lu")):
                if name_ in FX:
                    r = FX[name_]
                    M[f"Fx{t}{tag}Drop"], M[f"Fx{t}{tag}Last"] = pct(np.mean(r["drop"])), pct(r["last"])
                    M[f"Fx{t}{tag}Levels"], M[f"Fx{t}{tag}Top"] = str(r["levels"]), f"{r['n_top']:,}".replace(",", "{,}")
                    M[f"Fx{t}{tag}Largest"] = f"{r['largest_top']:,}".replace(",", "{,}")
                    M[f"Fx{t}{tag}Qtop"] = f"{r['top']:.3f}"
                    sc = stream_cost(g, None if name_ == "leiden" else name_)
                    if sc is not None:
                        M[f"Fx{t}{tag}Stream"] = f"{sc:.2f}"
            blocks.append(f"\\midrule\n\\multicolumn{{12}}{{l}}{{\\emph{{{gname}}}}}\\\\\n" + "\n".join(rows))
        (TAB / "stability.tex").write_text(
            "\\begin{tabular}{l c c c c c r r c c c r}\n\\toprule\n & repr. & 0.1\\% del. & last upd. & stream & levels & top & "
            "largest & $Q_\\mathrm{leaf}$ & $Q_\\mathrm{top}$ & cov. & reports\\\\\n" + "\n".join(blocks) + "\n\\bottomrule\n\\end{tabular}\n")

    cr = A / "consensus_runs.json"
    if cr.exists():
        CR = json.loads(cr.read_text())
        for g, t in (("entity", ""), ("wikidata", "Wd")):
            M[f"Cr{t}Plain"] = f"{CR[g]['plain_seconds']:.1f}"
            for k, kt in (("10", "Ten"), ("50", "Fifty"), ("100", "Hundred")):
                r = CR[g][k]
                M[f"Cr{t}{kt}Drop"] = pct(np.mean(r["drop"]))
                M[f"Cr{t}{kt}Last"] = pct(r["last"])
                M[f"Cr{t}{kt}Sec"] = f"{r['seconds']:.0f}"
                M[f"Cr{t}{kt}Rep"] = f"{r['reports']:,}".replace(",", "{,}")
                M[f"Cr{t}{kt}Cov"] = pct(r["coverage"])
                if k + "r" in CR[g]:
                    rr = CR[g][k + "r"]
                    M[f"Cr{t}{kt}RDrop"], M[f"Cr{t}{kt}RLast"] = pct(np.mean(rr["drop"])), pct(rr["last"])
                    M[f"Cr{t}{kt}RCov"], M[f"Cr{t}{kt}RLeaf"] = pct(rr["coverage"]), f"{rr['leaf']:.3f}"
                    M[f"Cr{t}{kt}RTop"] = f"{rr['top']:.3f}"
    for g, t in (("entity", ""), ("wikidata", "Wd")):
        f = A / f"graphrag_hier_{g}_consensus.json"
        if f.exists():
            D = json.loads(f.read_text())
            mm = lambda fn: float(np.mean([fn(D[k]["consensus"]) for k in D]))  # noqa: E731
            M[f"Cs{t}Cost"] = f"{mm(lambda r: r['cost']['0.9|1']):.2f}"
            M[f"Cs{t}Own"] = f"{mm(lambda r: r['cost_own']['0.9|1']):.2f}"
            M[f"Cs{t}Top"] = f"{mm(lambda r: r['Qtop']):.3f}"
            M[f"Cs{t}Cov"] = pct(mm(lambda r: r['coverage_final']['1']))
            M[f"Cs{t}N"] = str(len(D))
    tm = json.loads((A / "tokmodel_check.json").read_text())
    M["TokModelPearson"] = f"{tm['pearson']:.4f}"
    M["TokModelSpearman"] = f"{tm['spearman']:.2f}"
    M["TokModelRatio"] = f"{tm['ratio_median']:.2f}"
    if "entity" in tm:
        M["TokModelEgSpearman"] = f"{tm['entity']['spearman']:.2f}"
        M["TokModelEgPearson"] = f"{tm['entity']['pearson']:.2f}"


def v15_assets(M):
    """V15: macros for the decisions on the V14 flags (all from result files in analysis/v12/)."""
    V12 = A / "v12"
    pc = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    f = V12 / "t4_wiki.json"
    if f.exists():                          # D1: Wikipedia stream (deletions), lambda = 0.31, seeds 0-4
        TW = json.loads(f.read_text())
        sc = {m: float(np.mean([q - 0.31 * c for q, c in TW[m]["runs"]])) for m in TW}
        for m, t in (("warmstart_leiden", "Warm"), ("frontier_leiden", "Front"), ("priced_warm", "PrWarm"),
                     ("priced_frontier", "PrFront"), ("local_repair", "Lrep"), ("lazy", "Lazy")):
            M[f"WikiS{t}"] = f"{sc[m]:.3f}"
        M["WikiCWarm"] = f"{TW['warmstart_leiden']['C']:.2f}"
        M["WikiRbGain"] = f"{sc['priced_warm'] - sc['warmstart_leiden']:+.3f}"
    f = V12 / "v15_hybrid_qa.json"
    if f.exists():                          # D2: hybrid refreshes on the QA stream
        M["HybQaRefresh"] = str(json.loads(f.read_text())["refreshes"])
    f = V12 / "t5_reuse_checks.json"
    if f.exists():                          # D3: subst. over reports reused for another member set (Table 8 topic rows)
        T5R = json.loads(f.read_text())["table8"]
        v = [r["subst_nonidentical"] for k, r in T5R.items() if "topic" in k]
        M["TopicSubOtherMin"], M["TopicSubOtherMax"] = f"{100 * min(v):.0f}", pc(max(v))
    f = V12 / "v15_fid_stale.json"
    if f.exists():                          # D3: never-refreshed reports without self-comparisons
        FS = json.loads(f.read_text())
        M["FidStaleOther"], M["FidStaleN"], M["FidStaleSelf"] = pc(FS["subst_other"]), str(FS["n"]), str(FS["self_compared"])
        M["FidStaleNOther"] = str(FS["n_other"])
    f = V12 / "v15_delta_lcc.json"
    if f.exists():                          # D4: components of GraphRAG's delta graph (entity graph, small batches)
        DL = json.loads(f.read_text())
        for B, t in (("1", "One"), ("10", "Ten"), ("100", "Hundred")):
            M[f"DlComp{t}"] = f"{DL[B]['components']:.1f}"
            M[f"DlLcc{t}"] = f"{DL[B]['lcc_size']:.0f}"
    f = V12 / "v15_bfs_tokens.json"
    if f.exists():                          # F2: BFS stand-in report tokens relative to GraphRAG's
        BT = json.loads(f.read_text())
        M["BfsTokRatio"], M["BfsTokRatioWd"] = f"{BT['entity']['ratio']:.2f}", f"{BT['wikidata']['ratio']:.2f}"
    cov = []                                # F6: re-clustering coverage in the batch sweep (mean over versions)
    for B in (1, 10, 100):
        f = V12 / "b1" / f"entity_b{B}.json"
        if f.exists():
            R = json.loads(f.read_text())
            cov.append(float(np.mean([R[x]["reindex+match"]["coverage"]["1"] for x in R])))
    if len(cov) == 3:
        M["BsEgReCovMin"], M["BsEgReCovMax"] = f"{100 * min(cov):.0f}", pc(max(cov))


def v16_assets(M):
    """V16: macros from analysis/v16/ (CPU) and analysis/v16_qwen/."""
    V16 = A / "v16"
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    tops = {}
    for g, t in (("entity", ""), ("wikidata", "Wd")):   # P3: final top level with use_lcc on (1) and off (0), orders 0-2
        fs = {l: V16 / f"p3_{g}_lcc{l}.json" for l in (0, 1)}
        if all(f.exists() for f in fs.values()):
            D = {l: json.loads(f.read_text()) for l, f in fs.items()}
            if all(len(D[l]) == 3 for l in D):
                mean = lambda l, k: float(np.mean([D[l][x]["graphrag_update"][k] for x in D[l]]))  # noqa: E731
                M[f"Bs{t}GuNoLccTop"] = f"{mean(0, 'top_final'):,.0f}".replace(",", "{,}")
                M[f"Bs{t}GuLccTop"] = f"{mean(1, 'top_final'):,.0f}".replace(",", "{,}")
                M[f"Bs{t}GuNoLccTopX"] = f"{mean(0, 'top_tok_final') / mean(1, 'top_tok_final'):.1f}"

    f = V16 / "w5_instability.json"
    if f.exists():                          # W5: instability vs Leiden configuration (final graph, tau 0.9)
        W5 = json.loads(f.read_text())
        for g, t in (("entity", "Eg"), ("wikidata", "Wd")):
            r = W5[g]
            M[f"WfPerm{t}"] = pct(float(np.mean(r["perm"]["same_graph"])))
            M[f"WfXl{t}"] = pct(float(np.mean(r["xlevel"]["drop"])))
            M[f"WfConv{t}"] = pct(float(np.mean(r["conv"]["drop"])))
            M[f"WfConvTwice{t}"] = pct(r["conv"]["converged_vs_2x"])
        M["WfIter"] = str(W5["entity"]["conv"]["iterations"])
    for g, t, fo in (("entity", "Eg", "graphrag_hier.json"), ("wikidata", "Wd", "graphrag_hier_wikidata.json")):
        f = V16 / f"w5_stream_{g}_iter50.json"
        if f.exists():                      # W5b over the real stream (re-index + matching, orders 0-2)
            S = json.loads(f.read_text())
            M[f"WfConvStream{t}"] = f"{np.mean([S[x]['reindex+match']['cost']['0.9|1'] for x in S]):.2f}"
    f = V16 / "w3_summary.json"
    if f.exists():                          # W2+W3: GraphRAG-format report tokens, cascade, break-evens (entity graph)
        W3 = json.loads(f.read_text())
        M["XtRho"] = f"{W3['spearman']:.2f}"
        M["XtCells"] = str(W3["cells"])
        M["XtMatchNine"], M["XtMatchNineModel"] = f"{W3['exact']['reindex+match']['0.9']:.2f}", f"{W3['model']['reindex+match']['0.9']:.2f}"
        M["XtCascMax"] = f"{W3['cascade_max_increase']:.2f}"
        M["XtReindexAbs"] = f"{W3['full_reindex_exact_per_update'] / 1e6:.1f}M"
        M["XtGuAbs"] = f"{W3['exact_abs']['graphrag_update']['0.9'] / 1e6:.2f}M"
        M["XtBeGuMatch"] = f"{W3['break_even_exact']['graphrag_update|reindex+match']:.0f}"
        M["XtBeGuZep"] = f"{W3['break_even_exact']['graphrag_update|zep_attach']:.1f}"
    rf = A / "v16" / "w3_rowsum_check.json"
    if rf.exists():
        M["XtRowErr"] = f"{100 * json.loads(rf.read_text())['max_abs_rel_err']:.1f}\\%"
    f = V16 / "w10_kcore10.json"
    if f.exists():                          # W10: authors' k-core code with GraphRAG's size bound 10 (ours)
        K10 = json.loads(f.read_text())
        M["KtenDropEg"], M["KtenDropWd"] = pct(float(np.mean(K10["entity"]["drop"]))), pct(float(np.mean(K10["wikidata"]["drop"])))
    f = V16 / "w1_summary.json"
    if f.exists():                          # W1: hierarchical frontier maintainer (ours): small batches, size bound
        W1 = json.loads(f.read_text())
        bb = list(W1["batch"].values())
        M["BsEgHfMin"], M["BsEgHfMax"] = f"{100 * min(bb):.1f}", f"{100 * max(bb):.0f}\\%"
        M["HfOversizeWd"], M["GrOversizeWd"] = f"{W1['wikidata']['oversize_ours']:,.0f}".replace(",", "{,}"), f"{W1['wikidata']['oversize_graphrag']:,}".replace(",", "{,}")
        M["HfOversizeEg"], M["GrOversizeEg"] = f"{W1['entity']['oversize_ours']:,.0f}", f"{W1['entity']['oversize_graphrag']:,}"
    W8 = {}
    for g in ("entity", "collab12", "arxiv15"):          # W8: synthetic deletion streams (1 % of present nodes per update)
        for m in ("warmstart_leiden", "priced_warm", "frontier_leiden", "priced_frontier"):
            f = V16 / "w8" / f"{g}_{m}.json"
            if f.exists():
                d = json.loads(f.read_text())
                if len(d) == 3:
                    W8[(g, m)] = float(np.mean([d[k]["score"] for k in d]))
    if len(W8) == 12:
        gw = [W8[(g, "priced_warm")] - W8[(g, "warmstart_leiden")] for g in ("entity", "collab12", "arxiv15")]
        gf = [W8[(g, "priced_frontier")] - W8[(g, "frontier_leiden")] for g in ("entity", "collab12", "arxiv15")]
        M["DelRbWarmMin"], M["DelRbWarmMax"] = f"{min(gw):+.3f}", f"{max(gw):+.3f}"
        M["DelRbFrontMin"], M["DelRbFrontMax"] = f"{min(gf):+.3f}", f"{max(gf):+.3f}"
        M["DelRbAllPos"] = "all" if min(gw + gf) > 0 else "not all"
    f = V16 / "w6_summary.json"
    if f.exists():                          # W6: Leiden seeds 1-2 added to Table 5
        W6 = json.loads(f.read_text())
        same = W6["rank_all"] == W6["rank_seed0"] and W6["best_all"] == W6["best_seed0"]
        M["SeedRankText"] = "changes no ranking and no best cell" if same else "changes the ranking (see the technical report)"
    f = V16 / "w9_virtual_view.json"
    if f.exists():                          # W9: materialised vs virtual view (QA stream, measured tokens)
        W9 = json.loads(f.read_text())
        M["VvPerReport"] = f"{W9['per_report_tokens']:,.0f}".replace(",", "{,}")
        M["VvPerQuestion"] = f"{W9['per_question_virtual'] / 1e3:.0f}k"
        M["VvBeMin"], M["VvBeMax"] = f"{W9['break_even_min']:.1f}", f"{W9['break_even_max']:.1f}"


def v17_assets(M):
    """V17: macros from analysis/v17/ (and v16 results re-read)."""
    V17, V16 = A / "v17", A / "v16"
    pct = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
    f = V17 / "m2_eta_hf.json"
    if f.exists():                          # M2: variance shares with the hierarchical frontier maintainer included
        E2 = json.loads(f.read_text())
        for g, t in (("entity", "Ev"), ("wikidata", "EvWd")):
            for k, n in (("split3", "ThreeSplit"), ("split2_hf_as_recluster", "HfRecSplit")):
                v = [E2[g][dsg][k] for dsg in ("all", "no_never", "no_zep")]
                M[f"{t}{n}Min"], M[f"{t}{n}Max"] = pct(min(v)), pct(max(v))
    w1 = [V16 / "w1" / f"{g}_s{x}.json" for g in ("entity", "wikidata") for x in range(5)]
    w4 = [V16 / "w4" / f"{g}_s{x}.json" for g in ("entity", "wikidata") for x in range(5)]
    if all(x.exists() for x in w1 + w4):    # M3a: top level of the hierarchical frontier maintainer vs a re-index
        for g, t in (("entity", ""), ("wikidata", "Wd")):
            H1, H4 = {}, {}
            for x in range(5):
                H1.update(json.loads((V16 / "w1" / f"{g}_s{x}.json").read_text()))
                H4.update(json.loads((V16 / "w4" / f"{g}_s{x}.json").read_text()))
            top = lambda D, m, k: float(np.mean([D[s][m][k] for s in D]))  # noqa: E731
            M[f"Hf{t}Top"] = f"{top(H1, 'hier_frontier', 'top_final'):,.0f}".replace(",", "{,}")
            M[f"Re{t}Top"] = f"{top(H4, 'reindex', 'top_final'):,.0f}".replace(",", "{,}")
            M[f"Hf{t}TopX"] = f"{top(H1, 'hier_frontier', 'top_tok_final') / top(H4, 'reindex', 'top_tok_final'):.1f}"
            for m, tg in (("graphrag_update", "Gu"), ("zep_attach", "Zep")):      # M7: four decimals
                M[f"Hier{t}QleafFour{tg}"] = f"{top(H4, m, 'Qleaf_lcc'):.4f}"
    f = V17 / "m3_deletion_hf.json"
    if f.exists():                          # M3b: deletion leak of the hierarchical frontier maintainer
        D3 = json.loads(f.read_text())
        for p_, t in (("0.01", "One"), ("0.05", "Five")):
            r = D3[p_]["hier_frontier"]
            M[f"DelHf{t}"], M[f"DelHfCost{t}"] = f"{100 * r['leak_share']:.1f}\\%", pct(r["cost"])
            M[f"DelLinHf{t}"], M[f"DelDocHf{t}"] = f"{100 * r['lineage_cost']:.1f}\\%", f"{100 * r['doc_lineage_cost']:.1f}\\%"
    f = V17 / "m8_checks.json"
    if f.exists():                          # M8c: virtual view for global questions
        C8 = json.loads(f.read_text())
        M["VvGlobReports"] = str(C8["global_reports"])
        M["VvPerGlobal"] = f"{C8['per_global_question_virtual'] / 1e3:.0f}k"
        M["VvGlobBeMin"], M["VvGlobBeMax"] = f"{C8['global_break_even_min']:.2f}", f"{C8['global_break_even_max']:.1f}"


def v2_assets(M):
    """VLDB 2027 revision checks (analysis/v2/*.py; see CHANGES.md)."""
    V = A / "v2"
    # B4: size of the balanced design used by hier_anova.py (orders x maintainers x tau x granularity)
    d = json.loads((A / "v2" / "graphrag_hier_entity_5orders.json").read_text())
    ms = [m for m in next(iter(d.values())) if m != "reindex"]
    keys = next(iter(d.values()))[ms[0]]["cost"].keys()
    taus, sizes = {k.split("|")[0] for k in keys}, {k.split("|")[1] for k in keys}
    cells = {(s, m, k) for s in d for m in ms for k in d[s][m]["cost"]}
    assert len(cells) == len(d) * len(ms) * len(taus) * len(sizes)          # balanced, one observation per cell
    M["AnovaDesign"] = (f"{len(ms)}\\text{{ maintainers}}\\times{len(taus)}\\,\\tau\\times{len(sizes)}\\,s\\times"
                        f"{len(d)}\\text{{ orders}}={len(cells)}")   # A2 (V12): full re-index excluded
    M["AnovaNRe"] = str(len(d) * 2 * len(taus) * len(sizes))
    # largest |Q_leaf| or |Q_top| difference between the two re-clustering maintainers in Table 2 (5 orders)
    qd = 0.0
    for gfile in ("graphrag_hier_entity_5orders.json", "graphrag_hier_wikidata_5orders.json"):
        D5 = json.loads((A / "v2" / gfile).read_text())
        mq = lambda k, x: float(np.mean([D5[s_][k][x] for s_ in D5]))  # noqa: E731
        qd = max(qd, *(abs(mq("reindex+match", x) - mq("warm", x)) for x in ("Qleaf", "Qtop")))
    M["HierQMaxDiff"] = f"{qd:.3f}"
    R_ = {r["store"]: r for r in json.loads((A / "summary_reuse.json").read_text())["summary"]}
    M["GlobTopicComp"] = f"{R_['recompute_leiden|topic:0.95|full']['global_comprehensiveness']:.2f}"
    f = V / "kgv2_heldout_df_leiden_sahu.json"
    if f.exists():                          # B2b: Sahu's DF Leiden vs our frontier-restricted Leiden (held-out means)
        lam = json.loads((A / "kgv2_calibration.json").read_text())["lambda_"]
        S = json.loads(f.read_text())
        F = json.loads((A / "kgv2_heldout_frontier_leiden.json").read_text())
        R = json.loads((A / "kgv2_heldout.json").read_text())
        gm = lambda D, m, g, x: float(np.mean([D[k][m][x] for k in S if k.startswith(g + "|")]))  # noqa: E731
        M["SahuQArx"], M["OursQArx"] = f"{gm(S, 'df_leiden_sahu', 'arxiv|2015', 'Q'):.3f}", f"{gm(F, 'frontier_leiden', 'arxiv|2015', 'Q'):.3f}"
        M["RecQArxH"] = f"{gm(R, 'recompute_leiden', 'arxiv|2015', 'Q'):.3f}"
        M["SahuCCol"], M["OursCCol"] = f"{gm(S, 'df_leiden_sahu', 'collab|2012', 'C'):.2f}", f"{gm(F, 'frontier_leiden', 'collab|2012', 'C'):.2f}"
        grp = ("arxiv|2015", "collab|2012", "arxiv|2017", "collab|2010")
        sc = lambda D, m: float(np.mean([gm(D, m, g, "Q") - lam * gm(D, m, g, "C") for g in grp]))  # noqa: E731
        M["SahuMean"], M["OursMean"] = f"{sc(S, 'df_leiden_sahu'):.3f}", f"{sc(F, 'frontier_leiden'):.3f}"
    f = V / "b2_kcore_authors.json"
    if f.exists():                          # B2: authors' k-core hierarchy, ranges over the heuristics RkH/M2hC/MRC
        KA = json.loads(f.read_text())
        pc = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
        for g, t in (("entity", ""), ("wikidata", "Wd")):
            r = {k: v for k, v in KA.get(g, {}).items() if k in ("RkH", "M2hC", "MRC")}
            if not r:
                continue
            dr = [float(np.mean(v["drop"])) for v in r.values()]
            la = [v["last"] for v in r.values()]
            lf = [v["leaf"] for v in r.values()]
            sl = [v["second_level"]["n"] for v in r.values()]
            M[f"KcA{t}DropMin"], M[f"KcA{t}DropMax"] = pc(min(dr)), pc(max(dr))
            M[f"KcA{t}LastMin"], M[f"KcA{t}LastMax"] = pc(min(la)), pc(max(la))
            M[f"KcA{t}LeafMin"], M[f"KcA{t}LeafMax"] = f"{min(lf):.2f}", f"{max(lf):.2f}"
            M[f"KcA{t}SecMin"], M[f"KcA{t}SecMax"] = f"{min(sl):,}".replace(",", "{,}"), f"{max(sl):,}".replace(",", "{,}")
            M[f"KcA{t}NHeur"] = str(len(r))
            sf = A / "v2" / "hier5" / f"kcore_authors_{g}_RkH.json"
            if sf.exists():
                Dk = json.loads(sf.read_text())
                if len(Dk) == 3:
                    M[f"KcA{t}StreamRkH"] = f"{np.mean([Dk[x]['kcore_authors']['cost']['0.9|1'] for x in Dk]):.2f}"
    f = V / "b1_hitleiden_validate.json"
    if f.exists():                          # B1: validation of the unofficial HIT-Leiden implementation
        B1 = json.loads(f.read_text())
        M["HitQUpd"] = f"{B1['modularity_igraph_recomputed']['update']:.3f}".replace("-0.000", "0.000")
        M["HitQStatic"] = f"{B1['modularity_static_leiden_seed0']['update']:.3f}"
        M["HitQInit"] = f"{B1['modularity_igraph_recomputed']['initial']:.3f}"
    # ---- Block C: open-model (Qwen3-14B, local) replication -------------------------------------------------------
    f = V / "c2_qwen_qa.json"
    if f.exists():
        C2 = json.loads(f.read_text())
        mt = [r for r in C2["summary"] if r["store"] != "FRESH|recompute all"]
        fr = next(r for r in C2["summary"] if r["store"] == "FRESH|recompute all")
        M["CtwoNQ"], M["CtwoFresh"] = str(fr["n_q"]), f"{fr['recall']:.2f}"   # distinct questions (duplicates merged by the pivot)
        M["CtwoRecMin"], M["CtwoRecMax"] = f"{min(r['recall'] for r in mt):.2f}", f"{max(r['recall'] for r in mt):.2f}"
        M["CtwoNStores"] = str(len(mt))
        M["CtwoNNonInf"], M["CtwoNEq"] = str(sum(r["noninferior"] for r in mt)), str(sum(r["equivalent"] for r in mt))
        M["CtwoCiLo"], M["CtwoCiHi"] = f"{min(r['ci90'][0] for r in mt):.3f}", f"{max(r['ci90'][1] for r in mt):.3f}"
        M["CtwoMinP"] = f"{min(r['wilcoxon_p'] for r in mt):.2f}"
    f = V / "c1_entity_local_qwen.json"
    if f.exists():
        C1 = json.loads(f.read_text())
        w = next(r for r in C1["summary"] if r["store"] == "warm")
        fr = next(r for r in C1["summary"] if r["store"] == "fresh")
        M["ConeNQ"], M["ConeFresh"], M["ConeWarm"] = str(fr["n_q"]), f"{fr['recall']:.2f}", f"{w['recall']:.2f}"
        M["ConeDiff"], M["ConeP"] = f"{w['diff']:+.3f}", f"{w['wilcoxon_p']:.2f}"
        M["ConeCiLo"], M["ConeCiHi"] = f"{w['ci90'][0]:.3f}", f"{w['ci90'][1]:.3f}"
        M["ConeTostP"] = f"{w['tost_p']:.3f}"
        M["ConeVerdict"] = ("equivalent within $\\pm$0.05" if w["equivalent"] else
                            ("non-inferior (margin 0.05) but not shown equivalent" if w["noninferior"] else
                             "neither non-inferior nor equivalent at the 0.05 margin"))
        M["ConeStale"] = f"{100 * w['stale_retrieved_share']:.1f}\\%"
        M["ConeReports"] = f"{C1['reports_written']:,}".replace(",", "{,}")
        M["ConeComms"] = f"{C1['communities_levels_0_2']['fresh']:,}".replace(",", "{,}")
        M["ConeLevTwo"] = f"{100 * w['level_share']['2']:.0f}\\%"
        M["ConeLevOne"] = f"{100 * w['level_share']['1']:.0f}\\%"
    # ---- B3: cost-model sensitivity ----------------------------------------------------------------------------------
    f1, f2, f3 = V / "b3_flat_sensitivity.json", V / "b3_extra_sensitivity.json", V / "b3_hier_sensitivity.json"
    if f1.exists() and f3.exists():
        F1, F3 = json.loads(f1.read_text()), json.loads(f3.read_text())
        F2 = json.loads(f2.read_text()) if f2.exists() else None
        lab = {"a-50": "$a$=750", "a+50": "$a$=2250", "b-50": "$b$=125", "b+50": "$b$=375", "c-50": "$c$=4000", "c+50": "$c$=12000"}
        rows = []
        for vn in ("a-50", "a+50", "b-50", "b+50", "c-50", "c+50"):
            r = F1["variants"][vn]["recal"]
            cells = [lab[vn], f"{r['lambda_']:.2f}", f"{r['kendall']:.2f}", "yes" if r["top1_same"] else "no"]
            if F2:
                cells += [f"{F2['variants'][vn][g]['kendall']:.2f}" for g in ("wikidata", "entity")]
            cells += ["/".join(f"{F3[g][vn]['split'] * 100:.0f}" for g in ("entity", "wikidata")),
                      "/".join(f"{F3[g][vn]['tau_within'] * 100:.0f}" for g in ("entity", "wikidata"))]
            rows.append(" & ".join(cells) + " \\\\")
        head = ("variant & $\\lambda_v$ & T4 & best" + (" & T6 Wd & T6 EG" if F2 else "") + " & split\\,\\% & $\\tau$\\,\\%")
        ncol = len(rows[0].split(" & "))
        (TAB / "costsens.tex").write_text("\\begin{tabular}{l" + "c" * (ncol - 1) + "}\n\\toprule\n" + head + "\\\\\n\\midrule\n"
                                          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        M["CsFixedKendMin"] = f"{min(F1['variants'][v]['fixed']['kendall'] for v in F1['variants']):.2f}"
        M["CsHierKendMin"] = f"{min(min(F3[g][v]['kendall_vs_base'].values()) for g in F3 for v in F3[g]):.2f}"
        ks = [F1["variants"][v]["recal"]["kendall"] for v in F1["variants"]]
        M["CsKendMin"], M["CsKendMax"] = f"{min(ks):.2f}", f"{max(ks):.2f}"
        M["CsTopSame"] = str(sum(F1["variants"][v]["recal"]["top1_same"] for v in F1["variants"]))
        M["CsNVar"] = str(len(F1["variants"]))
        M["CsLamMin"] = f"{min(F1['variants'][v]['recal']['lambda_'] for v in F1['variants']):.2f}"
        M["CsLamMax"] = f"{max(F1['variants'][v]['recal']['lambda_'] for v in F1['variants']):.2f}"
        M["CsBaseCheck"] = f"{F1['base_check']['max_abs_diff_QC_vs_stored']:.1e}"
        sp = [F3[g][v]["split"] for g in F3 for v in F3[g] if v != "base"]
        tw = {g: [F3[g][v]["tau_within"] for v in F3[g] if v != "base"] for g in F3}
        M["CsSplitMin"], M["CsSplitMax"] = f"{100 * min(sp):.0f}\\%", f"{100 * max(sp):.0f}\\%"
        M["CsTauEgMin"], M["CsTauEgMax"] = f"{100 * min(tw['entity']):.0f}\\%", f"{100 * max(tw['entity']):.0f}\\%"
        M["CsTauWdMin"], M["CsTauWdMax"] = f"{100 * min(tw['wikidata']):.0f}\\%", f"{100 * max(tw['wikidata']):.0f}\\%"
        if F2:
            k2 = [F2["variants"][v][g]["kendall"] for v in F2["variants"] for g in F2["variants"][v]]
            M["CsExtraKendMin"] = f"{min(k2):.2f}"
    V12 = A / "v12"
    f = V / "a5_token_ratio.json"
    if f.exists():                          # A1 (V12): exact token counts behind the ratio
        T5 = json.loads(f.read_text())
        M["QATokMaxExact"] = f"{T5['max_tokens']:,}".replace(",", "{,}")
        M["QATokMinExact"] = f"{T5['min_tokens']:,}".replace(",", "{,}")
    f = V12 / "b1_summary.json"
    if f.exists():                          # B1 (V12): batch-size sweep table + macros
        BS = json.loads(f.read_text())
        pc1 = lambda x: f"{100 * x:.1f}" if x >= 0.001 else f"{100 * x:.2f}"  # noqa: E731
        # T3 (V13): one coverage definition for every row (mean over versions of the share of present nodes in a
        # reported community; hierarchy: every community, flat: >= 5 members), coverage of each update's NEW nodes,
        # and "n/a" where a maintainer does not apply (flat maintainers on GraphRAG's hierarchy)
        T3D = A / "v12" / "t3"
        T3A = json.loads((A / "v12" / "t3_arxiv.json").read_text()) if (A / "v12" / "t3_arxiv.json").exists() else {}
        def t3e(B, k):
            fs_ = [T3D / f"gu_entity_b{B}_s{x}.json" for x in range(3)]
            return float(np.mean([json.loads(f_.read_text())[str(x)]["graphrag_update"][k]["1"] for x, f_ in enumerate(fs_)])) if all(f_.exists() for f_ in fs_) else None
        def t3a(B, k, m="graphrag_delta"):
            v = [T3A.get(f"{B}|{x}|{m}", {}).get(k) for x in range(3)]
            return float(np.mean(v)) if all(x is not None for x in v) else None
        pc0 = lambda x: "--" if x is None else f"{100 * x:.0f}"  # noqa: E731
        rows = []
        for B, r in sorted(BS["entity"].items(), key=lambda kv: int(kv[0])):
            hfb = [A / "v16" / "w1" / f"batch_b{B}_s{x}.json" for x in range(3)]
            hfv = (pc1(float(np.mean([json.loads(f_.read_text())[str(x)]["hier_frontier"]["cost"]["0.9|1"] for x, f_ in enumerate(hfb)])))
                   + "$^{*}$") if all(f_.exists() for f_ in hfb) else "n/a"
            rows.append(" & ".join([f"entity, {B}", f"{r['new_nodes']:.0f}", pc1(r["reindex+match"]["C09"]), pc1(r["warm"]["C09"]),
                                    hfv, "n/a", pc1(r['graphrag_update']['C09']), pc0(t3e(B, "coverage")), pc0(t3e(B, "coverage_new"))]) + " \\\\")
        rows.append("\\midrule")
        for B, r in sorted(BS["arxiv"].items(), key=lambda kv: int(kv[0])):
            if "recompute_leiden" not in r:
                continue
            g = lambda m, k: pc1(r[m][k]) if m in r else "--"  # noqa: E731
            rows.append(" & ".join([f"arXiv, {B}", f"{r['new_nodes']:.0f}", g("recompute_leiden", "C09"), g("warmstart_leiden", "C09"),
                                    g("frontier_leiden", "C09"), g("df_leiden_sahu", "C09"), g("graphrag_delta", "C09"),
                                    pc0(r["graphrag_delta"]["coverage"]), pc0(t3a(B, "coverage_new"))]) + " \\\\")
        (TAB / "batch.tex").write_text(
            "\\begin{tabular}{l r c c c c c c c}\n\\toprule\n & & \\multicolumn{5}{c}{\\% of a re-index per update ($\\tau=0.9$)} & \\multicolumn{2}{c}{GraphRAG cov.\\,\\%}\\\\\n"
            "\\cmidrule(lr){3-7}\\cmidrule(lr){8-9}\nbatch & new & re-cl.\\,+\\,match & warm & frontier & DF & GraphRAG & all & new\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
        if T3A:
            M["BsArxPool"] = f"{T3A['pool_2020']:,}".replace(",", "{,}")
            for B, tg in (("100", "Hundred"), ("1000", "Thousand")):
                v = t3a(B, "new_reported_per_update")
                if v is not None:
                    M[f"BsArxGdNewRep{tg}"] = f"{v:.1f}"
        e1, e100 = BS["entity"]["1"], BS["entity"]["100"]
        M["BsEgOneNew"], M["BsEgHundredNew"] = f"{e1['new_nodes']:.0f}", f"{e100['new_nodes']:.0f}"
        M["BsEgReOne"], M["BsEgReHundred"] = f"{100 * e1['reindex+match']['C09']:.0f}\\%", f"{100 * e100['reindex+match']['C09']:.0f}\\%"
        M["BsEgWarmOne"] = f"{100 * e1['warm']['C09']:.0f}\\%"
        # T3 (V13): coverage = mean over versions (same definition as every row of Table 4)
        R100 = json.loads((V12 / "b1" / "entity_b100.json").read_text())
        cmean = lambda m: float(np.mean([R100[x][m]["coverage"]["1"] for x in R100]))  # noqa: E731
        M["BsEgGuHundred"], M["BsEgGuCovHundred"] = f"{100 * e100['graphrag_update']['C09']:.1f}\\%", f"{100 * cmean('graphrag_update'):.0f}\\%"
        M["BsEgReCovHundred"] = f"{100 * cmean('reindex+match'):.0f}\\%"
        M["BsEgGuCovNewHundred"] = f"{100 * float(np.mean([json.loads((A / 'v12' / 't3' / f'gu_entity_b100_s{x}.json').read_text())[str(x)]['graphrag_update']['coverage_new']['1'] for x in range(3)])):.0f}\\%"
        ar = {int(k): v for k, v in BS["arxiv"].items() if "frontier_leiden" in v}
        if ar:
            lo, hi = min(ar), max(ar)
            M["BsArxFrontMin"] = pc1(min(v["frontier_leiden"]["C09"] for v in ar.values())) + "\\%"
            M["BsArxFrontMax"] = pc1(max(v["frontier_leiden"]["C09"] for v in ar.values())) + "\\%"
            M["BsArxWarmMin"] = pc1(min(v["warmstart_leiden"]["C09"] for v in ar.values())) + "\\%"
            M["BsArxWarmMax"] = pc1(max(v["warmstart_leiden"]["C09"] for v in ar.values())) + "\\%"
            M["BsArxReMin"] = f"{100 * min(v['recompute_leiden']['C09'] for v in ar.values()):.0f}\\%"
            M["BsArxReMax"] = f"{100 * max(v['recompute_leiden']['C09'] for v in ar.values()):.0f}\\%"
            M["BsArxNewMin"], M["BsArxNewMax"] = f"{ar[lo]['new_nodes']:.0f}", f"{ar[hi]['new_nodes']:.0f}"
            dfv = [v["df_leiden_sahu"]["C09"] for v in ar.values() if "df_leiden_sahu" in v]
            if dfv:
                M["BsArxDfMin"], M["BsArxDfMax"] = f"{100 * min(dfv):.0f}\\%", f"{100 * max(dfv):.0f}\\%"
    f = A / "v12_qwen" / "c1_coarse_global.json"
    if f.exists():                          # C1 (V12): coarse levels on global answers (Qwen, pairwise)
        CQ = json.loads(f.read_text())
        st = CQ["stores"]
        M["CqBudget"] = str(CQ["budget"])
        M["CqCovCons"], M["CqCovKcl"] = f"{100 * CQ['coverage']['consensus_top']:.1f}\\%", f"{100 * CQ['coverage']['kcore_l1_top']:.0f}\\%"
        sec = [1 - st[s]["first_position_share"] for s in st]
        M["CqSecondMin"], M["CqSecondMax"] = f"{100 * min(sec):.1f}\\%", f"{100 * max(sec):.1f}\\%"
        M["CqTieMax"] = f"{100 * max(st[s]['tie_rate'] for s in st):.0f}\\%"
        wins = [st[s][c]["win"] for s in st for c in ("comprehensiveness", "diversity", "empowerment", "directness")]
        M["CqWinMin"], M["CqWinMax"] = f"{min(wins):.2f}", f"{max(wins):.2f}"
    f = V12 / "b2_summary.json"
    if f.exists():                          # B2 (V12): rewrite share by level / size; greedy vs Hungarian matching
        B2 = json.loads(f.read_text())
        e, w = B2["entity"]["reindex+match"], B2["wikidata"]["reindex+match"]
        lv_e = sorted(e["by_level"], key=int)
        M["BtTopEg"], M["BtTopWd"] = f"{100 * e['by_level']['0']['token_share']:.0f}\\%", f"{100 * w['by_level']['0']['token_share']:.0f}\\%"
        dl = [e["by_level"][k]["token_share"] for k in ("2", "3", "4")]
        M["BtDeepMin"], M["BtDeepMax"] = f"{100 * min(dl):.0f}\\%", f"{100 * max(dl):.0f}\\%"
        M["BtBigEg"], M["BtSmallEg"] = f"{100 * e['by_size']['>200']['token_share']:.0f}\\%", f"{100 * e['by_size']['<=10']['token_share']:.0f}\\%"
        M["BtMidEg"] = f"{100 * e['by_size']['51-200']['token_share']:.0f}\\%"
        hg = {}
        for gg in ("entity", "wikidata"):
            Gj = json.loads((V12 / "b2" / f"{gg}_greedy.json").read_text())
            Hj = json.loads((V12 / "b2" / f"{gg}_hungarian.json").read_text())
            for mm in ("reindex+match", "warm"):
                for t in ("0.9", "0.7", "0.5", "0.0"):
                    hg[(gg, mm, t)] = (np.mean([Hj[s][mm]["cost"][f"{t}|1"] for s in Hj]) -
                                       np.mean([Gj[s][mm]["cost"][f"{t}|1"] for s in Gj]))
        M["BtHungNine"] = f"{max(abs(v) for k, v in hg.items() if k[2] == '0.9'):.3f}"
        M["BtHungMid"] = f"{max(abs(v) for k, v in hg.items() if k[2] in ('0.7', '0.5')):.3f}"
        M["BtHungNever"] = f"{min(v for k, v in hg.items() if k[2] == '0.0'):.3f}"
    f = V12 / "b7" / "entity_gu_nolcc.json"
    if f.exists():                          # B7 (V12): GraphRAG update with use_lcc off (3 orders)
        for gg, t in (("entity", ""), ("wikidata", "Wd")):
            Nn = json.loads((V12 / "b7" / f"{gg}_gu_nolcc.json").read_text())
            mm = lambda D, fn: float(np.mean([fn(D[s]["graphrag_update"]) for s in D]))  # noqa: E731
            M[f"Bs{t}GuNoLccCov"] = f"{100 * mm(Nn, lambda r: r['coverage_final']['1']):.0f}\\%"
            M[f"Bs{t}GuNoLccCost"] = f"{mm(Nn, lambda r: r['cost']['0.9|1']):.2f}"
            M[f"Bs{t}GuNoLccQtop"] = f"{mm(Nn, lambda r: r['Qtop']):.3f}"
            M[f"Bs{t}GuNoLccRep"] = f"{mm(Nn, lambda r: r['reports_final']['1']):,.0f}".replace(",", "{,}")
    f = A / "v12_qwen" / "c3_never_300.json"
    if f.exists():                          # C3 (V12): 300 questions for the two never-refreshed cheap stores (Qwen)
        C3 = json.loads(f.read_text())
        mt = [r for r in C3["summary"] if r["store"] != "FRESH|recompute all"]
        M["CthNQ"] = str(next(r for r in C3["summary"] if r["store"] == "FRESH|recompute all")["n_q"])
        M["CthNEq"], M["CthN"] = str(sum(r["equivalent"] for r in mt)), str(len(mt))
        M["CthCiLo"], M["CthCiHi"] = f"{min(r['ci90'][0] for r in mt):.3f}", f"{max(r['ci90'][1] for r in mt):.3f}"
        M["CthDiff"] = f"{min(r['diff'] for r in mt):+.3f}"
    f = V12 / "b4_tco.json"
    if f.exists():                          # B4 (V12): total tokens per update
        TC = json.loads(f.read_text())
        M["TcoExtract"] = f"{TC['extraction_tokens_per_update_measured'] / 1e6:.2f}M"
        M["TcoReindex"] = f"{TC['full_reindex_tokens_per_update_modelled'] / 1e6:.1f}M"
        mm = TC["maintainers"]
        M["TcoMatchRep"] = f"{mm['reindex+match']['report_tokens_per_update_modelled'] / 1e6:.1f}M"
        M["TcoGuRep"] = f"{mm['graphrag_update']['report_tokens_per_update_modelled'] / 1e6:.1f}M"
        M["TcoGuQ"], M["TcoMatchQ"] = f"{mm['graphrag_update']['query_tokens_measured'] / 1e3:.0f}k", f"{mm['reindex+match']['query_tokens_measured'] / 1e3:.0f}k"
        be = TC["break_even_queries_per_update"]
        M["TcoBeGuMatch"], M["TcoBeGuZep"] = f"{be['graphrag_update|reindex+match']:.0f}", f"{be['graphrag_update|zep_attach']:.0f}"
    f = V12 / "b3_eta_variants.json"
    if f.exists():                          # B3 (V12): variance shares under restricted designs
        EV = json.loads(f.read_text())
        pc_ = lambda x: f"{100 * x:.0f}\\%"  # noqa: E731
        M["EvNoNeverSplit"], M["EvWdNoNeverSplit"] = pc_(EV["entity"]["no_never"]["split"]), pc_(EV["wikidata"]["no_never"]["split"])
        M["EvNoNeverTau"], M["EvWdNoNeverTau"] = pc_(EV["entity"]["no_never"]["tau_within"]), pc_(EV["wikidata"]["no_never"]["tau_within"])
        M["EvNoNeverSize"] = pc_(EV["entity"]["no_never"]["size_within"])
        M["EvNoZepSplit"], M["EvWdNoZepSplit"] = pc_(EV["entity"]["no_zep"]["split"]), pc_(EV["wikidata"]["no_zep"]["split"])
        for g, p in (("entity", "Ev"), ("wikidata", "EvWd")):      # ranges over the three designs (all / no never / no Zep)
            for k, n in (("split", "Split"), ("tau_within", "Tau")):
                v = [EV[g][d][k] for d in ("all", "no_never", "no_zep")]
                M[f"{p}{n}Min"], M[f"{p}{n}Max"] = pc_(min(v)), pc_(max(v))
        M["EvInitMax"] = f"{100 * max(EV[g][d]['initialisation_within'] for g in EV for d in EV[g]):.2f}\\%"
    f = V12 / "d2_tost_035.json"
    if f.exists():                          # D2 (V12): robustness of the non-inferiority tests at a margin of 0.035
        TM = json.loads(f.read_text())
        r = {k: TM[k]["0.035"] for k in TM}
        M["TmMargin"] = "0.035"
        for k, n in (("more300", "More"), ("own", "Own"), ("c2", "Ctwo"), ("c3", "Cth")):
            M[f"Tm{n}Lo"] = f"{r[k]['min_lower']:.3f}"
            M[f"Tm{n}NNi"], M[f"Tm{n}N"] = str(r[k]["n_noninferior"]), str(r[k]["n_stores"])
        M["TmCtwoFailLo"] = f"{min(s['lo'] for s in r['c2']['stores'].values() if not s['noninferior']):.3f}"
    f = V12 / "d2_margin_provenance.json"
    if f.exists():                          # D2 (V12)
        DP = json.loads(f.read_text())
        M["MarginShare"] = f"{100 * DP['margin_share_of_fresh']:.0f}\\%"
        M["TmShare"] = f"{100 * 0.035 / DP['fresh_recall']:.0f}\\%"
    f = V12 / "b5_coverage_t6.json"
    if f.exists():                          # B5 (V12)
        CT = json.loads(f.read_text())
        M["CovLazyEg"], M["CovWarmEg"] = f"{100 * CT['entity']['lazy']['mean']:.1f}\\%", f"{100 * CT['entity']['warmstart_leiden']['mean']:.1f}\\%"
        M["CovLazyEgFine"], M["CovGdEgFine"] = f"{100 * CT['entity']['lazy']['mean']:.2f}\\%", f"{100 * CT['entity']['graphrag_delta']['mean']:.2f}\\%"   # T4 (V13)
    f = V12 / "a5_rollback_factor.json"
    if f.exists():                          # A5 (V12): exact (T+1)/T per graph vs the fixed 16/15
        RF = json.loads(f.read_text())
        M["RbFacMin"], M["RbFacMax"] = f"{min(r['exact'] for r in RF.values()):.3f}", f"{max(r['exact'] for r in RF.values()):.3f}"
        M["RbDevMin"] = f"{100 * min(r['deviation_of_fixed'] for r in RF.values()):.1f}\\%"
        M["RbDevMax"] = f"{100 * max(r['deviation_of_fixed'] for r in RF.values()):+.1f}\\%"
    f = V / "b6_hier5.json"
    if f.exists():
        B6 = json.loads(f.read_text())
        mh = {k: max(B6[g]["max_halfwidth"][k] for g in B6) for k in ("cost", "Q", "coverage")}
        M["HierCiCost"], M["HierCiQ"] = f"{mh['cost']:.2f}", f"{mh['Q']:.3f}"
        M["HierCiCov"] = f"{100 * mh['coverage']:.1f}\\%"
        M["HierNOrders"] = str(B6["entity"]["n_orders"])
    f = V / "a1_hybrid_refresh.json"
    if f.exists():
        H = json.loads(f.read_text())
        held = {k: v for k, v in H.items() if not (k.endswith("|0") and k.split("|")[1] in ("2015", "2012"))}
        arx = [v["refreshes"] for k, v in held.items() if k.startswith("arxiv|2015|")]
        oth = [v["refreshes"] for k, v in held.items() if not k.startswith("arxiv|2015|")]
        assert all(v["rescored"] == {x: v["stored"][x] for x in ("Q", "C")} for v in held.values())
        M["HybRefArxMin"], M["HybRefArxMax"], M["HybRefOtherMax"] = str(min(arx)), str(max(arx)), str(max(oth))
    f = V / "a6_global_length.json"
    if f.exists():
        G = json.loads(f.read_text())
        t = next(r for r in G["holm"] if r["store"] == "recompute_leiden|topic:0.95|full" and r["criterion"] == "comprehensiveness")
        M["GlobTopicP"], M["GlobTopicPHolm"] = f"{t['p']:.3f}", f"{t['p_holm']:.2f}"
        M["GlobNTests"], M["GlobNSigHolm"] = str(G["n_tests"]), str(G["n_sig_holm"])
        M["GlobNAboveHalf"], M["GlobMeanCompAll"] = str(G["n_stores_comp_above_half"]), f"{G['mean_comp_win_all_stores']:.2f}"
        M["GlobNStoresReuse"] = str(len(G["stores"]))
        s = G["stores"]["recompute_leiden|topic:0.95|full"]
        M["GlobTopicWords"], M["GlobFreshWords"] = f"{s['mean_words']:.0f}", f"{s['fresh_mean_words']:.0f}"
        M["GlobLenRho"] = f"{G['spearman_store_level']['rho']:.2f}"


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
    revision2_assets(M)
    fig_noise(M)
    v2_assets(M)
    v15_assets(M)
    v16_assets(M)
    v17_assets(M)
    fig_pareto_heldout(M)
    lines = ["% AUTO-GENERATED by make_assets.py from measured result files. Do not edit."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    (HERE / "numbers.json").write_text(json.dumps(M, indent=1))
    for k in sorted(M):
        print(f"{k:22s} {M[k]}")


if __name__ == "__main__":
    main()
