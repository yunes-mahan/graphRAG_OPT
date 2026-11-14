"""B3 (Table 6 part): cost-model sensitivity on the Wikidata film graph and the flat LLM entity graph (5 orders each,
the 8 Table 6 maintainers).  Replays record community sizes (same rewrite rule as wikidata/wd_eval.score: >= 5 members,
Jaccard 0.9), so every cost variant (one constant -50%/+50%) is a re-scoring of the same replays; each variant is scored
with its lambda recalibrated on the development orders (b3_flat_sensitivity.lam_of).
Difference to the Table 4 part (runtime): the rollback maintainers are NOT re-run per variant; their base runs (priced
with the base cost model) are re-scored.  Base check: the base-constant Q and C equal wikidata/eval.json and
entity_graph/eval.json.  ogbl-citation2 (one order, hours per replay) is not included.
Outputs: v2/b3x/ (sizes) and v2/b3_extra_sensitivity.json        usage: python v2/b3_extra_sensitivity.py run|analyze
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "wikidata"))
sys.path.insert(0, str(HERE))
import b3_flat_sensitivity as b3  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402

OUT = HERE / "b3x"
OUT.mkdir(exist_ok=True)
METHODS = ["recompute_leiden", "lazy", "warmstart_leiden", "frontier_leiden", "hybrid", "priced_warm", "priced_frontier",
           "graphrag_delta"]
GRAPHS = ("wikidata", "entity")


def data(graph, seed):
    if graph == "wikidata":
        d = wd.build()[0]
        node_t, edge_t, T = se.schedule(d, seed)
    else:
        d, node_t, edge_t, T = eg.stream(seed)
    return d, node_t, edge_t, T


def job(a):
    graph, m, seed, tag, env = a
    f = OUT / f"{graph}__{m}__{tag}__{seed}.json"
    if f.exists():
        return str(f)
    path, spec = rs.SPECS[m]
    for k in b3.ENVK:
        os.environ.pop(k, None)
    os.environ.update(spec)
    os.environ.update(env)
    d, node_t, edge_t, T = data(graph, seed)
    labels, _ = se.replay(str(path), d, node_t, edge_t, T)
    f.write_text(json.dumps(b3.sizes_of(labels, d, node_t, edge_t)))
    return str(f)


def load(graph, m, tag, seed):
    return json.loads((OUT / f"{graph}__{m}__{tag}__{seed}.json").read_text())


def QC(graph, m, tag, seed, v):
    r, ref = load(graph, m, tag, seed), load(graph, "recompute_leiden", "base", seed)
    return float(np.mean(r["Q"])), float(np.mean([b3.tok(x, v) / b3.tok(y, v) for x, y in zip(r["regen_sizes"], ref["full_sizes"][1:])]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        lam = {vn: b3.lam_of(v) for vn, v in b3.VARIANTS.items()}
        jobs = [(g, m, s, "base", {}) for g in GRAPHS for s in range(5) for m in METHODS]
        # Runtime: the rollback maintainers are NOT re-run per variant here (unlike Table 4); their base runs (priced
        # with the base cost model) are re-scored under each variant.  Stated in the output and in the paper.
        with Pool(int(os.environ.get("NPROC", "4"))) as p:
            for i, f in enumerate(p.imap_unordered(job, jobs)):
                print(i, f, flush=True)
    else:
        stored = {"wikidata": json.loads((HERE.parent / "wikidata" / "eval.json").read_text()),
                  "entity": json.loads((HERE.parent / "entity_graph" / "eval.json").read_text())}
        res = dict(base_check={}, variants={})
        for g in GRAPHS:
            res["base_check"][g] = max(max(abs(QC(g, m, "base", s, b3.BASE)[0] - stored[g][f"{m}|{s}"]["Q"]),
                                           abs(QC(g, m, "base", s, b3.BASE)[1] - stored[g][f"{m}|{s}"]["C"]))
                                       for m in METHODS for s in range(5))
        base = {g: {m: float(np.mean([QC(g, m, "base", s, b3.BASE)[0] - 0.31 * QC(g, m, "base", s, b3.BASE)[1]
                                      for s in range(5)])) for m in METHODS} for g in GRAPHS}
        res["base_scores"] = base
        for vn, v in b3.VARIANTS.items():
            if vn == "base":
                continue
            lam = b3.lam_of(v)
            out = {}
            for g in GRAPHS:
                sc = {m: float(np.mean([QC(g, m, "base", s, v)[0]
                                        - lam * QC(g, m, "base", s, v)[1]
                                        for s in range(5)])) for m in METHODS}
                x, y = [base[g][m] for m in METHODS], [sc[m] for m in METHODS]
                out[g] = dict(lambda_=lam, scores=sc, ranking=sorted(METHODS, key=lambda m: -sc[m]),
                              spearman=float(stats.spearmanr(x, y).statistic), kendall=float(stats.kendalltau(x, y).statistic),
                              top1_same=max(sc, key=sc.get) == max(base[g], key=base[g].get))
            res["variants"][vn] = out
            print(vn, {g: (round(o["spearman"], 3), round(o["kendall"], 3), o["ranking"][:2]) for g, o in out.items()})
        res["rollback_pricing"] = "base cost model (not re-run per variant); re-scored under each variant"
        print("base check", res["base_check"])
        (HERE / "b3_extra_sensitivity.json").write_text(json.dumps(res, indent=1))
