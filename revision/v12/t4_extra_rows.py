"""T4 (V13): two more rows for Table 7 on Wikidata films and the flat LLM entity graph (5 orders each, same streams,
same evaluator and the same recompute reference as wikidata/eval.json and entity_graph/eval.json):
  local_repair    programs/local_repair_winner.py (unchanged), replayed with stream_eval.replay
  df_leiden_sahu  Sahu's Dynamic Frontier Leiden (authors' code, v2/dfleiden_driver, single thread, rust:1.83
                  container, as in v2/b2b_dfleiden.py); labels mapped to persistent identifiers with the same
                  map_to_existing_ids as all maintainers
Coverage = mean over versions of the share of present nodes in a community with >= 5 members (as b5_coverage_t6).
usage: python v12/t4_extra_rows.py local | export | df | check
  local  : run local repair (+ the recompute reference) -> v12/t4/extra_rows.json
  export : write DF-driver input files to v12/t4/df_work/
  df     : run the driver in the container (needs podman), then score -> v12/t4/extra_rows.json
  check  : the replayed recompute reference must equal Q of recompute_leiden in eval.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "wikidata"), str(A / "programs")]
import entity_graph_eval as eg  # noqa: E402
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402
from hybrid_refresh import map_to_existing_ids  # noqa: E402

OUT = HERE / "t4" / "extra_rows.json"
W = HERE / "t4" / "df_work"
LAM = 0.31
_WD = []


def stream(g, seed):
    if g == "wikidata":
        if not _WD:
            _WD.append(wd.build()[0])
        d = _WD[0]
        node_t, edge_t, T = se.schedule(d, seed)
        return d, node_t, edge_t, T
    return eg.stream(seed)


def coverage(labels, node_t):
    out = []
    for t, lab in enumerate(labels):
        _, cnt = np.unique(lab, return_counts=True)
        out.append(float(cnt[cnt >= 5].sum()) / len(lab))
    return float(np.mean(out))


def replay(m, d, node_t, edge_t, T):
    path, env = rs.SPECS[m]
    import os
    for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "PRICE_LAMBDA"):
        os.environ.pop(k, None)
    os.environ.update(env)
    return se.replay(str(path), d, node_t, edge_t, T)[0]


def record(res, g, m, seed, labels, d, node_t, edge_t, ref):
    r = wd.score(labels, d, node_t, edge_t, ref)
    res[f"{g}|{m}|{seed}"] = dict(Q=r["Q"], C=r["C"], score=r["Q"] - LAM * r["C"], coverage=coverage(labels, node_t))
    print(g, m, seed, {k: round(v, 4) for k, v in res[f"{g}|{m}|{seed}"].items()}, flush=True)


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def ref_of(res, g, seed, d, node_t, edge_t, T):
    k = f"{g}|_ref|{seed}"
    if k not in res:
        lab = replay("recompute_leiden", d, node_t, edge_t, T)
        res[k] = dict(full=wd.score(lab, d, node_t, edge_t)["full"], Q=wd.score(lab, d, node_t, edge_t)["Q"])
    return res[k]["full"]


def main(mode):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = load()
    jobs = [(g, s) for g in ("entity", "wikidata") for s in range(5)]
    if mode == "local":
        for g, s in jobs:
            if f"{g}|local_repair|{s}" in res:
                continue
            d, node_t, edge_t, T = stream(g, s)
            ref = ref_of(res, g, s, d, node_t, edge_t, T)
            record(res, g, "local_repair", s, replay("local_repair", d, node_t, edge_t, T), d, node_t, edge_t, ref)
            OUT.write_text(json.dumps(res, indent=1))
    elif mode == "export":
        W.mkdir(parents=True, exist_ok=True)
        for g, s in jobs:
            d, node_t, edge_t, T = stream(g, s)
            E = d["edges"]
            with open(W / f"{g}_s{s}.in", "w") as f:
                f.write(f"{len(node_t)} {T}\n")
                for t in range(T):
                    e = E[:, edge_t == t]
                    f.write(f"{e.shape[1]}\n" + "".join(f"{a} {b}\n" for a, b in e.T))
    elif mode == "df":
        script = " && ".join(f"/drv/df_driver < /work/{g}_s{s}.in > /work/{g}_s{s}.out 2>/dev/null" for g, s in jobs
                             if not (W / f"{g}_s{s}.out").exists() or (W / f"{g}_s{s}.out").stat().st_size == 0)
        if script:
            subprocess.run(["docker", "run", "--rm", "-v", f"{A / 'v2' / 'dfleiden_driver'}:/drv:ro", "-v", f"{W}:/work",
                            "docker.io/library/rust:1.83", "sh", "-c", script], check=True)
        for g, s in jobs:
            d, node_t, edge_t, T = stream(g, s)
            raw = [np.array(x.split(), dtype=np.int64) for x in (W / f"{g}_s{s}.out").read_text().splitlines()]
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
            ref = ref_of(res, g, s, d, node_t, edge_t, T)
            record(res, g, "df_leiden_sahu", s, labels, d, node_t, edge_t, ref)
            OUT.write_text(json.dumps(res, indent=1))
    elif mode == "check":
        for g, f in (("entity", A / "entity_graph" / "eval.json"), ("wikidata", A / "wikidata" / "eval.json")):
            ev = json.loads(f.read_text())
            for s in range(5):
                if f"{g}|_ref|{s}" in res:
                    print(g, s, "recompute Q replay", round(res[f"{g}|_ref|{s}"]["Q"], 6), "eval.json",
                          round(ev[f"recompute_leiden|{s}"]["Q"], 6))


if __name__ == "__main__":
    main(sys.argv[1])
