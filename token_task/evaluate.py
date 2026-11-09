"""
Evaluator for kgversion2: summary-aware community maintenance for GraphRAG on TWO growing graphs.

Streams (development; the program sees one stream at a time, online):
  arxiv  : ogbn-arxiv citation graph. V0 = papers up to 2015, then 2016-2020 in 3 random batches per year
           (16 versions). A paper and its references arrive together.
  collab : ogbl-collab co-authorship graph. V0 = everything up to 2012, then 2013-2017 in 3 random batches
           per year (16 versions). An author arrives with the first collaboration; every collaboration edge
           arrives in its own year, so later edges also connect EXISTING nodes.

Score (higher is better), averaged over the two streams:
    stream_score = mean_t Q_t  -  LAMBDA * mean_{t>=1} C_t
    combined_score = (stream_score_arxiv + stream_score_collab) / 2
  Q_t : modularity (resolution 1) of the partition of version t
  C_t : SUMMARY TOKEN COST of update t, relative to recomputing Leiden from scratch:
        C_t = regenerated_tokens_t / reference_tokens_t
        Only communities with >= 5 members carry an LLM summary. A summary must be (re)generated if the
        community ID is new or newly reaches 5 members, or Jaccard(old members, new members) < 0.9.
        tokens(summary) = 1500 + min(8000, 250 * size)   (prompt overhead + output, member texts capped)
        reference_tokens_t = tokens to summarise ALL >= 5 communities of Leiden recomputed on version t
        (precomputed constant, resources/reference_tokens.json), so recompute-from-scratch costs ~1.
  LAMBDA calibrated so that "recompute Leiden every version" and "never recompute (lazy)" tie on average.
"""

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List

import igraph as ig
import numpy as np

from capture import OutputCapture


def _task_dir() -> Path:
    for c in [os.environ.get("TASK_SOURCE_DIR", "").strip(), "/task-source", str(Path(__file__).resolve().parent)]:
        if c and (Path(c) / "resources" / "arxiv_versions.npz").is_file():
            return Path(c).resolve()
    raise FileNotFoundError("resources/ not found (checked TASK_SOURCE_DIR, /task-source, evaluator dir)")


TASK_DIR = _task_dir()
RES = TASK_DIR / "resources"
PROGRAM_TIMEOUT_SECONDS = 20 * 60          # per stream
LAMBDA = float(json.loads((RES / "calibration.json").read_text())["lambda"]) \
    if (RES / "calibration.json").exists() else 0.15
JACCARD_KEEP, MIN_SIZE, OVERHEAD, PER_MEMBER, CAP = 0.9, 5, 1500, 250, 8000
STREAMS = {"arxiv": dict(base=2015, last=2020, seed=0), "collab": dict(base=2012, last=2017, seed=0)}
BATCHES = 3


# --------------------------------------------------------------------------- data / schedule
def load(name: str):
    if name == "arxiv":
        z = np.load(RES / "arxiv_versions.npz")
        edges = z["edges"].astype(np.int64)
        year = z["year"].astype(np.int64)
        return dict(edges=edges, edge_year=np.maximum(year[edges[0]], year[edges[1]]), node_year=year,
                    feat=z["feat"], edge_mode="nodes")
    z = np.load(RES / "collab_versions.npz")
    return dict(edges=z["edges"].astype(np.int64), edge_year=z["edge_year"].astype(np.int64),
                node_year=z["node_year"].astype(np.int64), feat=z["feat"], edge_mode="own-year")


def schedule(d: dict, base: int, last: int, seed: int):
    rng = np.random.default_rng(seed)
    year = d["node_year"]
    node_t = np.full(len(year), -1, dtype=np.int64)
    node_t[year <= base] = 0
    t, year_start = 1, {}
    for yr in range(base + 1, last + 1):
        year_start[yr] = t
        for part in np.array_split(rng.permutation(np.where(year == yr)[0]), BATCHES):
            node_t[part] = t
            t += 1
    edges = d["edges"]
    if d["edge_mode"] == "nodes":
        edge_t = np.maximum(node_t[edges[0]], node_t[edges[1]])
    else:
        ey = d["edge_year"]
        own = np.zeros(len(ey), dtype=np.int64)
        late = ey > base
        own[late] = np.array([year_start[y] for y in ey[late]]) + rng.integers(0, BATCHES, late.sum())
        edge_t = np.maximum(np.maximum(node_t[edges[0]], node_t[edges[1]]), own)
    return node_t, edge_t, t


# --------------------------------------------------------------------------- candidate runner
def load_function_from_file(program_path: str, fn_name: str):
    program_dir = os.path.dirname(os.path.abspath(program_path))
    if program_dir and program_dir not in sys.path:
        sys.path.insert(0, program_dir)
    spec = importlib.util.spec_from_file_location("program_module", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["program_module"] = module
    spec.loader.exec_module(module)
    return getattr(module, fn_name)


def _run_candidate(program_path: str, stream: str, q: mp.Queue) -> None:
    try:
        d = load(stream)
        s = STREAMS[stream]
        node_t, edge_t, T = schedule(d, s["base"], s["last"], s["seed"])
        maint = load_function_from_file(program_path, "make_maintainer")()
        edges, feat, year = d["edges"], d["feat"], d["node_year"]
        out, times = [], []
        for t in range(T):
            new_nodes = np.where(node_t == t)[0]
            new_edges = edges[:, edge_t == t]
            all_nodes = np.where((node_t >= 0) & (node_t <= t))[0]
            t0 = time.time()
            if t == 0:
                lab = maint.initialize(nodes=new_nodes.copy(), edges=new_edges.copy(),
                                       features=feat[new_nodes].astype(np.float32), years=year[new_nodes].copy())
            else:
                lab = maint.update(new_nodes=new_nodes.copy(), new_edges=new_edges.copy(),
                                   new_features=feat[new_nodes].astype(np.float32),
                                   new_years=year[new_nodes].copy(), all_nodes=all_nodes.copy())
            times.append(time.time() - t0)
            lab = np.asarray(lab)
            if lab.shape != (len(all_nodes),):
                raise ValueError(f"{stream} version {t}: expected ({len(all_nodes)},) labels, got {lab.shape}")
            if not np.issubdtype(lab.dtype, np.integer):
                raise ValueError(f"{stream} version {t}: community IDs must be integers")
            out.append(lab.astype(np.int64))
        q.put({"ok": True, "labels": out, "times": times})
    except Exception as e:  # noqa: BLE001
        q.put({"ok": False, "error": str(e), "traceback": traceback.format_exc()})


def run_candidate(program_path: str, stream: str):
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    q = ctx.Queue(maxsize=1)
    p = ctx.Process(target=_run_candidate, args=(program_path, stream, q))
    p.start()
    deadline, payload = time.time() + PROGRAM_TIMEOUT_SECONDS, None
    while time.time() < deadline:
        try:
            payload = q.get(timeout=5)
            break
        except Empty:
            if not p.is_alive():
                break
    if payload is None:
        if p.is_alive():
            p.terminate()
            p.join(5)
            raise TimeoutError(f"{stream}: timed out after {PROGRAM_TIMEOUT_SECONDS} s")
        raise RuntimeError(f"{stream}: candidate exited with code {p.exitcode} without a result")
    p.join(10)
    if not payload.get("ok"):
        raise RuntimeError(f"{payload.get('error')}\n{payload.get('traceback', '')}".strip())
    return payload["labels"], payload["times"]


# --------------------------------------------------------------------------- scoring
def summary_tokens(size):
    return OVERHEAD + np.minimum(CAP, PER_MEMBER * size)


def score_stream(labels: List[np.ndarray], stream: str, ref_tokens=None) -> Dict[str, Any]:
    d = load(stream)
    s = STREAMS[stream]
    node_t, edge_t, _ = schedule(d, s["base"], s["last"], s["seed"])
    edges, n_total = d["edges"], len(d["node_year"])
    SENT = np.iinfo(np.int64).min
    Q, regen_tok, full_tok, n_summ, n_regen, moved = [], [], [], [], [], []
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
        full_tok.append(float(summary_tokens(size_new[big]).sum()))
        n_summ.append(int(big.sum()))
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
            regen_tok.append(float(summary_tokens(size_new[regen]).sum()))
            n_regen.append(int(regen.sum()))
            moved.append(float((prev_full[old_mask] != full[old_mask]).mean()))
        prev_full = full
    out = dict(mean_modularity=float(np.mean(Q)), regenerated_tokens_per_update=[round(x) for x in regen_tok],
               summaries_per_version=n_summ, regenerated_summaries_per_update=n_regen,
               modularity_per_version=[round(x, 4) for x in Q],
               moved_existing_nodes_per_update=[round(x, 4) for x in moved],
               full_tokens_per_version=full_tok)
    if ref_tokens is not None:
        C = [r / ref for r, ref in zip(regen_tok, ref_tokens[1:])]
        out["mean_relative_token_cost"] = float(np.mean(C))
        out["relative_token_cost_per_update"] = [round(x, 4) for x in C]
        out["stream_score"] = out["mean_modularity"] - LAMBDA * out["mean_relative_token_cost"]
    return out


def main(program_path: str, results_dir: str, verbose: bool = True):
    os.makedirs(results_dir, exist_ok=True)
    metrics: Dict[str, Any] = {}
    correct, error_msg = False, None
    ref = json.loads((RES / "reference_tokens.json").read_text())
    with OutputCapture(results_dir, tee=verbose) as capture:
        print(f"Evaluating program: {program_path}  (lambda = {LAMBDA:.4f})")
        try:
            pub = {}
            for stream in STREAMS:
                t0 = time.time()
                print(f"[{stream}] replaying 16 versions (timeout {PROGRAM_TIMEOUT_SECONDS}s)...")
                labels, times = run_candidate(program_path, stream)
                sc = score_stream(labels, stream, ref[stream])
                sc["candidate_seconds_total"] = float(sum(times))
                pub[stream] = sc
                print(f"[{stream}] score={sc['stream_score']:.4f} (modularity {sc['mean_modularity']:.4f}, "
                      f"relative token cost {sc['mean_relative_token_cost']:.4f}) in {time.time() - t0:.0f}s")
            combined = float(np.mean([pub[s]["stream_score"] for s in STREAMS]))
            metrics = {"combined_score": combined, "public": pub,
                       "private": {"lambda": LAMBDA, "jaccard_keep": JACCARD_KEEP, "min_size": MIN_SIZE}}
            correct = bool(np.isfinite(combined))
            error_msg = None if correct else "non-finite score"
            print(f"combined_score={combined:.4f}")
        except Exception as e:  # noqa: BLE001
            error_msg = f"Execution error: {e}"
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            metrics = {"combined_score": 0.0, "error": error_msg}
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(results_dir, "correct.json"), "w") as f:
            json.dump({"correct": correct, "message": error_msg}, f, indent=2)
        print(f"Saved {capture.stdout_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="kgversion2 evaluator (writes metrics.json + correct.json)")
    ap.add_argument("--program_path", type=str, default="program_0/src/program.py")
    ap.add_argument("--results_dir", type=str, default="results")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    main(a.program_path, a.results_dir, verbose=not a.quiet)
