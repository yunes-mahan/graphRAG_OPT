"""Maintainers on an LLM-extracted GraphRAG entity graph (structural evaluation, 5 document orders).

Graph: entities and relationships extracted by entity_extract.py from 6,000 cs.IR/DB/SI/DL abstracts (2010-2020),
entities merged by normalised name, relationships undirected and de-duplicated (as GraphRAG's entity graph).
Stream over DOCUMENTS: V0 = documents up to 2016, then 2017-2020 in 3 random batches per year (seed = order);
an entity arrives with the first document that mentions it, a relationship with the first document that states it
(so later documents also add relationships between existing entities).
Score: token-level score (lambda 0.31) as for the other graphs. Writes entity_graph/eval.json and graph.json.
usage: python entity_graph_eval.py build | run
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
sys.path.insert(0, str(HERE / "wikidata"))
import run_suite as rs  # noqa: E402
import stream_eval as se  # noqa: E402
import wd_eval as wd  # noqa: E402

G = HERE / "entity_graph"
LAM, BASE, LAST, BATCHES = 0.31, 2016, 2020, 3
METHODS = ["recompute_leiden", "lazy", "warmstart_leiden", "frontier_leiden", "hybrid", "graphrag_delta",
           "priced_warm", "priced_frontier"]


def build():
    ex = json.loads((G / "extractions.json").read_text())
    docs = sorted(((int(d), v) for d, v in ex.items()), key=lambda x: x[0])
    ent, etype, edges, doc_ents, doc_rels = {}, {}, {}, {}, {}
    for d, v in docs:
        names = set()
        for e in v["entities"]:
            if e["name"] not in ent:
                ent[e["name"]] = len(ent)
                etype[e["name"]] = e["type"]
            names.add(ent[e["name"]])
        rels = set()
        for r in v["relationships"]:
            for x in (r["source"], r["target"]):
                if x not in ent:
                    ent[x] = len(ent)
                    etype[x] = "concept"
                names.add(ent[x])
            a, b = ent[r["source"]], ent[r["target"]]
            if a != b:
                key = (min(a, b), max(a, b))
                edges.setdefault(key, r["description"])
                rels.add(key)
        doc_ents[d], doc_rels[d] = sorted(names), sorted(rels)
    keys = sorted(edges)
    kidx = {k: i for i, k in enumerate(keys)}
    out = dict(entities=sorted(ent, key=ent.get), types=[etype[n] for n in sorted(ent, key=ent.get)],
               edges=[list(k) for k in keys], edge_desc=[edges[k] for k in keys],
               docs={str(d): dict(year=v["year"], ents=doc_ents[d], rels=[kidx[k] for k in doc_rels[d]])
                     for d, v in docs})
    (G / "graph.json").write_text(json.dumps(out))
    deg = np.bincount(np.array(keys).ravel(), minlength=len(ent))
    print(f"documents {len(docs)}, entities {len(ent)} ({(deg > 0).sum()} with relations), relationships {len(keys)}")


def stream(seed):
    g = json.loads((G / "graph.json").read_text())
    n, E = len(g["entities"]), np.array(g["edges"], dtype=np.int64).T
    rng = np.random.default_rng(seed)
    years = {d: v["year"] for d, v in g["docs"].items()}
    doc_t, t = {}, 1
    for d, y in years.items():
        if y <= BASE:
            doc_t[d] = 0
    for y in range(BASE + 1, LAST + 1):
        ids = rng.permutation(sorted(d for d, yy in years.items() if yy == y))
        for part in np.array_split(ids, BATCHES):
            for d in part:
                doc_t[d] = t
            t += 1
    node_t = np.full(n, 10 ** 6, dtype=np.int64)
    edge_t = np.full(E.shape[1], 10 ** 6, dtype=np.int64)
    for d, v in g["docs"].items():
        for e in v["ents"]:
            node_t[e] = min(node_t[e], doc_t[d])
        for r in v["rels"]:
            edge_t[r] = min(edge_t[r], doc_t[d])
    node_t[node_t == 10 ** 6] = -1
    d = dict(edges=E, node_year=np.zeros(n, dtype=np.int64), edge_year=np.zeros(E.shape[1], dtype=np.int64),
             feat=np.zeros((n, 1), np.float32), label=None, base=BASE, last=LAST, edge_mode="nodes")
    return d, node_t, edge_t, t


def run():
    f = G / "eval.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    for seed in range(5):
        d, node_t, edge_t, T = stream(seed)
        ref = None
        for m in METHODS:
            path, env = rs.SPECS[m]
            for k in ("HYBRID_DELTA", "HYBRID_BASE", "HYBRID_TRIGGER", "HYBRID_K", "SEM_CENTER", "PRICE_LAMBDA"):
                os.environ.pop(k, None)
            os.environ.update(env)
            labels, times = se.replay(str(path), d, node_t, edge_t, T)
            if m == "recompute_leiden":
                ref = wd.score(labels, d, node_t, edge_t)["full"]
            r = wd.score(labels, d, node_t, edge_t, ref)
            res[f"{m}|{seed}"] = dict(Q=r["Q"], C=r["C"], score=r["Q"] - LAM * r["C"], seconds=float(sum(times)))
            if seed == 0:
                res[f"{m}|labels0"] = [lab.tolist() for lab in labels]
            f.write_text(json.dumps(res))
            print(seed, m, {k: round(v, 4) for k, v in res[f"{m}|{seed}"].items()}, flush=True)
    print("\nmeans over 5 orders:")
    for m in METHODS:
        v = [res[f"{m}|{s}"] for s in range(5)]
        print(f"{m:18s} score={np.mean([x['score'] for x in v]):.4f}  Q={np.mean([x['Q'] for x in v]):.4f}  "
              f"C={np.mean([x['C'] for x in v]):.4f}")
    (G / "stream_meta.json").write_text(json.dumps(dict(versions=T, nodes=int((node_t >= 0).sum()),
                                                        edges=int(d["edges"].shape[1]))))


if __name__ == "__main__":
    {"build": build, "run": run}[sys.argv[1]]()
