"""Neo4j demo: version-stable GraphRAG communities on a growing graph stored in Neo4j.

For every graph version (ogbn-arxiv replayed as 16 versions) we
  1. write the new :Paper nodes and :CITES relationships into Neo4j,
  2. LOCAL REPAIR (ours): update the community assignment incrementally in Python from the delta and
     write it to the node property `cid_local`,
  3. GDS WARM START (Neo4j-native baseline): project the current graph and run gds.<algo>.write with
     seedProperty = previous `cid_gds`, writing `cid_gds`,
  4. read both assignments back from Neo4j and measure modularity, summary regeneration and time.
"""

import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / 'benchmark'))
import config  # noqa: E402


import json
import sys
import time
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

import stream_eval as se  # noqa: E402

import os
URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])
HERE = config.RESULTS
BATCH = 20000


def chunks(rows, n=BATCH):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def run(algo="leiden", max_versions=None, seed=0, dataset="arxiv"):
    d = se.load(dataset)
    node_t, edge_t, T = se.schedule(d, seed)
    T = min(T, max_versions or T)
    edges, year = d["edges"], d["node_year"]
    maint = se.load_program(str(config.MAINTAINERS / "local_repair_winner.py"))()
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 50000 ROWS").consume()
        s.run("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE").consume()
        try:
            s.run("CALL gds.graph.drop('g', false)").consume()
        except Exception:
            pass
    log, labels_local, labels_gds, labels_re = [], [], [], []
    for t in range(T):
        new_nodes = np.where(node_t == t)[0]
        new_edges = edges[:, edge_t == t]
        all_nodes = np.where((node_t >= 0) & (node_t <= t))[0]
        rec = dict(version=t, nodes=int(len(all_nodes)), new_nodes=int(len(new_nodes)),
                   new_edges=int(new_edges.shape[1]))
        # 1) ingest the delta
        t0 = time.time()
        with drv.session() as s:
            for part in chunks([{"id": int(v), "year": int(year[v]) if int(year[v]) < 10000 else 0} for v in new_nodes]):
                s.run("UNWIND $rows AS r CREATE (:Paper {id: r.id, year: r.year})", rows=part).consume()
            for part in chunks([[int(a), int(b)] for a, b in new_edges.T]):
                s.run("UNWIND $rows AS r MATCH (a:Paper {id: r[0]}), (b:Paper {id: r[1]}) "
                      "CREATE (a)-[:CITES]->(b)", rows=part).consume()
        rec["ingest_s"] = time.time() - t0
        # 2) local repair (ours) on the delta, written back to Neo4j
        t0 = time.time()
        if t == 0:
            lab = maint.initialize(nodes=new_nodes, edges=new_edges, features=None, years=None)
        else:
            lab = maint.update(new_nodes=new_nodes, new_edges=new_edges, new_features=None,
                               new_years=None, all_nodes=all_nodes)
        rec["local_compute_s"] = time.time() - t0
        t0 = time.time()
        with drv.session() as s:
            for part in chunks([[int(v), int(c)] for v, c in zip(all_nodes, lab)]):
                s.run("UNWIND $rows AS r MATCH (p:Paper {id: r[0]}) SET p.cid_local = r[1]", rows=part).consume()
        rec["local_write_s"] = time.time() - t0
        # 3a) GDS warm start = seeded Louvain (seeded Leiden needs ~51 GB in GDS 2.13 at this size)
        t0 = time.time()
        with drv.session() as s:
            if t > 0:
                # new nodes get fresh, compact seed IDs above the current maximum (GDS sizes its
                # arrays by the largest community ID, so huge IDs would explode memory)
                maxc = s.run("MATCH (p:Paper) WHERE p.cid_gds IS NOT NULL RETURN max(p.cid_gds) AS m").single()["m"]
                rows = [[int(v), int(maxc) + 1 + i] for i, v in enumerate(new_nodes)]
                for part in chunks(rows):
                    s.run("UNWIND $rows AS r MATCH (p:Paper {id: r[0]}) SET p.cid_gds = r[1]", rows=part).consume()
            props = "['cid_gds']" if t > 0 else "[]"
            s.run(f"CALL gds.graph.project('g', 'Paper', {{CITES: {{orientation: 'UNDIRECTED'}}}}, "
                  f"{{nodeProperties: {props}}})").consume()
            cfg = "{writeProperty: 'cid_gds'" + (", seedProperty: 'cid_gds'" if t > 0 else "") + "}"
            res = s.run(f"CALL gds.louvain.write('g', {cfg}) YIELD modularity RETURN modularity").single()
            s.run("CALL gds.graph.drop('g')").consume()
        rec["gds_warm_s"] = time.time() - t0
        rec["gds_warm_modularity"] = float(res["modularity"])
        # 3b) naive GraphRAG baseline: GDS Leiden recomputed from scratch every version
        t0 = time.time()
        with drv.session() as s:
            s.run("CALL gds.graph.project('g', 'Paper', {CITES: {orientation: 'UNDIRECTED'}})").consume()
            res = s.run("CALL gds.leiden.write('g', {writeProperty: 'cid_recompute', randomSeed: 42, "
                        "concurrency: 1}) YIELD modularity RETURN modularity").single()
            s.run("CALL gds.graph.drop('g')").consume()
        rec["gds_recompute_s"] = time.time() - t0
        rec["gds_recompute_modularity"] = float(res["modularity"])
        # 4) read both assignments back from Neo4j (what a GraphRAG pipeline would see)
        with drv.session() as s:
            got = s.run("MATCH (p:Paper) RETURN p.id AS id, p.cid_local AS a, p.cid_gds AS b, p.cid_recompute AS c").data()
        m = {r["id"]: (r["a"], r["b"], r["c"]) for r in got}
        labels_local.append(np.array([m[int(v)][0] for v in all_nodes], dtype=np.int64))
        labels_gds.append(np.array([m[int(v)][1] for v in all_nodes], dtype=np.int64))
        labels_re.append(np.array([m[int(v)][2] for v in all_nodes], dtype=np.int64))
        log.append(rec)
        print(json.dumps(rec), flush=True)
    drv.close()
    d_cut = dict(d)
    sc_local = se.score(labels_local, d_cut, np.where(node_t < T, node_t, -1), np.where(edge_t < T, edge_t, 10**6))
    sc_gds = se.score(labels_gds, d_cut, np.where(node_t < T, node_t, -1), np.where(edge_t < T, edge_t, 10**6))
    sc_re = se.score(labels_re, d_cut, np.where(node_t < T, node_t, -1), np.where(edge_t < T, edge_t, 10**6))
    out = dict(algo="louvain-seeded+leiden-recompute", versions=T, log=log,
               local=dict(Q=sc_local["Q"], R=sc_local["R"], Q_curve=sc_local["Q_curve"], R_curve=sc_local["R_curve"]),
               gds_warm_louvain=dict(Q=sc_gds["Q"], R=sc_gds["R"], Q_curve=sc_gds["Q_curve"], R_curve=sc_gds["R_curve"]),
               gds_recompute_leiden=dict(Q=sc_re["Q"], R=sc_re["R"], Q_curve=sc_re["Q_curve"], R_curve=sc_re["R_curve"]))
    (HERE / f"demo_results_{dataset}_s{seed}_{T}.json").write_text(json.dumps(out, indent=1))
    upd = log[1:]
    print(f"\n== Neo4j demo, {dataset}, seed {seed}, {T} versions ==")
    print(f"local repair : Q={sc_local['Q']:.4f} R={sc_local['R']:.4f} "
          f"update time/version={np.mean([r['local_compute_s']+r['local_write_s'] for r in upd]):.1f}s")
    print(f"GDS Louvain warm (seeded): Q={sc_gds['Q']:.4f} R={sc_gds['R']:.4f} "
          f"update time/version={np.mean([r['gds_warm_s'] for r in upd]):.1f}s")
    print(f"GDS Leiden recompute : Q={sc_re['Q']:.4f} R={sc_re['R']:.4f} "
          f"update time/version={np.mean([r['gds_recompute_s'] for r in upd]):.1f}s")


if __name__ == "__main__":
    run(algo=sys.argv[1] if len(sys.argv) > 1 else "leiden",
        max_versions=int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None,
        dataset=sys.argv[3] if len(sys.argv) > 3 else "arxiv",
        seed=int(sys.argv[4]) if len(sys.argv) > 4 else 0)
