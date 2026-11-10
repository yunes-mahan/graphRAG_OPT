"""Do hierarchy-preserving consensus variants stay stable?  Table 3 columns plus hierarchy structure (levels, number
of top-level communities, largest top-level community) for GraphRAG's Leiden and consensus with 100 runs:
threshold (0.5, reattached), soft (all edges, weight = max(agreement, 0.05)), leafup (consensus leaves, upper levels by
GraphRAG's Leiden on the graph of leaves).  usage: python consensus_fix.py entity|wikidata -> consensus_fix_<g>.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import consensus_lib as cl  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import noise_hier as nh  # noqa: E402
import stability_variants as sv  # noqa: E402

graph = sys.argv[1]
if graph == "wikidata":
    import stream_eval as se
    import wd_eval as wd
    d = wd.build()[0]
    node_t, edge_t, T = se.schedule(d, 0)
else:
    d, node_t, edge_t, T = eg.stream(0)
n, E = len(node_t), d["edges"]
ok = (edge_t >= 0) & (edge_t < T)
el = [(int(a), int(b)) for a, b in E[:, ok].T]
before = [e for e, t in zip(el, edge_t[ok]) if t < T - 1]
rng = np.random.default_rng(0)
drops = []
for _ in range(3):
    keep = rng.random(len(el)) >= 0.001
    drops.append([e for e, k in zip(el, keep) if k])
variants = {"leiden": lambda e: sv.leiden(e, n),
            "threshold": lambda e: cl.consensus(e, n, 100, True, "threshold"),
            "soft": lambda e: cl.consensus(e, n, 100, True, "soft"),
            "leafup": lambda e: cl.consensus(e, n, 100, True, "leafup")}
out = {}
for name, fn in variants.items():
    t0 = time.time()
    ref = fn(el)
    q = sv.quality(ref, el, n)
    q["seconds"] = time.time() - t0
    q["drop"] = [nh.rewrite(ref, fn(x), 0.9) for x in drops]
    q["last"] = nh.rewrite(fn(before), ref, 0.9)
    out[name] = q
    print(graph, name, {k: (np.round(v, 3).tolist() if isinstance(v, (list, float)) else v) for k, v in q.items()}, flush=True)
    (HERE / f"consensus_fix_{graph}.json").write_text(json.dumps(out, indent=1, default=float))
