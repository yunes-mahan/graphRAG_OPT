"""C1 step 1 (CPU, run with analysis/v2/.venv_ext because the authors' k-core code needs graspologic):
build the three hierarchies of the FINAL entity graph (order 0, as Table 3) and store their communities.
  leiden    : GraphRAG's hierarchical Leiden (max cluster 10, LCC, seed 0xDEADBEEF)          [level 0 = top, 111]
  kcore_rkh : authors' k-core hierarchy, RkH (v2/b2_kcore_authors.authors_kcore)             [level 0 = 1 community]
  consensus : consensus clustering, 100 runs, threshold 0.5 + reattachment (Table 3 row)       [level 0 = top]
Output: v12_qwen/c1_hierarchies.json  {name: [[level, [members...]], ...]}
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path[:0] = [str(A), str(A / "v2"), str(A / "wikidata")]
import b2_kcore_authors as ka  # noqa: E402
import consensus_lib as cl  # noqa: E402
import entity_graph_eval as eg  # noqa: E402
import stability_variants as sv  # noqa: E402

d, node_t, edge_t, T = eg.stream(0)
n, E = len(node_t), d["edges"]
ok = (edge_t >= 0) & (edge_t < T)
el = [(int(a), int(b)) for a, b in E[:, ok].T]
H = {"leiden": sv.leiden(el, n), "kcore_rkh": ka.authors_kcore(el, n, "RkH"), "consensus": cl.consensus(el, n, 100, True)}
out = {k: [[int(l), sorted(int(v) for v in m)] for (l, _c), m in c.items()] for k, c in H.items()}
for k, c in out.items():
    lv = {}
    for l, m in c:
        lv[l] = lv.get(l, 0) + 1
    print(k, "communities per level", dict(sorted(lv.items())), flush=True)
(HERE / "c1_hierarchies.json").write_text(json.dumps(out))
