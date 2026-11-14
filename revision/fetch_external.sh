#!/bin/bash
# Download third-party code used in the VLDB revision at pinned commits (not redistributed: the k-core repository
# has no licence).  Files land in analysis/v2/external/ and are used unmodified.
set -e
cd "$(dirname "$0")/external"
# Hossain & Sariyuce, "Core-based Hierarchies for Efficient GraphRAG" (KDD 2026), authors' code
K=https://raw.githubusercontent.com/jakir-sust/Kcore-GraphRAG/ae7b2ef
mkdir -p kcore_graphrag/graphrag/index/operations kcore_graphrag/graphrag/index/utils
curl -sfL "$K/graphrag/index/operations/kcore_cluster_graph.py" -o kcore_graphrag/graphrag/index/operations/kcore_cluster_graph.py
curl -sfL "$K/graphrag/index/utils/stable_lcc.py" -o kcore_graphrag/graphrag/index/utils/stable_lcc.py
# Unofficial third-party HIT-Leiden (validation only, see b1_hitleiden_validate.py; NOT used in the paper's tables)
[ -d hit-leiden ] || git clone -q https://github.com/randomvariable/hit-leiden hit-leiden
git -C hit-leiden checkout -q 924742400b06e36f90642757c2f3fc0ba8030507
# Sahu's dynamic-frontier Leiden (C++/OpenMP), see b2b_dfleiden.py
[ -d leiden-dynamic ] || git clone -q https://github.com/puzzlef/leiden-communities-openmp-dynamic leiden-dynamic
shasum -a 256 kcore_graphrag/graphrag/index/operations/kcore_cluster_graph.py kcore_graphrag/graphrag/index/utils/stable_lcc.py
# PATCH (documented in CHANGES.md, B2): the authors' RkH branch increments PRINT_SINGLETON (a print-only debug
# counter, never read) before it is initialised, which raises UnboundLocalError when a 2-hop group exceeds
# MAX_CLUSTER_SIZE.  We initialise it once at function start; the clustering logic is unchanged.
F=kcore_graphrag/graphrag/index/operations/kcore_cluster_graph.py
grep -q "PRINT_SINGLETON = 0  # patch" $F || sed -i.orig 's/^    MAX_CLUSTER_SIZE, MIN_CLUSTER_SIZE = 20, 1$/    MAX_CLUSTER_SIZE, MIN_CLUSTER_SIZE = 20, 1\n    PRINT_SINGLETON = 0  # patch: print-only counter, see CHANGES.md/' $F
diff $F.orig $F > kcore_patch.diff || true
