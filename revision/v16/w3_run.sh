#!/bin/bash
# V16/W2+W3: entity graph, all hierarchical maintainers, 5 orders, GraphRAG-format context tokens and the cascade.
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do
  [ -f v16/w3/entity_s${s}.json ] && continue
  HIER_GRAPH=entity HIER_EXACT=1 HIER_SEEDS=$s HIER_OUT=v16/w3/entity_s${s}.json OMP_NUM_THREADS=1 \
    ${PY:-python} ${HIER:-graphrag_hier.py} > v16/w3/entity_s${s}.log 2>&1 &
done; wait; echo DONE > v16/w3/done.txt
