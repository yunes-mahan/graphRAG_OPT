#!/bin/bash
# V16/P3: GraphRAG update with use_lcc on (default) and off, orders 0-2, recording the final top level.
cd "$(dirname "$0")/.."
for g in entity wikidata; do for lcc in 1 0; do
  HIER_GRAPH=$g HIER_GU_LCC=$lcc HIER_ONLY=graphrag_update HIER_SEEDS=0,1,2 HIER_OUT=v16/p3_${g}_lcc${lcc}.json OMP_NUM_THREADS=1 \
    ${PY:-python} ${HIER:-graphrag_hier.py} > v16/p3_${g}_lcc${lcc}.log 2>&1 &
done; done; wait; echo DONE > v16/p3_done.txt
