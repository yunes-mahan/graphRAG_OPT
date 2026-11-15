#!/bin/bash
# V16/W1: hierarchical frontier maintainer (ours): 5 orders on both graphs (entity with GraphRAG-format tokens),
# then the small-batch sweep (entity graph, batches of 1/10/100 documents, 10 updates, seeds 0-2).
cd "$(dirname "$0")/.."
P=${PY:-python}
for s in 0 1 2 3 4; do
  [ -f v16/w1/entity_s${s}.json ] || HIER_GRAPH=entity HIER_W4=1 HIER_EXACT=1 HIER_ONLY=hier_frontier HIER_SEEDS=$s \
     HIER_OUT=v16/w1/entity_s${s}.json OMP_NUM_THREADS=1 $P ${HIER:-graphrag_hier.py} > v16/w1/entity_s${s}.log 2>&1 &
  [ -f v16/w1/wikidata_s${s}.json ] || HIER_GRAPH=wikidata HIER_W4=1 HIER_ONLY=hier_frontier HIER_SEEDS=$s \
     HIER_OUT=v16/w1/wikidata_s${s}.json OMP_NUM_THREADS=1 $P ${HIER:-graphrag_hier.py} > v16/w1/wikidata_s${s}.log 2>&1 &
done; wait
for B in 1 10 100; do for s in 0 1 2; do
  [ -f v16/w1/batch_b${B}_s${s}.json ] || HIER_SMALLBATCH=$B,10 HIER_W4=1 HIER_ONLY=hier_frontier HIER_SEEDS=$s \
     HIER_OUT=v16/w1/batch_b${B}_s${s}.json OMP_NUM_THREADS=1 $P ${HIER:-graphrag_hier.py} > v16/w1/batch_b${B}_s${s}.log 2>&1 &
done; done; wait; echo DONE > v16/w1/done.txt
