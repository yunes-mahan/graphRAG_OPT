#!/bin/bash
# V16/W4: Table 2 maintainers again (5 orders, both graphs) with Q on the LCC (singletons) and content coverage.
cd "$(dirname "$0")/.."
for g in entity wikidata; do for s in 0 1 2 3 4; do
  [ -f v16/w4/${g}_s${s}.json ] && continue
  HIER_GRAPH=$g HIER_W4=1 HIER_SEEDS=$s HIER_OUT=v16/w4/${g}_s${s}.json OMP_NUM_THREADS=1 \
    ${PY:-python} ${HIER:-graphrag_hier.py} > v16/w4/${g}_s${s}.log 2>&1 &
done; done; wait; echo DONE > v16/w4/done.txt
