#!/bin/bash
# V12 CPU chain 2: waits for chain 1 (done-file), then B7 GraphRAG-update use_lcc ablation.
cd "$(dirname "$0")/.."
until [ -f v12/cpu_chain.done ]; do sleep 60; done
P=${PYTHON:-python}
export OMP_NUM_THREADS=1
for g in entity wikidata; do
  HIER_GRAPH=$g HIER_GU_LCC=0 HIER_ONLY=graphrag_update HIER_SEEDS=0,1,2 HIER_OUT=v12/b7/${g}_gu_nolcc.json \
  $P graphrag_hier.py > v12/b7/${g}_gu_nolcc.log 2>&1
done
echo done > v12/cpu_chain2.done
