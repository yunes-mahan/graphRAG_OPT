#!/bin/bash
# V12 CPU chain (sequential, one process at a time): B1 flat sweep, then B2 (greedy/hungarian, entity/wikidata).
cd "$(dirname "$0")/.."
P=${PYTHON:-python}
export OMP_NUM_THREADS=1
$P v12/b1_flat_sweep.py > v12/b1/flat_arxiv.log 2>&1
for g in entity wikidata; do for mt in greedy hungarian; do
  HIER_GRAPH=$g HIER_MATCH=$mt HIER_BREAKDOWN=1 HIER_ONLY=reindex+match,warm HIER_SEEDS=0,1,2 \
  HIER_OUT=v12/b2/${g}_${mt}.json $P graphrag_hier.py > v12/b2/${g}_${mt}.log 2>&1
done; done
echo CPU_CHAIN_DONE > v12/cpu_chain.done
