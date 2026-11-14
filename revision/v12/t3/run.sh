#!/bin/bash
# T3 (V13): GraphRAG update on the small-batch entity streams again, now also recording the coverage of new entities.
cd "$(dirname "$0")/../.."
for B in 1 10 100; do for s in 0 1 2; do
  [ -f v12/t3/gu_entity_b${B}_s${s}.json ] && continue
  HIER_SMALLBATCH=$B,10 HIER_ONLY=graphrag_update HIER_SEEDS=$s HIER_OUT=v12/t3/gu_entity_b${B}_s${s}.json OMP_NUM_THREADS=1 \
    v2/.venv_ext/bin/python graphrag_hier.py > v12/t3/gu_entity_b${B}_s${s}.log 2>&1 &
done; wait; done
echo DONE > v12/t3/done.txt
