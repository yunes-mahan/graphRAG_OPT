#!/bin/bash
# V16/W5b: re-index + matching over the real streams with Leiden iterated 50 times (GraphRAG default: 1), orders 0-2.
cd "$(dirname "$0")/.."
for g in entity wikidata; do
  HIER_GRAPH=$g HIER_ITER=50 HIER_ONLY=reindex+match HIER_SEEDS=0,1,2 HIER_OUT=v16/w5_stream_${g}_iter50.json OMP_NUM_THREADS=1 \
    ${PY:-python} ${HIER:-graphrag_hier.py} > v16/w5_stream_${g}_iter50.log 2>&1 &
done; wait; echo DONE > v16/w5_stream_done.txt
