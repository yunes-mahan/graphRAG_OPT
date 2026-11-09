#!/bin/bash
# Scalability study: every method in its own process (clean timing and peak memory), recompute first (reference).
cd "$(dirname "$0")"
P=${PYTHON:-python}
for f in 0.25 0.5 1.0; do
  for m in recompute_leiden lazy warmstart_leiden frontier_leiden priced_warm priced_frontier; do
    $P scale_citation2.py run $f $m || echo "FAILED $f $m"
  done
done
echo SCALE_DONE
