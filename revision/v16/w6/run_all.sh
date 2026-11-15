#!/bin/bash
cd "$(dirname "$0")/../.."
xargs -P 5 -L 1 sh -c 'OMP_NUM_THREADS=1 ${PY:-python} v16/w6_seeds.py $0 $1 > v16/w6/seed$0_$1.log 2>&1' < v16/w6/jobs.txt
echo DONE > v16/w6/done.txt
