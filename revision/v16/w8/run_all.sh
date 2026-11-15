#!/bin/bash
cd "$(dirname "$0")/../.."
until [ -f v16/w3/done.txt ]; do sleep 60; done
xargs -P 6 -L 1 sh -c 'OMP_NUM_THREADS=1 ${PY:-python} v16/w8_deletions.py $0 $1 > v16/w8/$0_$1.log 2>&1' < v16/w8/jobs.txt
echo DONE > v16/w8/done.txt
