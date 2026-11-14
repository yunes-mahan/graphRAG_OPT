#!/bin/bash
cd "$(dirname "$0")"
xargs -P 4 -L 1 ./run_one.sh < jobs.txt
echo ALLDONE >> done.txt
