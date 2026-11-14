#!/bin/bash
# T2 (V13): one stream order of a Table-3 row that had no "stream" value. Args: graph mode type seed
cd "$(dirname "$0")/../.."
g=$1; m=$2; t=$3; s=$4; tag=${m}_${g}_${t}_s${s}
[ -f v12/t2/$tag.json ] && exit 0
HIER_GRAPH=$g HIER_ONLY=$m KCORE_TYPE=$t HIER_SEEDS=$s HIER_OUT=v12/t2/$tag.json OMP_NUM_THREADS=1 \
  v2/.venv_ext/bin/python graphrag_hier.py > v12/t2/$tag.log 2>&1
echo "$tag exit $?" >> v12/t2/done.txt
