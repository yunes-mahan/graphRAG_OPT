#!/bin/bash
# Quick reproducibility check (about 10-15 minutes on a laptop; no LLM calls, no API key needed).
#  1. regenerate every table, figure and number of the paper from artifacts/ and compare with the paper's values
#  2. download ogbn-arxiv / ogbl-collab (Open Graph Benchmark) and build the streams
#  3. re-run warm-started Leiden and warm-started Leiden + summary-priced rollback from scratch on the two
#     development streams and compare their token-level scores with the paper ("dev." column of the token table)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}

echo "[1/3] regenerating paper tables and numbers from artifacts/ ..."
( cd paper_assets && PAPER_RESULTS=../artifacts $PY make_assets.py > /dev/null )
if diff -q paper_assets/numbers.tex artifacts/expected_numbers.tex > /dev/null; then
  echo "      OK: all $(grep -c newcommand artifacts/expected_numbers.tex) reported numbers reproduced exactly"
else
  echo "      MISMATCH:"; diff paper_assets/numbers.tex artifacts/expected_numbers.tex | head; exit 1
fi

echo "[2/3] downloading and preparing ogbn-arxiv / ogbl-collab ..."
$PY benchmark/prepare_data.py
$PY -c "import sys; sys.path.insert(0, 'benchmark'); import stream_eval as se; se.load('collab')"
DATA=$($PY -c "import config; print(config.DATA)")
mkdir -p token_task/resources
ln -sf "$DATA/arxiv_versions.npz" token_task/resources/arxiv_versions.npz
ln -sf "$DATA/collab_versions.npz" token_task/resources/collab_versions.npz

echo "[3/3] re-running warm-started Leiden with and without summary-priced rollback (development orders) ..."
$PY benchmark/kgv2_dev.py warmstart_leiden,priced_warm | tee quick_reproduce.log
$PY paper_assets/check_quick.py quick_reproduce.log
