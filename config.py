"""Central paths. Override the data/results locations with GCM_DATA / GCM_RESULTS."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = Path(os.environ.get("GCM_DATA", REPO / "data")).resolve()
RESULTS = Path(os.environ.get("GCM_RESULTS", REPO / "results")).resolve()
MAINTAINERS = REPO / "maintainers"
ARXIV_NPZ = DATA / "arxiv_versions.npz"
OGB_ROOT = DATA / "ogb"
TEXTS = DATA / "text" / "titleabs.tsv.gz"
RESULTS.mkdir(parents=True, exist_ok=True)
