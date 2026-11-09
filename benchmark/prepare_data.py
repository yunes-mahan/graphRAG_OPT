"""Download ogbn-arxiv / ogbl-collab (Open Graph Benchmark) and build the stream inputs.

Writes  $GCM_DATA/arxiv_versions.npz   (undirected unique edges, publication year, label, features)
        $GCM_DATA/collab_versions.npz  (built lazily by stream_eval.load("collab"))
        $GCM_DATA/text/titleabs.tsv.gz (titles/abstracts for the cost model and the QA experiment)
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TITLEABS = "https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"


def main():
    from ogb.linkproppred import LinkPropPredDataset
    from ogb.nodeproppred import NodePropPredDataset

    config.OGB_ROOT.mkdir(parents=True, exist_ok=True)
    NodePropPredDataset("ogbn-arxiv", root=str(config.OGB_ROOT))  # downloads raw files
    LinkPropPredDataset("ogbl-collab", root=str(config.OGB_ROOT))
    raw = config.OGB_ROOT / "ogbn_arxiv" / "raw"
    e = pd.read_csv(raw / "edge.csv.gz", header=None).values.T
    e = e[:, e[0] != e[1]]
    e = np.unique(np.sort(e, 0), axis=1)
    year = pd.read_csv(raw / "node_year.csv.gz", header=None).values.ravel()
    label = pd.read_csv(raw / "node-label.csv.gz", header=None).values.ravel()
    feat = pd.read_csv(raw / "node-feat.csv.gz", header=None).values.astype(np.float16)
    np.savez_compressed(config.ARXIV_NPZ, edges=e.astype(np.int32), year=year.astype(np.int16),
                        label=label.astype(np.int16), feat=feat)
    config.TEXTS.parent.mkdir(parents=True, exist_ok=True)
    if not config.TEXTS.exists():
        urllib.request.urlretrieve(TITLEABS, config.TEXTS)
    print("data ready in", config.DATA)


if __name__ == "__main__":
    main()
