"""Wikipedia stream: (a) deletions and community splits, (b) cold-start placement.

(a) per year: links and pages that disappeared; communities (>= 5 members) whose surviving members are no longer
    connected inside the community after removing the deleted links (a split a maintainer must handle).
(b) pages first appearing in the last snapshot, placed by the nearest community prototype (mean-centred
    text-embedding-3-small of the previous snapshot's members); truth = community warm-started Leiden assigns
    them from links; candidates = communities with >= 5 members in the previous snapshot.
Writes wiki/extra.json.
"""

import hashlib
import json
from pathlib import Path

import igraph as ig
import numpy as np

import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
HERE = config.RESULTS / "wiki"
HERE.mkdir(parents=True, exist_ok=True)


def main():
    st = json.loads((HERE / "stream.json").read_text())
    cm = json.loads((HERE / "communities.json").read_text())
    years = [str(y) for y in st["years"]]
    S = st["snapshots"]
    lab = {y: {int(k): v for k, v in cm["labels"][y].items()} for y in years}
    out = dict(deletions=[], splits=[])
    for y0, y1 in zip(years, years[1:]):
        e0 = {(int(a), b) for a, v in S[y0].items() for b in v["links"]}
        e1 = {(int(a), b) for a, v in S[y1].items() for b in v["links"]}
        p0, p1 = set(map(int, S[y0])), set(map(int, S[y1]))
        out["deletions"].append(dict(year=y1, links_removed=len(e0 - e1), links_added=len(e1 - e0),
                                     pages_removed=len(p0 - p1), pages_added=len(p1 - p0)))
        # split check: members (at y0) of each community, links of y1 restricted to those members
        comm = {}
        for v, c in lab[y0].items():
            comm.setdefault(c, []).append(v)
        und = {(min(a, b), max(a, b)) for a, b in e1}
        n_split, n_big = 0, 0
        for c, mem in comm.items():
            if len(mem) < 5:
                continue
            n_big += 1
            alive = [v for v in mem if v in p1]
            pos = {v: i for i, v in enumerate(alive)}
            g = ig.Graph(n=len(alive), edges=[(pos[a], pos[b]) for a, b in und if a in pos and b in pos])
            comps = g.connected_components()
            big = [len(x) for x in comps if len(x) >= 2]
            if len(big) >= 2:
                n_split += 1
        out["splits"].append(dict(year=y1, communities=n_big, split=n_split))
    # (b) cold start in the last snapshot
    f = np.load(HERE / "emb_cache.npz")
    titles = st["titles"]
    yp, yl = years[-2], years[-1]

    def emb(y, v):
        return f[hashlib.sha1((titles[v] + ". " + S[y][str(v)]["text"]).encode()).hexdigest()]

    old = sorted(int(k) for k in S[yp])
    new = sorted(int(k) for k in S[yl] if k not in S[yp])
    sizes = {}
    for v in old:
        sizes[lab[yp][v]] = sizes.get(lab[yp][v], 0) + 1
    cand = [c for c, n in sizes.items() if n >= 5]
    new = [v for v in new if lab[yl].get(v) in cand]
    Xo = np.stack([emb(yp, v) for v in old])
    mu = Xo.mean(0)
    norm = lambda X: (X - mu) / np.linalg.norm(X - mu, axis=1, keepdims=True)  # noqa: E731
    Xo = norm(Xo)
    C = np.stack([Xo[[i for i, v in enumerate(old) if lab[yp][v] == c]].mean(0) for c in cand])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    Xn = norm(np.stack([emb(yl, v) for v in new]))
    truth = np.array([cand.index(lab[yl][v]) for v in new])
    rank = np.argsort(-(Xn @ C.T), 1)
    out["cold_start"] = dict(n_new=len(new), n_communities=len(cand), top1=float((rank[:, 0] == truth).mean()),
                             top3=float(np.mean([t in r[:3] for t, r in zip(truth, rank)])),
                             majority=float(max(np.mean(truth == i) for i in range(len(cand)))))
    print(json.dumps(out, indent=1))
    (HERE / "extra.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
