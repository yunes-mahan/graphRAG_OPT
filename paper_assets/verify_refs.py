"""Verify candidate references against Crossref (DOI) and the arXiv API; print canonical metadata."""

import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "ref-check/1.0 (academic paper preparation)"}

CANDIDATES = {
    # key: ("doi", value) or ("arxiv", value)
    "blondel2008louvain": ("doi", "10.1088/1742-5468/2008/10/P10008"),
    "traag2019leiden": ("doi", "10.1038/s41598-019-41695-z"),
    "newman2004modularity": ("doi", "10.1103/PhysRevE.69.026113"),
    "fortunato2010community": ("doi", "10.1016/j.physrep.2009.11.002"),
    "rossetti2018dynamic": ("doi", "10.1145/3172867"),
    "chakrabarti2006evolutionary": ("doi", "10.1145/1150402.1150467"),
    "greene2010tracking": ("doi", "10.1109/ASONAM.2010.17"),
    "zhuang2019dynamo": ("doi", "10.1109/TKDE.2019.2951419"),
    "hogan2021kg": ("doi", "10.1145/3447772"),
    "lewis2020rag": ("arxiv", "2005.11401"),
    "edge2024graphrag": ("arxiv", "2404.16130"),
    "guo2024lightrag": ("arxiv", "2410.05779"),
    "peng2024graphragsurvey": ("arxiv", "2408.08921"),
    "wang2025archrag": ("arxiv", "2502.09891"),
    "hu2020ogb": ("arxiv", "2005.00687"),
    "sahu2024dfleiden": ("arxiv", "2405.11658"),
}


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        m = json.load(r)["message"]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    authors = [f"{a.get('given','')} {a.get('family','')}".strip() for a in m.get("author", [])]
    return dict(title=(m.get("title") or [""])[0], authors=authors, year=year,
                venue=(m.get("container-title") or [""])[0], volume=m.get("volume"),
                pages=m.get("page"), doi=doi, type=m.get("type"))


def arxiv(aid):
    url = "http://export.arxiv.org/api/query?id_list=" + aid
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        root = ET.fromstring(r.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = root.find("a:entry", ns)
    title = re.sub(r"\s+", " ", e.find("a:title", ns).text).strip()
    if title.lower() == "error":
        raise ValueError("arXiv id not found")
    authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    year = int(e.find("a:published", ns).text[:4])
    return dict(title=title, authors=authors, year=year, venue="arXiv", arxiv=aid)


def main():
    out = {}
    for key, (kind, val) in CANDIDATES.items():
        try:
            meta = crossref(val) if kind == "doi" else arxiv(val)
            out[key] = meta
            a = meta["authors"]
            print(f"OK  {key:28s} {meta['year']} | {meta['title'][:80]} | {a[0] if a else '?'}"
                  f"{' et al.' if len(a) > 1 else ''} | {meta.get('venue','')[:40]}")
        except Exception as ex:  # noqa: BLE001
            out[key] = dict(error=str(ex))
            print(f"ERR {key:28s} {ex}")
        time.sleep(3.1 if kind == "arxiv" else 0.3)
    json.dump(out, open("refs_verified.json", "w"), indent=1)


if __name__ == "__main__":
    main()
