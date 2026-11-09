"""Crawl an evolving-text Wikipedia graph: ML/AI articles at yearly snapshots (links + lead text).

1. article set   : pages (namespace 0) in 8 root categories and their direct subcategories
2. revisions     : for every page, the revision that was current on 1 January of each snapshot year
3. content       : wikitext of those revisions (batched 50 revids per request)
4. redirects     : redirects to the article set (to resolve links)
5. parse         : per (page, year) the lead text (plain, first 1500 chars) and the wikilinks inside the set
Output: wiki/stream.json  {titles, years, snapshots: {year: {page_idx: {"text": str, "links": [idx]}}}}
Polite: one request at a time, >= 1 s apart, honours HTTP 429 / Retry-After; fully resumable (cache/).
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import mwparserfromhell
import requests

import sys as _s
_s.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
HERE = config.RESULTS / "wiki"
HERE.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
API = "https://en.wikipedia.org/w/api.php"
YEARS = list(range(2017, 2026))
ROOTS = ["Machine learning", "Artificial intelligence", "Natural language processing", "Computer vision",
         "Deep learning", "Data mining", "Knowledge representation", "Information retrieval"]
S = requests.Session()
S.headers["User-Agent"] = "graphrag-maintenance-research/0.1 (academic benchmark; python-requests)"
_last = [0.0]


def get(**params):
    params.update(action="query", format="json", formatversion=2)
    for attempt in range(12):
        wait = 1.1 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        try:
            r = S.get(API, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            time.sleep(int(ra) if ra and ra.isdigit() else min(600, 30 * 2 ** attempt))
            continue
        return r.json()
    raise RuntimeError("API keeps failing")


def cached(name, fn):
    f = CACHE / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    v = fn()
    f.write_text(json.dumps(v))
    return v


def members(cat, kind):
    out, cont = [], {}
    while True:
        r = get(list="categorymembers", cmtitle="Category:" + cat, cmtype=kind, cmlimit=500, **cont)
        out += r["query"]["categorymembers"]
        if "continue" not in r:
            return out
        cont = r["continue"]


def article_set():
    pages = {}
    for root in ROOTS:
        cats = [root] + [c["title"].split(":", 1)[1] for c in members(root, "subcat")]
        for c in cats:
            for p in cached("cat_" + re.sub(r"\W", "_", c), lambda c=c: members(c, "page")):
                if p["ns"] == 0 and not p["title"].startswith(("List of", "Outline of", "Glossary of", "Timeline of")):
                    pages[p["title"]] = p["pageid"]
        print(f"{root}: total pages {len(pages)}", flush=True)
    return pages


def snapshot_revids(pageid):
    """revision id current at 1 Jan of each year (None if the page did not exist yet).
    Lists revision ids/timestamps newest-first from 1 Jan of the last year until one older than the first year."""
    stamps, cont = [], {}
    first = f"{YEARS[0]}-01-01T00:00:00Z"
    while True:
        r = get(prop="revisions", pageids=pageid, rvprop="ids|timestamp", rvlimit=500, rvdir="older",
                rvstart=f"{YEARS[-1]}-01-01T00:00:00Z", **cont)
        revs = r["query"]["pages"][0].get("revisions", [])
        stamps += [(rv["timestamp"], rv["revid"]) for rv in revs]
        if not revs or revs[-1]["timestamp"] < first or "continue" not in r:
            break
        cont = r["continue"]
    out = {}
    for y in YEARS:
        d = f"{y}-01-01T00:00:00Z"
        older = [rv for ts, rv in stamps if ts < d]
        out[str(y)] = older[0] if older else None      # newest-first: first one older than d
    return out


def contents(revids):
    out = {}
    for i in range(0, len(revids), 50):
        r = get(prop="revisions", revids="|".join(map(str, revids[i:i + 50])), rvprop="ids|content", rvslots="main")
        for p in r["query"]["pages"]:
            for rv in p.get("revisions", []):
                out[str(rv["revid"])] = rv.get("slots", {}).get("main", {}).get("content", "")
    return out


def redirects(titles):
    out = {}
    for i in range(0, len(titles), 50):
        cont = {}
        while True:
            r = get(prop="redirects", titles="|".join(titles[i:i + 50]), rdlimit="max", rdnamespace=0, **cont)
            for p in r["query"]["pages"]:
                for rd in p.get("redirects", []):
                    out[rd["title"]] = p["title"]
            if "continue" not in r:
                break
            cont = r["continue"]
    return out


def parse(wikitext):
    code = mwparserfromhell.parse(wikitext)
    links = {str(l.title).split("#")[0].strip() for l in code.filter_wikilinks()}
    lead = wikitext.split("\n==", 1)[0]
    text = mwparserfromhell.parse(lead).strip_code()
    text = re.sub(r"\[\[(File|Image|Category):[^\]]*\]\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1500], links


def norm(t):
    t = t.replace("_", " ").strip()
    return t[:1].upper() + t[1:] if t else t


def main():
    pages = cached("pages", article_set)
    titles = sorted(pages)[:int(os.environ.get("MAX_PAGES", "2500"))]   # alphabetical cap (rate limits)
    print(f"articles: {len(titles)}", flush=True)
    rev_file = CACHE / "revids.json"
    revs = json.loads(rev_file.read_text()) if rev_file.exists() else {}
    for i, t in enumerate(titles):
        if t in revs:
            continue
        revs[t] = snapshot_revids(pages[t])
        if i % 50 == 0:
            rev_file.write_text(json.dumps(revs))
            print(f"revids {i}/{len(titles)}", flush=True)
    rev_file.write_text(json.dumps(revs))
    con_file = CACHE / "content.json"
    con = json.loads(con_file.read_text()) if con_file.exists() else {}
    todo = sorted({v for r in revs.values() for v in r.values() if v and str(v) not in con})
    for i in range(0, len(todo), 500):
        con.update(contents(todo[i:i + 500]))
        con_file.write_text(json.dumps(con))
        print(f"content {min(i + 500, len(todo))}/{len(todo)}", flush=True)
    red = cached("redirects", lambda: redirects(titles))
    idx = {t: i for i, t in enumerate(titles)}
    snaps = {}
    for y in YEARS:
        snap = {}
        for t in titles:
            rv = revs[t].get(str(y))
            if not rv or str(rv) not in con or con[str(rv)].lstrip().upper().startswith("#REDIRECT"):
                continue
            text, links = parse(con[str(rv)])
            tgt = {idx[red.get(norm(l), norm(l))] for l in links if red.get(norm(l), norm(l)) in idx}
            tgt.discard(idx[t])
            snap[idx[t]] = dict(text=text, links=sorted(tgt))
        snaps[str(y)] = snap
        print(f"{y}: {len(snap)} pages, {sum(len(v['links']) for v in snap.values())} links", flush=True)
    (HERE / "stream.json").write_text(json.dumps(dict(titles=titles, years=YEARS, snapshots=snaps)))


if __name__ == "__main__":
    sys.exit(main())
