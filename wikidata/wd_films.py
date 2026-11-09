"""Fetch a typed, time-stamped Wikidata knowledge graph of films (the domain of WikiMovies/MetaQA).

Nodes (typed): films (instance of film, Q11424) first released 2000-2023 with at least MIN_SITELINKS Wikipedia
sitelinks, and the people, companies and genres linked to them.
Edges (typed): film -director (P57)-> person, -cast member (P161)-> person, -screenwriter (P58)-> person,
-genre (P136)-> genre, -production company (P272)-> company.
Time: a film arrives in its (earliest) publication year; every other node arrives with its first film.
Output: wikidata/films.json. One SPARQL query per (year, relation), sequential, with pauses and retries.
"""

import json
import time
from pathlib import Path

import requests

HERE = Path(__file__).parent
CACHE = HERE / "cache_films"
CACHE.mkdir(exist_ok=True)
EP = "https://query.wikidata.org/sparql"
S = requests.Session()
S.headers.update({"User-Agent": "graphrag-maintenance-research/0.1 (academic benchmark)",
                  "Accept": "application/sparql-results+json"})
MIN_SITELINKS = 10
RELS = {"director": "P57", "cast": "P161", "screenwriter": "P58", "genre": "P136", "company": "P272"}


def q(query, name):
    f = CACHE / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    for attempt in range(8):
        time.sleep(1.5)
        try:
            r = S.get(EP, params={"query": query}, timeout=120)
        except requests.RequestException:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code == 200:
            rows = r.json()["results"]["bindings"]
            f.write_text(json.dumps(rows))
            return rows
        ra = r.headers.get("Retry-After")
        time.sleep(int(ra) if ra and ra.isdigit() else 30 * (attempt + 1))
    raise RuntimeError(f"query {name} failed")


def qid(x):
    return x["value"].rsplit("/", 1)[1]


def main():
    films, edges = {}, []
    for y in range(2000, 2024):
        rows = q(f"""SELECT ?f (MIN(YEAR(?d)) AS ?y) WHERE {{ ?f wdt:P31 wd:Q11424 ; wdt:P577 ?d ;
                 wikibase:sitelinks ?sl . FILTER(?sl >= {MIN_SITELINKS}) FILTER(YEAR(?d) = {y}) }} GROUP BY ?f""",
                 f"films_{y}")
        for r in rows:
            films[qid(r["f"])] = min(films.get(qid(r["f"]), 9999), int(r["y"]["value"]))
        print(f"{y}: films so far {len(films)}", flush=True)
    fl = sorted(films)
    for rel, pid in RELS.items():
        for i in range(0, len(fl), 400):
            vals = " ".join("wd:" + f for f in fl[i:i + 400])
            for r in q(f"SELECT ?f ?o WHERE {{ VALUES ?f {{ {vals} }} ?f wdt:{pid} ?o . }}", f"{rel}_{i}"):
                edges.append((qid(r["f"]), rel, qid(r["o"])))
        print(f"{rel}: edges so far {len(edges)}", flush=True)
    (HERE / "films.json").write_text(json.dumps(dict(films=films, edges=edges)))
    print(f"done: films {len(films)}, edges {len(edges)}", flush=True)


if __name__ == "__main__":
    main()
