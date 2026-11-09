"""Versioned (time-travel) GraphRAG index in Neo4j for the Wikipedia ML/AI stream.

Instead of copying the graph per version, every element carries a validity interval [valid_from, valid_to):
  (:Article {id, title, valid_from, valid_to})
  (:Article)-[:LINKS {valid_from, valid_to}]->(:Article)
  (:Article)-[:IN_COMMUNITY {valid_from, valid_to}]->(:Community {cid})
  (:Community)-[:HAS_SUMMARY {valid_from, valid_to}]->(:Summary {text})    (a new Summary only when it changed)
valid_to = 9999 means "still valid". Demonstrated and timed:
  1. storage: interval relationships vs the sum of per-snapshot copies
  2. time travel: the graph, a community and its summary "as of" any year
  3. history: when did an article change community, which summaries changed between two years
  4. GDS on a past version: project the 2019 graph with a Cypher filter and run Leiden on it
Inputs: analysis/wiki/stream.json and communities.json (written by wiki_analysis.py).
usage: NEO4J_PASSWORD=... python versioned_demo.py
"""

import json
import os
import time
from pathlib import Path

from neo4j import GraphDatabase

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
W = config.RESULTS / "wiki"
URI, AUTH = "bolt://localhost:7687", ("neo4j", os.environ["NEO4J_PASSWORD"])
OPEN = 9999


def intervals(present_years, years):
    """[(from, to)] maximal runs of consecutive years in which something is present."""
    out, start = [], None
    for i, y in enumerate(years):
        if y in present_years and start is None:
            start = y
        if y not in present_years and start is not None:
            out.append((start, y))
            start = None
    if start is not None:
        out.append((start, OPEN))
    return out


def chunks(rows, n=5000):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def main():
    st = json.loads((W / "stream.json").read_text())
    cm = json.loads((W / "communities.json").read_text())
    years = [int(y) for y in st["years"]]
    titles = st["titles"]
    art_years, link_years, memb_years = {}, {}, {}
    for y in years:
        for a, v in st["snapshots"][str(y)].items():
            art_years.setdefault(int(a), set()).add(y)
            for b in v["links"]:
                link_years.setdefault((int(a), int(b)), set()).add(y)
        for a, c in cm["labels"][str(y)].items():
            memb_years.setdefault((int(a), int(c)), set()).add(y)
    arts = [dict(id=a, title=titles[a], f=f, t=t) for a, ys in art_years.items() for f, t in intervals(ys, years)]
    links = [dict(a=a, b=b, f=f, t=t) for (a, b), ys in link_years.items() for f, t in intervals(ys, years)]
    memb = [dict(a=a, c=c, f=f, t=t) for (a, c), ys in memb_years.items() for f, t in intervals(ys, years)]
    # summaries: a new Summary node only when the text changed for that community
    sums, last = [], {}
    for key in sorted(cm["summaries"], key=lambda k: (int(k.split("|")[0]), int(k.split("|")[1]))):
        y, c = map(int, key.split("|"))
        txt = cm["summaries"][key]
        if last.get(c, (None, None))[1] != txt:
            if c in last:
                sums[last[c][0]]["t"] = y
            last[c] = (len(sums), txt)
            sums.append(dict(c=c, text=txt, f=y, t=OPEN))
    copies = dict(articles=sum(len(st["snapshots"][str(y)]) for y in years),
                  links=sum(len(v["links"]) for y in years for v in st["snapshots"][str(y)].values()),
                  memberships=sum(len(cm["labels"][str(y)]) for y in years), summaries=len(cm["summaries"]))
    stored = dict(articles=len(arts), links=len(links), memberships=len(memb), summaries=len(sums))
    print("per-snapshot copies:", copies)
    print("interval-versioned :", stored)
    res = dict(copies=copies, stored=stored, timings={})
    drv = GraphDatabase.driver(URI, auth=AUTH)
    with drv.session() as s:
        s.run("MATCH (n) WHERE n:Article OR n:Community OR n:Summary DETACH DELETE n").consume()
        s.run("CREATE INDEX art_id IF NOT EXISTS FOR (a:Article) ON (a.id)").consume()
        s.run("CREATE INDEX com_id IF NOT EXISTS FOR (c:Community) ON (c.cid)").consume()
        t0 = time.time()
        for part in chunks([dict(id=a, title=titles[a]) for a in art_years]):
            s.run("UNWIND $rows AS r CREATE (:Article {id: r.id, title: r.title})", rows=part).consume()
        for part in chunks(arts):
            s.run("UNWIND $rows AS r MATCH (a:Article {id: r.id}) SET a.valid_from = coalesce(a.valid_from, r.f), "
                  "a.valid_to = r.t", rows=part).consume()
        for part in chunks(links):
            s.run("UNWIND $rows AS r MATCH (a:Article {id: r.a}), (b:Article {id: r.b}) "
                  "CREATE (a)-[:LINKS {valid_from: r.f, valid_to: r.t}]->(b)", rows=part).consume()
        s.run("UNWIND $cs AS c CREATE (:Community {cid: c})", cs=sorted({m["c"] for m in memb})).consume()
        for part in chunks(memb):
            s.run("UNWIND $rows AS r MATCH (a:Article {id: r.a}), (c:Community {cid: r.c}) "
                  "CREATE (a)-[:IN_COMMUNITY {valid_from: r.f, valid_to: r.t}]->(c)", rows=part).consume()
        for part in chunks(sums, 500):
            s.run("UNWIND $rows AS r MATCH (c:Community {cid: r.c}) "
                  "CREATE (c)-[:HAS_SUMMARY {valid_from: r.f, valid_to: r.t}]->(:Summary {text: r.text})",
                  rows=part).consume()
        res["timings"]["load_s"] = time.time() - t0

        def timed(name, q, **kw):
            t = time.time()
            out = s.run(q, **kw).data()
            res["timings"][name + "_ms"] = 1000 * (time.time() - t)
            return out

        y_mid = years[len(years) // 2]
        res["as_of"] = timed("graph_as_of", "MATCH (a:Article)-[l:LINKS]->(b:Article) WHERE l.valid_from <= $y < "
                             "l.valid_to RETURN count(l) AS links, count(DISTINCT a) AS articles", y=y_mid)
        snap_links = sum(len(v["links"]) for v in st["snapshots"][str(y_mid)].values())
        res["as_of_check"] = dict(year=y_mid, neo4j=res["as_of"][0]["links"], stream=snap_links)
        top = timed("most_moved", "MATCH (a:Article)-[m:IN_COMMUNITY]->() WITH a, count(m) AS n "
                    "RETURN a.title AS title, n ORDER BY n DESC LIMIT 1")[0]
        res["history_example"] = timed("community_history", "MATCH (a:Article {title: $t})-[m:IN_COMMUNITY]->(c) "
                                       "RETURN c.cid AS cid, m.valid_from AS from, m.valid_to AS to "
                                       "ORDER BY m.valid_from", t=top["title"])
        res["history_title"] = top["title"]
        res["changed_summaries"] = timed("summaries_changed", "MATCH (c:Community)-[h:HAS_SUMMARY]->() WHERE "
                                         "$y1 < h.valid_from <= $y2 RETURN count(DISTINCT c) AS n",
                                         y1=years[1], y2=years[-1])[0]["n"]
        res["summary_as_of"] = timed("summary_as_of", "MATCH (a:Article {title: $t})-[m:IN_COMMUNITY]->(c)"
                                     "-[h:HAS_SUMMARY]->(s) WHERE m.valid_from <= $y < m.valid_to AND "
                                     "h.valid_from <= $y < h.valid_to RETURN c.cid AS cid, left(s.text, 160) AS summary",
                                     t=top["title"], y=y_mid)
        s.run("CALL gds.graph.drop('past', false)").consume()
        t = time.time()
        g = s.run("MATCH (a:Article)-[l:LINKS]->(b:Article) WHERE l.valid_from <= $y < l.valid_to "
                  "WITH gds.graph.project('past', a, b, {}, {undirectedRelationshipTypes: ['*']}) AS g "
                  "RETURN g.nodeCount AS nodes, g.relationshipCount AS rels", y=y_mid).single()
        lei = s.run("CALL gds.leiden.stats('past', {randomSeed: 1, concurrency: 1}) "
                    "YIELD communityCount, modularity RETURN communityCount, modularity").single()
        res["gds_past"] = dict(year=y_mid, nodes=g["nodes"], rels=g["rels"], communities=lei["communityCount"],
                               modularity=lei["modularity"], seconds=time.time() - t)
        s.run("CALL gds.graph.drop('past', false)").consume()
    drv.close()
    print(json.dumps(res, indent=1, default=str))
    (config.RESULTS / "versioned_results.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
