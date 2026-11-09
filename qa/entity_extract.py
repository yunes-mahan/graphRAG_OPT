"""GraphRAG-style entity/relationship extraction from a time-ordered document stream.

Documents: 6,000 papers (titles + abstracts) of the QA subgraph (cs.IR, cs.DB, cs.SI, cs.DL), sampled uniformly
(seed 0) from 2010-2020. For every document gpt-5.6-luna extracts entities (name, type) and relationships
(source, target, short description) with a GraphRAG-like prompt. Entity names are normalised (lower case,
whitespace/punctuation collapsed) and merged across documents, as in GraphRAG's entity resolution by name.
Output: entity_graph/extractions.json (cached per document; resumable).
"""

import concurrent.futures as cf
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wikidata"))
sys.path.insert(0, str(Path(__file__).parent))
import config  # noqa: E402
HERE = config.RESULTS
import qa_plan as qp  # noqa: E402
import qa_run as qr  # noqa: E402

OUT = HERE / "entity_graph"
OUT.mkdir(exist_ok=True)
N_DOCS, Y0, Y1 = 6000, 2010, 2020
TYPES = "method, task, dataset, metric, system, concept, organization"
PROMPT = ("Extract a knowledge graph from the research abstract below, as GraphRAG does when indexing documents.\n"
          f"1. Entities: the important named or technical entities, each with a type from: {TYPES}.\n"
          "2. Relationships: pairs of extracted entities that the text clearly relates, with a short description.\n"
          "Use short canonical names (e.g. 'collaborative filtering', not 'a novel collaborative filtering approach').\n"
          'Return JSON {"entities": [{"name": str, "type": str}], "relationships": '
          '[{"source": str, "target": str, "description": str}]}. At most 12 entities.\n\n')


def norm(name):
    s = re.sub(r"[^a-z0-9+#]+", " ", str(name).lower()).strip()
    return re.sub(r"\s+", " ", s)


def extract(doc_id, title, abstract, cache):
    k = hashlib.sha1(f"extract|{doc_id}".encode()).hexdigest()
    hit = cache.get(k)
    if hit:
        return hit
    for _ in range(3):
        text, usage = qr.call(qr.SUM_MODEL, [{"role": "user", "content": PROMPT + f"Title: {title}\nAbstract: {abstract[:2500]}"}],
                              json_mode=True, max_tokens=1500)
        try:
            j = json.loads(text)
            ents = [dict(name=norm(e["name"]), type=str(e.get("type", "concept")).lower()) for e in j.get("entities", [])
                    if norm(e.get("name", ""))]
            rels = [dict(source=norm(r["source"]), target=norm(r["target"]), description=str(r.get("description", ""))[:200])
                    for r in j.get("relationships", []) if norm(r.get("source", "")) and norm(r.get("target", ""))]
            out = dict(entities=ents, relationships=rels, usage=usage)
            cache.put(k, out)
            return out
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    out = dict(entities=[], relationships=[], usage=usage)
    cache.put(k, out)
    return out


def main():
    nodes = pd.read_csv(HERE / "qa" / "nodes.csv")
    sel = nodes[(nodes.year >= Y0) & (nodes.year <= Y1)]
    sel = sel.sample(n=min(N_DOCS, len(sel)), random_state=0).sort_values("idx")
    texts = qp.load_texts(sel.paper_id.values)
    cache = qr.Cache.__new__(qr.Cache)
    cache.path = OUT / "cache_extract.json"
    cache.data = json.loads(cache.path.read_text()) if cache.path.exists() else {}
    jobs = [(int(r.idx), int(r.year), *texts.get(int(r.paper_id), ("", ""))) for r in sel.itertuples()]
    jobs = [j for j in jobs if j[2] or j[3]]
    print(f"documents: {len(jobs)} ({Y0}-{Y1})", flush=True)
    res = {}
    with cf.ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(extract, d, t, a, cache): (d, y) for d, y, t, a in jobs}
        for i, f in enumerate(cf.as_completed(futs)):
            d, y = futs[f]
            res[d] = dict(year=y, **{k: v for k, v in f.result().items() if k != "usage"})
            if i % 500 == 0:
                print(f"extracted {i}/{len(jobs)}", flush=True)
    tok = [cache.data[k]["usage"] for k in cache.data]
    print(f"tokens in {sum(u['inp'] for u in tok):,} out {sum(u['out'] for u in tok):,}", flush=True)
    (OUT / "extractions.json").write_text(json.dumps(res))
    ents = {e["name"] for r in res.values() for e in r["entities"]}
    rels = {(min(x["source"], x["target"]), max(x["source"], x["target"])) for r in res.values()
            for x in r["relationships"] if x["source"] != x["target"]}
    print(f"entities {len(ents)}, relationships {len(rels)}", flush=True)


if __name__ == "__main__":
    main()
