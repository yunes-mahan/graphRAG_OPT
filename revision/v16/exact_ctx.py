"""V16/W2+W3: report input tokens in GraphRAG's community-report context format, counted with GraphRAG's default
tokenizer (tiktoken o200k_base; graphrag 3.2.0/3.3.0 config/defaults.py ENCODING_MODEL), for the LLM entity graph.

GraphRAG (index/operations/summarize_communities/graph_context/sort_context.py) gives the report LLM a CSV context:
  "-----Entities-----" table (human_readable_id, title, description, degree) of the community's entities and
  "-----Relationships-----" table (human_readable_id, source, target, description, degree) of the relationships with
  both endpoints in the community, joined by blank lines, inserted into the community-report prompt
  (prompts/index/community_report.py).  If the context exceeds community_reports.max_input_length (8,000 tokens,
  config/defaults.py), build_mixed_context replaces sub-community contexts by sub-community REPORTS.
Our extraction has no entity descriptions and one description per relationship, so this is GraphRAG's format and
limit applied to OUR data, not a re-run of GraphRAG (no LLM-summarised descriptions).  Token counts are summed per
CSV row (row strings tokenised separately; check_rowsum() measures the error against tokenising whole contexts).
Output tokens are not measured here: OUT = mean output tokens of the 835 cached hierarchical entity-graph reports.
"""
import csv
import io
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(HERE / "tiktoken_cache"))
LIMIT = 8000
PROMPT_FILE = HERE / "community_report_prompt_3.3.0.txt"   # copied verbatim from graphrag 3.3.0 (MIT licence)


def _csv_row(vals):
    b = io.StringIO()
    csv.writer(b, lineterminator="\n").writerow(vals)      # pandas to_csv quoting (minimal) is the csv default
    return b.getvalue()


class Ctx:
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("o200k_base")
        g = json.loads((A / "entity_graph" / "graph.json").read_text())
        self.names, E, self.desc = g["entities"], np.array(g["edges"]), g["edge_desc"]
        deg = np.bincount(E.ravel(), minlength=len(self.names))
        n = self.enc.encode
        self.erow = np.array([len(n(_csv_row([i, nm, "", int(deg[i])]))) for i, nm in enumerate(self.names)])
        self.edrow = np.array([len(n(_csv_row([j, self.names[a], self.names[b], self.desc[j], int(deg[a] + deg[b])])))
                               for j, (a, b) in enumerate(E)])
        self.hdr = (len(n("-----Entities-----\n" + _csv_row(["human_readable_id", "title", "description", "degree"])))
                    + len(n("\n\n-----Relationships-----\n" + _csv_row(["human_readable_id", "source", "target", "description", "degree"]))))
        tpl = PROMPT_FILE.read_text()
        self.prompt = len(n(tpl.replace("{input_text}", "")))
        hr = json.loads((A / "entity_graph" / "cache_hreports.json").read_text())
        self.out = float(np.mean([v["usage"]["out"] for v in hr.values() if isinstance(v, dict) and "usage" in v]))

    def context(self, members, adj):
        """context tokens of a community; adj: node -> list of (neighbour, edge index) of the present graph."""
        e = int(self.erow[list(members)].sum())
        r = sum(int(self.edrow[j]) for v in members for u, j in adj.get(v, ()) if v < u and u in members)
        return self.hdr + e + r

    def report_tokens(self, ctx):
        return self.prompt + min(ctx, LIMIT) + self.out


def check_rowsum(n=200, seed=0):
    """relative error of the per-row token sum against tokenising whole contexts (random communities of the final graph)."""
    c = Ctx()
    g = json.loads((A / "entity_graph" / "graph.json").read_text())
    E = g["edges"]
    adj = {}
    for j, (a, b) in enumerate(E):
        adj.setdefault(a, []).append((b, j))
        adj.setdefault(b, []).append((a, j))
    rng = np.random.default_rng(seed)
    errs = []
    for v in rng.choice(len(c.names), n, replace=False):
        mem = {int(v)} | {u for u, _ in adj.get(int(v), [])}
        ents = "".join(_csv_row([i, c.names[i], "", 0]) for i in sorted(mem))
        rels = "".join(_csv_row([j, c.names[a], c.names[b], c.desc[j], 0]) for j, (a, b) in enumerate(E) if a in mem and b in mem)
        whole = len(c.enc.encode("-----Entities-----\nhuman_readable_id,title,description,degree\n" + ents +
                                 "\n\n-----Relationships-----\nhuman_readable_id,source,target,description,degree\n" + rels))
        rs = (len(c.enc.encode("-----Entities-----\nhuman_readable_id,title,description,degree\n")) +
              sum(len(c.enc.encode(_csv_row([i, c.names[i], "", 0]))) for i in sorted(mem)) +
              len(c.enc.encode("\n\n-----Relationships-----\nhuman_readable_id,source,target,description,degree\n")) +
              sum(len(c.enc.encode(_csv_row([j, c.names[a], c.names[b], c.desc[j], 0]))) for j, (a, b) in enumerate(E) if a in mem and b in mem))
        errs.append((rs - whole) / whole)
    return dict(mean_rel_err=float(np.mean(errs)), max_abs_rel_err=float(np.max(np.abs(errs))), n=n)


if __name__ == "__main__":
    c = Ctx()
    print(dict(prompt=c.prompt, out=c.out, header=c.hdr, mean_entity_row=float(c.erow.mean()), mean_rel_row=float(c.edrow.mean())))
    r = check_rowsum(100)
    print(r)
    (HERE / "w3_rowsum_check.json").write_text(json.dumps(r, indent=1))


def export(path=HERE / "w3_rows.npz"):
    """per-row token counts for graphrag_hier.py (HIER_EXACT=1), which runs without tiktoken."""
    c = Ctx()
    np.savez(path, erow=c.erow, edrow=c.edrow, hdr=c.hdr, prompt=c.prompt, out=c.out, limit=LIMIT)
    return path
