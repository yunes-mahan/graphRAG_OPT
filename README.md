# Maintaining GraphRAG Community Summaries on Evolving Knowledge Graphs

Code, benchmark and reproducibility artifacts for the paper *"Maintaining GraphRAG Community Summaries on
Evolving Knowledge Graphs [Experiment, Analysis & Benchmark]"* (Yunes Mahan, Machine Learning Reply).

GraphRAG indexes a knowledge graph by clustering it into communities and storing one LLM-written report per
community. When the graph changes, a system must decide whether to re-cluster and which reports to rewrite. This
repository benchmarks these maintenance choices on seven evolving graphs, including an LLM-extracted entity graph
clustered with Microsoft GraphRAG's hierarchical Leiden settings. It measures **report-rewrite cost** (LLM tokens),
**coverage** (entities in a reported community), clustering quality, and the effect on **answers** to local and global
questions.

## Contents

| Folder | What |
|---|---|
| `maintainers/` | Online community maintainers with persistent IDs: recompute / warm-start / frontier Leiden, lazy attach, local repair, hybrid with drift-triggered refresh, hybrid with embedding-based stress (`semantic_hybrid.py`, raw / mean-centred via `SEM_CENTER=1`) |
| `benchmark/` | Stream construction (ogbn-arxiv, ogbl-collab), evaluator (modularity, regeneration at several Jaccard tolerances, per-community cost), experiment runner, token-level cost model |
| `neo4j/` | Replays a stream into Neo4j 5 + GDS, maintains a `communityId` property, compares with seeded GDS Louvain and recomputed GDS Leiden |
| `qa/` | Downstream experiment: LLM community summaries, generated topic questions with key points, retrieval over summaries, key-point recall judged by an LLM |
| `paper_assets/` | Reference verification (Crossref / arXiv) and BibTeX generation |

## Quick start

```bash
pip install -r requirements.txt
export GCM_DATA=$PWD/data GCM_RESULTS=$PWD/results
python benchmark/prepare_data.py                        # downloads OGB data (~1 GB) and paper texts
python benchmark/run_suite.py arxiv 2015 0 lazy,local_repair,hybrid,frontier_leiden
```

This replays four maintainers on one arXiv stream. To check every number of the paper instead, run
`./quick_reproduce.sh` (no API key needed).

### Full benchmark

```bash
M=lazy,local_repair,hybrid,place_drift,local_periodic5,local_periodic3,frontier_leiden,warmstart_leiden,recompute_leiden
python benchmark/run_suite.py arxiv 2015 0,1,2,3,4,5,6,7,8,9 $M
python benchmark/run_suite.py collab 2012 0,1,2,3,4,5,6,7,8,9 $M
python benchmark/run_suite.py arxiv 2017 0,1,2 $M
python benchmark/run_suite.py collab 2010 0,1,2 $M
python benchmark/cost_model.py
```

### Neo4j

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/<password> \
  -e 'NEO4J_PLUGINS=["graph-data-science"]' -e NEO4J_server_memory_heap_max__size=3G neo4j:5-community
export NEO4J_PASSWORD=<password>
python neo4j/neo4j_replay.py x all arxiv 0
```

### QA experiment (uses the OpenAI API, about USD 5)

```bash
export OPENAI_API_KEY=...
python qa/qa_plan.py      # no LLM calls: builds the subgraph stream and the summary plan
python qa/qa_run.py       # summaries, questions, answers, judging (cached, resumable)
```

## Maintainer interface

```python
def make_maintainer():
    return Maintainer()

class Maintainer:
    def initialize(self, nodes, edges, features, years): ...   # -> community id per node
    def update(self, new_nodes, new_edges, new_features, new_years, all_nodes): ...  # -> ids for all_nodes
```

Maintainers are online: they only see the current and past versions. Keeping an ID means re-using
that community's summary; a summary is regenerated when the ID is new or the member set changed
beyond the Jaccard tolerance τ.

## Notes

- The local-repair maintainer was selected on the development order and checked to use only past
  versions. See the paper for its limitations
  (development-graph overfitting, fragmentation under per-summary cost).
- Data: ogbn-arxiv and ogbl-collab from the Open Graph Benchmark (Hu et al., 2020) under their
  respective licences; paper texts from the OGB arXiv title/abstract dump.

## Licence

Apache-2.0.

## Evolving-text stream (Wikipedia)
`wiki/wiki_crawl.py` downloads AI/ML articles (links + lead sections at 1 January 2017-2025) politely from the MediaWiki API (cap with `MAX_PAGES`); `wiki/wiki_analysis.py` builds communities, yearly summaries and staleness labels; `wiki/wiki_extra.py` reports deletions, splits and cold-start placement; `neo4j/versioned_demo.py` loads everything into Neo4j as one time-versioned graph (`valid_from`/`valid_to`). Wikipedia text is CC BY-SA and is not redistributed here.

## Token-level score on two graphs
`token_task/` contains a self-contained evaluator for the token-level score on the development orders of arXiv-15 and collab-12 (lambda in `calibration.json`); `benchmark/kgv2_calibrate.py calibrate` recomputes the reference tokens and baseline scores, and `benchmark/kgv2_calibrate.py heldout PROGRAM NAME` scores any maintainer on the 24 held-out orders and streams.

## Summary-aware methods
`maintainers/priced_warmstart.py` and `maintainers/priced_frontier.py` implement summary-priced rollback on warm-started / frontier Leiden (`benchmark/kgv2_dev.py`, `benchmark/kgv2_lambda.py`, `wiki/wiki_priced.py`); `qa/summary_reuse.py` implements topic-anchored (identifier-free) summary reuse and `qa/qa_pipeline.py` evaluates the combined pipeline end to end.

## Reproducing the paper

**Every table, figure and number** in the paper is generated from the files in `artifacts/`
(benchmark results and cached LLM outputs). `./quick_reproduce.sh` checks all of them automatically:

```bash
pip install -r requirements.txt
cd paper_assets
PAPER_RESULTS=../artifacts python make_assets.py   # writes tables/*.tex, figures/*.pdf, numbers.tex
python make_overview.py                             # Fig. 1
```

To **re-run the experiments** from scratch (no LLM calls needed for the structural benchmark; LLM outputs are
cached in `artifacts/qa/cache_*.json`, so the QA experiments re-run offline unless the cache is removed):

| Paper result | Command |
|---|---|
| Benchmark streams (data) | `python benchmark/prepare_data.py` (downloads ogbn-arxiv / ogbl-collab) |
| Main / generalisation / metric tables | `python benchmark/run_suite.py arxiv 2015 0,1,...,9 recompute_leiden,warmstart_leiden,frontier_leiden,lazy,local_repair,hybrid` (and collab 2012, arxiv 2017, collab 2010) |
| Token-level score, held-out table | `python benchmark/kgv2_calibrate.py calibrate`, then `python benchmark/kgv2_calibrate.py heldout - METHOD` |
| Summary-priced rollback | maintainers `priced_warm`, `priced_frontier` in the commands above; `benchmark/kgv2_lambda.py 0.1,0.5,1.0` |
| QA experiments | `qa/qa_plan.py`, `qa/qa_run.py`, `qa/qa_global.py`, `qa/proto_rag.py`, `qa/summary_reuse.py run`, `qa/qa_pipeline.py topic:0.95` (set `OPENAI_API_KEY` only if the cache is removed) |
| Scalability (ogbl-citation2, up to 2.9M nodes / 30M edges) | `python benchmark/scale_citation2.py prepare`, then `benchmark/scale_run.sh` |
| Microsoft GraphRAG update baseline | maintainer `graphrag_delta` in the commands above |
| LLM-extracted entity graph | `qa/entity_extract.py` (LLM), `qa/entity_graph_eval.py build` and `run`, `qa/qa_entity.py run` (cached reports in `artifacts/entity_graph/`) |
| Wikipedia stream | `wiki/wiki_crawl.py`, `wiki/wiki_analysis.py`, `wiki/wiki_extra.py`, `wiki/wiki_priced.py` |
| Wikidata film KG | `wikidata/wd_films.py`, `wikidata/wd_eval.py` |
| GraphRAG hierarchical clustering study (re-index, matching, warm start, GraphRAG update, Zep-style attach) | `python qa/graphrag_hier.py` (entity graph) and `HIER_GRAPH=wikidata python qa/graphrag_hier.py`, then `python qa/hier_anova.py` |
| Instability of GraphRAG's hierarchical clustering (edge deletion, last update, seed) | `python qa/noise_hier.py` |
| Flat Leiden instability | `python qa/noise_floor.py` |
| End-to-end global QA on the hierarchical stores (top level) | `python qa/hier_qa.py run` (cached) |
| Fresh reports of each maintainer's own partition (control) | `python qa/own_fresh.py run` (cached) |
| Score validation and equivalence tests (TOST) | `python qa/validate_score.py` |
| Topic-fingerprint control (random replacement) | `python qa/cosine_baseline.py` |
| 300 local questions | `python qa/qa_more.py` (cached) |
| Second judge (Qwen3-14B, local via Ollama, absolute 1-10 ratings) | `ollama pull qwen3:14b`, then `python qa/qwen_judge.py entity`, `global`, `local` |
| Targeted staleness and coverage probes | `python qa/probes.py run` (cached); local Qwen pipeline incl. tau=0.9: `python qa/qwen_probes.py 0.0` and `0.9` |
| Deletion leak test and lineage rule | `python qa/deletion_leak.py` |
| Sensitivity to GraphRAG's max cluster size | `HIER_MAXC=25 python qa/graphrag_hier.py` (also 50, `HIER_GRAPH=wikidata`, `qa/noise_hier.py`), then `python qa/sens_summary.py` |
| Reproducibility vs stability (consensus, k-core, warm start) | `python qa/stability_variants.py`; 10/50/100 consensus runs with timing: `python qa/consensus_runs.py`; consensus + id matching over the update streams: `HIER_ONLY=consensus python qa/graphrag_hier.py` (and `HIER_GRAPH=wikidata`) |
| Refresh scheduling under a token budget | `python qa/refresh_sched.py` |
| Dilution check for the staleness probe | `python qa/dilution_check.py` |
| Token model vs measured tokens | `python qa/tokmodel_check.py` |
| Second answering model (Qwen3-14B, local) | `python qa/qwen_reader.py answer`, then `judge` (cached) |
| Neo4j harness / time-versioned graph | `neo4j/neo4j_replay.py`, `neo4j/versioned_demo.py` (set `NEO4J_PASSWORD`) |

Hardware used: one Apple M5 Pro (18 cores, 24 GB RAM); single-process Python 3.12, python-igraph 1.0,
leidenalg 0.12; Neo4j 5 Community + GDS 2.13.

## Licence
Code: Apache-2.0 (see `LICENSE`). Derived data: ogbn-arxiv / ogbl-collab / ogbl-citation2 follow the OGB licences
(ODC-BY); Wikidata content is CC0; Wikipedia text is CC BY-SA and is re-crawled by `wiki/wiki_crawl.py`.
