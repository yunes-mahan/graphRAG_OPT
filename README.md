# Maintaining GraphRAG Community Summaries on Evolving Knowledge Graphs

Code, benchmark and reproducibility artifacts for the paper *"Maintaining GraphRAG Community Summaries on
Evolving Knowledge Graphs [Experiment, Analysis & Benchmark]"* (Yunes Mahan, Machine Learning Reply).

GraphRAG indexes a knowledge graph by clustering it into communities and storing one LLM-written report per
community. When the graph changes, a system must decide whether to re-cluster and which reports to rewrite. This
repository benchmarks these maintenance choices on eight update streams over five graph sources (plus ogbl-citation2
for scalability), including an LLM-extracted entity graph
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

**Every table, figure and number** in the paper is generated from the files in `artifacts/` (benchmark results and
cached LLM outputs); no LLM call or API key is needed. Install (Python 3.12) and use one command per item:

```bash
pip install -r requirements.txt
make all          # every table (paper_assets/tables), figure (paper_assets/figures) and quoted number (numbers.tex)
make check        # regenerate and compare EVERY number with the paper (artifacts/expected_*); exit 1 on any mismatch
make table4       # one table:  table1 ... table9
make fig3         # one figure: fig1 ... fig5
make reproduce    # quick_reproduce.sh: the check above + re-running two maintainers from the raw OGB data
```

Expected runtime on the hardware below: `make all` / `make check` 2-5 minutes; `make reproduce` 10-15 minutes
(downloads about 1 GB of OGB data). Table/figure numbers follow the VLDB 2027 submission. Table 1 is typed in the
paper; its numbers are recomputed from the data by `revision/a12_handtyped.py` (`make table1`).

The patch applied to the authors' k-core code (one line: an uninitialised print-only counter) is
`revision/kcore_patch.diff`. The experiments added for the VLDB 2027 revision (k-core and dynamic-frontier Leiden with the authors' code, HIT-Leiden
validation, cost-model sensitivity, Pareto view, 5 document orders, open-model replication with a local Qwen3-14B) are
in `revision/`; see `revision/README.md` for how to re-run them, including the Qwen experiments.

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
| Reproducibility vs stability (consensus, k-core, warm start) | `python qa/stability_variants.py`; 10/50/100 consensus runs with timing: `python qa/consensus_runs.py`; consensus + id matching over the update streams: `HIER_ONLY=consensus python qa/graphrag_hier.py` (and `HIER_GRAPH=wikidata`; `HIER_CONS_MODE=soft|leafup` for the hierarchy-preserving variants); hierarchy structure: `python qa/consensus_fix.py entity|wikidata`, `python qa/table3_structure.py` |
| Refresh scheduling under a token budget | `python qa/refresh_sched.py` |
| Dilution check for the staleness probe | `python qa/dilution_check.py` |
| Token model vs measured tokens | `python qa/tokmodel_check.py` |
| Second answering model (Qwen3-14B, local) | `python qa/qwen_reader.py answer`, then `judge` (cached) |
| Neo4j harness / time-versioned graph | `neo4j/neo4j_replay.py`, `neo4j/versioned_demo.py` (set `NEO4J_PASSWORD`) |

Hardware used: one Apple M5 Pro (18 cores, 24 GB RAM); single-process Python 3.12, python-igraph 1.0,
leidenalg 0.12; Neo4j 5 Community + GDS 2.13.

### Pinned versions

- Python packages: `requirements.txt` (all pinned; e.g. leidenalg 0.12.0, python-igraph 1.0.0, graspologic-native 1.3.1,
  numpy 2.5.4, scipy 1.18.1). Third-party code of the revision that needs `graspologic` (numpy < 2) runs in a separate
  environment: `revision/requirements-ext.txt` (graspologic 3.4.4).
- Microsoft GraphRAG is **not** imported. Its clustering is reproduced with graspologic-native 1.3.1 exactly as in
  GraphRAG's `cluster_graph` (hierarchical Leiden, `max_cluster_size=10`, largest connected component, seed
  `0xDEADBEEF`), and its incremental update is re-implemented from its source code, versions 3.2.0 and 3.3.0 (identical
  in this respect). Flat Leiden uses leidenalg 0.12.0 with seed 0.
- LLMs: gpt-5.6-luna / gpt-5.6-terra (OpenAI Chat Completions, accessed 6-10 Nov 2026, API default sampling; all
  outputs cached in `artifacts/`). Local model: Ollama 0.40.15, `qwen3:14b` (digest bdbd181c33f2, Q4_K_M), temperature
  0, seed 0, thinking off, context 16,384 (`revision/qwen_backend.py`).
- Qwen model file: Ollama blob `sha256-a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e` (manifest digest
  bdbd181c33f2); V12 global runs use context 32,768, V11/C3 local runs 16,384 (`revision/v12_qwen/qwen12.py`,
  `revision/qwen_backend.py`). Ollama is run with `OLLAMA_NUM_PARALLEL=1` (one request at a time).
- Third-party code is fetched at pinned commits by `revision/fetch_external.sh` (Kcore-GraphRAG ae7b2ef,
  leiden-communities-openmp-dynamic 28c304f, unofficial hit-leiden 9247424); containers: `docker.io/library/rust:1.83`.

## Licence
Code: Apache-2.0 (see `LICENSE`). Derived data: ogbn-arxiv / ogbl-collab / ogbl-citation2 follow the OGB licences
(ODC-BY); Wikidata content is CC0; Wikipedia text is CC BY-SA and is re-crawled by `wiki/wiki_crawl.py`.

## VLDB 2027 revision, round 2 (V12)

New experiments (scripts in `revision/v12/` and `revision/v12_qwen/`, outputs in `artifacts/v12*/`): batch-size sweep
(`make batch_sweep`), rewrite share by level/size and greedy vs Hungarian matching (`make matching`), variance shares
for restricted designs (`make eta_variants`), total tokens per update (`make tco`), coverage of Table 6
(`make coverage_t6`), GraphRAG update with `use_lcc` off (`make gu_lcc`), coarse levels on global answers with Qwen
(`make coarse_answers`; judge uninformative, see paper), 300 local questions for the never-refreshed stores with Qwen
(`make never300`), TOST-margin provenance (`make margin`) and the non-inferiority tests at a stricter margin of
±0.035 (`make margin035`, `revision/v12/d2_tost_035.py`), rollback factor per graph (`make rollback_factor`).
The A/A test of the pairwise judge was stopped (the local judge prefers the second answer almost always); its partial
caches are in `artifacts/v12_qwen/`. To re-run: copy `revision/v12` and `revision/v12_qwen` into `$GCM_RESULTS/` next
to `v2/` (see `revision/README.md`), then run the script named in the target.
