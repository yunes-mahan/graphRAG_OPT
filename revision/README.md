# VLDB 2027 revision experiments (`revision/`)

Scripts added for the VLDB 2027 EA&B revision. Their cached outputs are in `artifacts/v2/` and are read by
`paper_assets/make_assets.py`, so **every number in the paper regenerates without running anything here**
(`make all` / `make check`, see the top-level README). This folder is only needed to *re-run* the experiments.

## Layout and environment

Prerequisites (derived data that is rebuilt, not shipped): `python benchmark/prepare_data.py` and the links
`token_task/resources/{arxiv,collab}_versions.npz` (both done by `./quick_reproduce.sh`); the entity graph
`python qa/entity_graph_eval.py build` (from the cached extractions); the Wikidata film graph
`python wikidata/wd_films.py` (SPARQL crawl). The Qwen experiments need Ollama (below).

The scripts expect the results tree (`$GCM_RESULTS`, by default `results/`; `artifacts/` has the same layout) as their
parent directory and the repository's code on the Python path:

```bash
export GCM_RESULTS=$PWD/artifacts            # or a copy of it, if you do not want to overwrite cached outputs
export GCM_TASK=$PWD/token_task              # development-stream task (reference tokens, calibration)
mkdir -p $GCM_RESULTS/v2 && cp -r revision/* $GCM_RESULTS/v2/
export PYTHONPATH=$PWD:$PWD/benchmark:$PWD/qa:$PWD/wikidata:$PWD/maintainers
cd $GCM_RESULTS
```

Third-party code is downloaded at pinned commits (not redistributed; the k-core repository has no licence):
`bash v2/fetch_external.sh`. The authors' k-core code needs `graspologic` (numpy < 2); use a separate environment:
`python -m venv v2/.venv_ext && v2/.venv_ext/bin/pip install -r v2/requirements-ext.txt`.
Rust/C++ builds run in containers (`docker.io/library/rust:1.83`, which also provides g++ 12).

## Scripts (paper item → script → output)

| Item | Script | Output (`v2/`) | Runtime (Apple M5 Pro) |
|---|---|---|---|
| A1 hybrid refresh count | `a1_hybrid_refresh.py` | `a1_hybrid_refresh.json` | ~10 min (12 proc.) |
| A3 Q per version | `a3_q_per_version.py` | `a3_q_per_version.json` | ~10 min |
| A5 token ratio | `a5_token_ratio.py` | `a5_token_ratio.json` | seconds |
| A6 global-judge check (cache only) | `a6_global_length.py` | `a6_global_length.json` | seconds |
| A12 hand-typed numbers | `a12_handtyped.py` | `a12_handtyped.json` | ~2 min |
| B1 HIT-Leiden validation | `b1_hitleiden_validate.py <hit-leiden src> <workdir>` + `hitleiden_check/` | `b1_hitleiden_validate.json` | ~10 min (incl. Rust build) |
| B2 authors' k-core (Table 3) | `.venv_ext/bin/python b2_kcore_authors.py` | `b2_kcore_authors.json` | ~40 min |
| B2 k-core full streams | `HIER_ONLY=kcore_authors KCORE_TYPE=RkH HIER_SEEDS=0,1,2 HIER_OUT=v2/hier5/kcore_authors_<g>_RkH.json HIER_GRAPH=<g> .venv_ext/bin/python $REPO/qa/graphrag_hier.py` | `hier5/kcore_authors_*` | 1 h (entity), ~10 h (Wikidata) |
| B2b Sahu's DF Leiden | `b2b_dfleiden.py` (+ `dfleiden_driver/`) | `kgv2_*_df_leiden_sahu.json` | ~5 min |
| B3 cost sensitivity, Table 4 | `b3_flat_sensitivity.py base|priced|analyze` | `b3/`, `b3_flat_sensitivity.json` | ~8 h (4 proc.) |
| B3 cost sensitivity, Table 6 | `b3_extra_sensitivity.py run|analyze` | `b3x/`, `b3_extra_sensitivity.json` | ~1 h |
| B3 cost sensitivity, Table 2 | `b3_hier_sensitivity.py` (uses B6 outputs) | `b3_hier_sensitivity.json` | seconds |
| B5 Pareto view | `b5_pareto.py` (uses B3 base + B2b) | `b5_pareto.json` | ~5 min |
| B6 Table 2 with 5 orders | `graphrag_hier.py` with `HIER_SEEDS=<s> HIER_TOKVARIANTS=1 HIER_OUT=v2/hier5/graphrag_hier_<g>_s<s>.json`, then `b6_hier5.py`, then `HIER_ANOVA_SUFFIX=_5orders python $REPO/qa/hier_anova.py` | `graphrag_hier_*_5orders.json`, `b6_hier5.json`, `hier_anova_5orders.json` | ~1 h |
| C1 entity-graph local questions (Qwen) | `c1_entity_local_qwen.py 150` | `c1_entity_local_qwen.json` | ~4 h |
| C2 open-model replication (Qwen) | `c2_qwen_qa.py 150` | `c2_qwen_qa.json` | ~3 h |

## Re-running the Qwen experiments (C1, C2)

All Qwen outputs are cached (`qa/cache_*_c1qwen.json`, `qa/cache_*_c2qwen.json`); with the caches present the scripts
only recompute statistics. To regenerate from scratch, delete those caches and run the model locally:

```bash
ollama pull qwen3:14b                 # digest bdbd181c33f2, Q4_K_M, 14.8B (Ollama 0.40.15 used)
OLLAMA_NUM_PARALLEL=1 ollama serve &  # one request at a time (deterministic, and the machine stays responsive)
C2_WORKERS=1 python v2/c2_qwen_qa.py 150
C1_WORKERS=1 python v2/c1_entity_local_qwen.py 150
```

Settings (`qwen_backend.py`): temperature 0, seed 0, thinking off, num_ctx 16,384, output limit = the paper
pipeline's limit per call type (at most 1,500). Every result file records these settings.
