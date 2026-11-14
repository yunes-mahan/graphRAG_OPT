# One command per table / figure of the paper. Everything is regenerated from the cached results in artifacts/
# (no LLM calls, no API key, a few minutes on a laptop). Override the results tree with RES=/path/to/results.
#   make all        regenerate every table, figure and number (paper_assets/{tables,figures,numbers.tex})
#   make check      regenerate and compare EVERY number with the paper's (artifacts/expected_*)
#   make table4     regenerate and show Table 4 (same for table1..table9, fig1..fig4)
#   make reproduce  quick end-to-end check incl. re-running two maintainers from the raw OGB data (quick_reproduce.sh)
PY  ?= python
RES ?= $(CURDIR)/artifacts
A    = paper_assets

.PHONY: all assets check reproduce numbers table1 table2 table3 table4 table5 table6 table7 table8 table9 \
        fig1 fig2 fig3 fig4 fig_lambda batch_sweep matching eta_variants tco coverage_t6 gu_lcc rollback_factor margin margin035 \
        lcc kcore_levels batch_newcov wiki_all extra_rows reuse_checks \
        coarse_answers never300 costsens

all assets:
	cd $(A) && PAPER_RESULTS=$(RES) $(PY) make_assets.py > make_assets.log && $(PY) make_overview.py > /dev/null
	@echo "tables -> $(A)/tables/, figures -> $(A)/figures/, numbers quoted in the text -> $(A)/numbers.tex"

check:
	PAPER_RESULTS=$(RES) $(PY) $(A)/check_numbers.py

reproduce:
	./quick_reproduce.sh

numbers: assets
	@cat $(A)/numbers.tex

# Table 1 (benchmark graphs) is typed in the paper; its numbers are recomputed from the data by revision/a12_handtyped.py
table1:
	@cat $(RES)/v2/a12_handtyped.json
table2: assets ; @cat $(A)/tables/hier.tex            # GraphRAG hierarchy under maintenance (5 orders)
table3: assets ; @cat $(A)/tables/stability.tex       # reproducibility vs stability (consensus, k-core, warm start)
table4: assets ; @cat $(A)/tables/batch.tex           # batch-size sweep
table5: assets ; @cat $(A)/tables/tokenmetric.tex     # token-level score of the flat maintainers (+ coverage)
table6: assets ; @cat $(A)/tables/qa.tex              # local questions
table7: assets ; @cat $(A)/tables/extra.tex           # Wikidata, entity graph, Wikipedia, ogbl-citation2 (+ coverage)
table8: assets ; @cat $(A)/tables/reuse.tex           # report reuse rules
table9: assets                                       # key findings: text table, all numbers are macros
	@grep -E "Hier(Wd)?Split|HierEtaTau|HierCov|NhDrop|BfRollFrom|MoreN|ProbeStaleLoss|LocVec|LocSums|Del(Gu|Zep)Five|PrHeldGain|CosRandSeven" $(A)/numbers.tex

fig1:
	cd $(A) && $(PY) make_overview.py && echo "$(A)/figures/overview.pdf"
fig2: assets ; @echo "$(A)/figures/noise.pdf"
fig3: assets ; @echo "$(A)/figures/pareto_heldout.pdf"
fig4: assets ; @echo "$(A)/figures/qa.pdf"
# figures/tables kept in the artifact only (removed from the paper for space): held-out ranking over lambda, cost sensitivity
fig_lambda: assets ; @echo "$(A)/figures/lambda_heldout.pdf"

# ---- VLDB revision round 2 (V12): show the cached result of each new experiment (re-run: see README, V12 section)
V12 = $(RES)/v12
batch_sweep: assets ; @cat $(A)/tables/batch.tex; cat $(V12)/b1_summary.json
matching:        ; @cat $(V12)/b2_summary.json
eta_variants:    ; @cat $(V12)/b3_eta_variants.json
tco:             ; @cat $(V12)/b4_tco.json
coverage_t6:     ; @cat $(V12)/b5_coverage_t6.json
gu_lcc:          ; @ls $(V12)/b7/
rollback_factor: ; @cat $(V12)/a5_rollback_factor.json
margin:          ; @cat $(V12)/d2_margin_provenance.json
margin035:       ; @$(PY) -c "import json;d=json.load(open('$(V12)/d2_tost_035.json'));[print(k,m,v['n_noninferior'],'/',v['n_stores'],'non-inferior, lowest bound',round(v['min_lower'],3)) for k in d for m,v in d[k].items()]"
coarse_answers:  ; @$(PY) -c "import json;d=json.load(open('$(RES)/v12_qwen/c1_coarse_global.json'));d.pop('rows');print(json.dumps(d,indent=1))"
never300:        ; @$(PY) -c "import json;d=json.load(open('$(RES)/v12_qwen/c3_never_300.json'));d.pop('per_question');print(json.dumps(d,indent=1))"
costsens: assets ; @cat $(A)/tables/costsens.tex

# ---- V13 table fixes (T1-T5): cached results; re-run with the script named in revision/v12/
lcc:          ; @cat $(V12)/t1_lcc.json                 # Table 2: largest connected component vs all entities
kcore_levels: ; @cat $(V12)/t2_kcore_levels.json        # Table 3: k-core communities per level (no singletons)
batch_newcov: assets ; @cat $(A)/tables/batch.tex; cat $(V12)/t3_arxiv.json   # Table 4: coverage of new nodes, 2020 pool
wiki_all:     ; @cat $(V12)/t4_wiki.json                # Table 7: all maintainers on Wikipedia (snapshot adapter)
extra_rows:   ; @cat $(V12)/t4/extra_rows.json; cat $(V12)/t4/scale_*.json   # Table 7: DF Leiden, local repair, citation2 cells
reuse_checks: ; @cat $(V12)/t5_reuse_checks.json        # Table 8: fresh cells, self-comparison in the fidelity check
