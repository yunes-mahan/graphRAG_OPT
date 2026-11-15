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
        lcc kcore_levels batch_newcov wiki_all extra_rows reuse_checks v15_checks \
        tech_report w5_instability hier_frontier cascade exact_tokens w4_metrics deletions seeds virtual_view kcore10 use_lcc_top v17_checks hybrid_p \
        coarse_answers never300 costsens

all assets:
	cd $(A) && PAPER_RESULTS=$(RES) $(PY) make_assets.py > make_assets.log && $(PY) make_overview.py > /dev/null
	@echo "tables -> $(A)/tables/, figures -> $(A)/figures/, numbers quoted in the text -> $(A)/numbers.tex"

check:
	PAPER_RESULTS=$(RES) $(PY) $(A)/check_numbers.py

reproduce:
	PYTHON=$(PY) ./quick_reproduce.sh

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
v15_checks:   ; @for f in v15_hybrid_qa v15_fid_stale v15_delta_lcc v15_bfs_tokens; do echo "== $$f"; cat $(V12)/$$f.json; echo; done   # V15 checks (D2-D4, F2)

# ---- V16 (scripts revision/v16/, cached results artifacts/v16/; rerun: copy revision/v16 into $GCM_RESULTS/v16,
#      run the named script there; graphrag_hier.py hooks: HIER_W4, HIER_EXACT, HIER_ITER, HIER_ONLY=hier_frontier)
V16 = $(RES)/v16
tech_report: assets           # artifact technical report, every number filled in from paper_assets/numbers.json
	cd $(A) && $(PY) make_tech_report.py ../docs/TECH_REPORT.md
w5_instability: ; @cat $(V16)/w5_instability.json; for g in entity wikidata; do $(PY) -c "import json;d=json.load(open('$(V16)/w5_stream_$${g}_iter50.json'));print('$$g', [round(d[s]['reindex+match']['cost']['0.9|1'],3) for s in d])"; done
hier_frontier:  ; @cat $(V16)/w1_summary.json; ls $(V16)/w1
cascade exact_tokens: ; @cat $(V16)/w3_summary.json $(V16)/w3_rowsum_check.json
w4_metrics:     ; @ls $(V16)/w4      # Table 2 runs with Q on the LCC and content coverage (make table2 shows the table)
deletions:      ; @ls $(V16)/w8
seeds:          ; @cat $(V16)/w6_summary.json
virtual_view:   ; @cat $(V16)/w9_virtual_view.json
kcore10:        ; @cat $(V16)/w10_kcore10.json
use_lcc_top:    ; @ls $(V16)/p3_*.json
v17_checks:     ; @for f in m2_eta_hf m3_deletion_hf m8_checks; do echo "== $$f"; cat $(RES)/v17/$$f.json; echo; done   # V17 (M2, M3b, M8)
hybrid_p: assets ; @grep -E "MoreHyb" $(A)/numbers.tex   # V17c: hybrid stores vs fresh, 300 questions (Wilcoxon, uncorrected)
