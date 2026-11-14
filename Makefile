# One command per table / figure of the paper. Everything is regenerated from the cached results in artifacts/
# (no LLM calls, no API key, a few minutes on a laptop). Override the results tree with RES=/path/to/results.
#   make all        regenerate every table, figure and number (paper_assets/{tables,figures,numbers.tex})
#   make check      regenerate and compare EVERY number with the paper's (artifacts/expected_*)
#   make table4     regenerate and show Table 4 (same for table1..table9, fig1..fig5)
#   make reproduce  quick end-to-end check incl. re-running two maintainers from the raw OGB data (quick_reproduce.sh)
PY  ?= python
RES ?= $(CURDIR)/artifacts
A    = paper_assets

.PHONY: all assets check reproduce numbers table1 table2 table3 table4 table5 table6 table7 table8 table9 \
        fig1 fig2 fig3 fig4 fig5

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
table4: assets ; @cat $(A)/tables/tokenmetric.tex     # token-level score of the flat maintainers
table5: assets ; @cat $(A)/tables/qa.tex              # local questions
table6: assets ; @cat $(A)/tables/extra.tex           # Wikidata, entity graph, ogbl-citation2
table7: assets ; @cat $(A)/tables/reuse.tex           # report reuse rules
table8: assets                                       # key findings: text table, all numbers are macros
	@grep -E "Hier(Wd)?Split|HierEtaTau|HierCov|NhDrop|BfRollFrom|MoreN|ProbeStaleLoss|LocVec|LocSums|Del(Gu|Zep)Five" $(A)/numbers.tex
table9: assets ; @cat $(A)/tables/costsens.tex        # cost-model sensitivity

fig1:
	cd $(A) && $(PY) make_overview.py && echo "$(A)/figures/overview.pdf"
fig2: assets ; @echo "$(A)/figures/noise.pdf"
fig3: assets ; @echo "$(A)/figures/pareto_heldout.pdf"
fig4: assets ; @echo "$(A)/figures/lambda_heldout.pdf"
fig5: assets ; @echo "$(A)/figures/qa.pdf"
