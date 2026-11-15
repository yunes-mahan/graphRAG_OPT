# Technical report (artifact) — secondary results of the VLDB 2027 EA&B paper

Every number below is filled in from `paper_assets/numbers.json` by `paper_assets/make_tech_report.py`
(the same macros as the paper; `make tech_report`). Section numbers are cited in the paper as "artifact TR §k".

## §1 Rollback price factor (paper §10)
The rollback price uses (T+1)/T with T = 15 (the 16-version main streams) for all graphs. With the exact T of each
stream the factor ranges from 1.042 to 1.125, so the fixed factor misprices a report by -5.2% to
+2.4% (`revision/v12/a5_rollback_factor.py`).

## §2 Sensitivity to the token cost model (paper §11, threats)
Changing each constant of tok(S) by ±50 %, with λ recalibrated (0.23–0.60) and the rollback re-run:
- Kendall τ between the base and the variant rankings: 0.89–1.00 for Table 5; at least
  0.79 for all rows of Table 7 except DF Leiden (the sensitivity analysis predates that row);
- the best flat maintainer is the same in 6 of 6 variants; Table 2 rankings unchanged;
- re-cluster split share 61%–63% (both graphs); τ share within re-clustering
  81%–85% on the entity graph and 97%–98% on Wikidata.

## §3 Where the rewrite cost sits; greedy vs. optimal matching (paper §6)
**Where the cost sits.** At τ=0.9, re-indexing with matching rewrites 95% of the top-level report tokens (Wikidata 87%) but 46%–54% at levels 2–4, and 100% of the tokens of communities with more than 200 members against 51% for those with at most ten. Part of the fine-level cost comes from the rule rather than from Leiden: swapping a single member of a ten-member community gives J=9/11≈0.82 and forces a rewrite. Optimal (Hungarian) instead of greedy matching changes the cost by 0.000 at τ=0.9 (pairs with J≥0.9 are unique best matches), at most 0.001 at τ=0.7/0.5 and -0.011 with "never".

## §4 Refreshing under a token budget (paper §6)
**Refreshing under a token budget.** With a fixed budget of 10% of a re-index per update, refreshing the lowest-Jaccard reports first leaves the least structural staleness (QA stream 21% vs. 25% for round-robin), which is close to built into the metric.

## §5 Sensitivity to the maximum cluster size (paper §6)
**Sensitivity to the cluster size.** With maximum cluster sizes of 25 and 50 instead of GraphRAG's default 10, the re-cluster split explains 65%–71% of the cost variance, τ at least 99% within re-clustering, and deleting 0.1% of the edges rewrites 66%–82%: larger clusters make re-clustering costlier and less stable, not cheaper.

## §6 Structural scores vs. recall (paper §8)
**Structural scores: no evidence either way.** The 16 QA-stream stores have only 4 distinct modularity values, and the score falls with every refresh by construction; its rank correlations with recall are not significant (ρ=-0.40 for modularity, ρ=-0.33 for the score). We therefore report cost, coverage and answers separately.

## §7 Local questions in full (paper §8)
**Local questions: non-inferior.** Recall ranges from 0.29 to 0.38 across the 16 stores against 0.32 for fresh reports (Table (paper), Fig. (paper)a), while update input tokens range from 22k to 744k; no store is significantly worse after Holm correction. Because non-significance is not evidence of equality, we ran two one-sided tests (TOST) with a margin of ±0.05 recall, about 16% of the fresh-report recall; the margin was set after the first 120 results and before the 180 further questions were generated (file times in the artifact). With 120 questions, 7 of 16 stores are equivalent to fresh (all 90% intervals within [-0.069,0.105]). With 300 questions for six key stores, all 6 maintained stores are non-inferior (intervals within [-0.023,0.086]), 3 are equivalent, and the two hybrid stores score higher than fresh. This gain is not due to staleness: fresh reports of each maintainer's *own* final partition already differ from the recomputed reference (hybrid 0.369 vs. 0.326, p=0.006), and against its own-partition reference, which isolates staleness, every maintainer's τ=0.9 store is equivalent (4 of 4; intervals within [-0.029,0.040]). On broad questions, the partition thus affected recall more than the age of the reports. Three further checks are consistent with this: a second judge (Qwen3-14B; 3 of 16 equivalent, weak agreement with the first judge, ρ=0.18), a second answering model (Qwen3-14B reading the same reports; all 5 maintained stores equivalent at lower recall, 0.17–0.21), and a fully open replication (Qwen3-14B writes all reports of both sides and 145 questions from the same prompt and seed papers, answers and marks key points): all 4 maintained stores at τ=0.9 are non-inferior (intervals within [-0.038,0.090], 2 equivalent; recall 0.17–0.24, fresh 0.19). The six stores of the 300-question test were the four maintainers at τ=0.9 and the never-refreshed recompute and hybrid stores, chosen without a rule fixed beforehand; in the open-model pipeline with 288 questions, the two never-refreshed stores not in the 300-question test, warm-started and frontier-restricted Leiden, are also equivalent to fresh reports (Δ=-0.004, intervals within [-0.031,0.022]). With a stricter margin of ±0.035 (11% of the fresh recall), all 6 stores of the 300-question test, all 4 own-partition comparisons and both 288-question open-model stores stay non-inferior (lowest bounds -0.023, -0.029, -0.031); in the smaller open replication, 3 of 4 do (warm start: -0.038).

## §8 Global questions in full (paper §8)
**Global questions: no store worse, equivalence not established.** Both global tests use the same pairwise judge (gpt-5.6-terra, four criteria, both presentation orders); a store answers from all its reports on the QA stream and by GraphRAG-style map–reduce over its reports on the entity graph. On the QA stream, maintained, never-refreshed and title-only report stores obtain win rates between 0.41 and 0.60 against fresh reports (smallest p=0.06); on the flat entity-graph stores none is significantly worse after correction (smallest uncorrected p=0.008 over 28 tests). With 30 questions, however, equivalence within [0.4,0.6] is established for only 3 of 16 store–criterion pairs, so differences of about 0.1 in win rate cannot be excluded.

## §9 Entity graph at levels 0–2 with the open model (paper §8)
**Entity graph, GraphRAG's levels 0–2 (open model).** On the LLM-extracted entity graph we compared the maintained index (warm-started hierarchical Leiden with identifier matching, τ=0.9) with a fresh re-index at GraphRAG's default levels 0–2 (8,422 communities in the fresh index), fully with Qwen3-14B: 148 distinct local questions were generated from recent documents and their seven most similar documents with the prompt above; each store retrieves the six communities whose report input (member entities and their relationships) is most similar to the question, so that only retrieved communities need a report (862 reports written lazily, each from the member set at its last rewrite). This retrieval is our own setup (top six similar communities, reports written lazily), not GraphRAG's local search. Only 0.6% of the retrieved maintained reports were stale, so this test mainly measures the partition, not staleness: the warm-start partition loses a little (Δ=-0.030, Wilcoxon p=0.36, 90% interval [-0.063,0.002]; recall 0.26 vs. 0.29), and non-inferiority within ±0.05 is not established on this graph.

## §10 Targeted staleness and coverage probes (paper §8)
**Targeted probes: where staleness shows.** Broad questions may not touch what changed; we built two probes (100 questions each, same pipeline). *Staleness probe:* for the four never-refreshed stores, 25 reports per maintainer whose community has since gained at least four papers, with questions supported only by the papers that joined after the report was written, answered by the stale store and by fresh reports of the same partition: stale reports reach recall 0.21 against 0.41 (Δ=-0.20, p<0.001; significant for 2 of 4 maintainers after Holm correction). This is a worst case by construction; the expected loss is about 0.20 times the share of questions about recent content. *Coverage probe:* for GraphRAG's update and Zep-style attachment, questions from documents whose entities lie mostly outside every top-level community are answered from the top-level reports: recall 0.12 against 0.16 for the fresh re-index (Δ=-0.04, p=0.022 pooled; neither rule is significant alone).

## §11 Reports vs. vector retrieval (paper §8)
**Summaries serve global questions.** For local questions, plain vector retrieval of 12 papers reaches recall 0.56 against 0.32 for fresh reports (p<0.001), also with the same retriever for both. For global questions the picture reverses: vector retrieval wins only 0.15 of the comprehensiveness comparisons (0.28 with length-matched context), and reading only the reports closest to the question loses against reading all of them. In the flat entity-graph experiment, GraphRAG's delta store holds 676 reports instead of 149, so every global question reads 4.4× the measured report tokens (its reports are slightly shorter, so less than the report ratio) and its update savings are used up after 7 questions.

## §12 Summary-priced rollback in full (paper §10)
 A gain on the score is expected by construction (the price comes from it). On held-out orders, rollback improves warm-started Leiden on 20 of 24 orders (mean +0.019, p<10^{-5}; it is slightly worse on collab-10) and frontier-restricted Leiden on 24 of 24 (p<10^{-7}), and it improves both on all five orders of the Wikidata and entity graphs and on ogbl-citation2 (Table (paper)), at +3% running time over warm-started Leiden. On the development orders, priced and scored with the same λ, the gain grows with the weight (at λ=1, 0.642 vs. 0.505) and disappears at λ=0.1 (0.783 vs. 0.786). On Wikipedia it lowers warm start's relative token cost from 0.52 to 0.07 (modularity 0.583 vs. 0.616). On the entity graph its answers are not significantly worse than those of fresh reports (smallest p over the four criteria 0.07), but equivalence is not established.

## §13 Recomputation and appending (paper §7)
**Recomputation and appending score low.** Recomputing Leiden after every update rewrites the reports of communities that hold 95% of the nodes on both main streams, and GraphRAG's delta update fragments its partition as the graph grows (mean modularity 0.358 on arXiv-15 against 0.743 for warm-started Leiden); both are below the warm-started, frontier-restricted and lazy maintainers in Table (paper).

## §14 V16 robustness checks (CPU; scripts in `revision/v16/`, results in `artifacts/v16/`)
- **Instability vs. configuration (W5, `w5_instability.py`, `w5_stream.sh`).** Final graph, GraphRAG settings, τ = 0.9:
  permuting node identifiers and edge order of the unchanged graph rewrites 49% (Wikidata 71%) of the
  report tokens; matching across levels after a 0.1 % edge deletion 49% (71%); 50 Leiden
  iterations instead of 1, for reference and perturbation, 47% (75%), over the real streams
  0.56 and 0.77 of a re-index per update; 50 and twice as many iterations still
  differ by 20% (64%), i.e. Leiden has not converged.
- **GraphRAG-format tokens and the report cascade (W2, W3; `exact_ctx.py`, `w3_summary.py`).** Report input tokens counted
  in GraphRAG 3.3.0's context format (entity and relationship CSV tables, prompt template, o200k_base, 8,000-token
  limit), output tokens = mean of our measured reports: Spearman 1.00 with the token model over 20
  maintainer × τ cells (re-index + matching at τ = 0.9: 0.55 vs. 0.57 modelled). Per-row
  summation error ≤ 1.7%. GraphRAG's build_mixed_context substitutes sub-community reports when a parent's
  context exceeds the limit; forcing such parents to be rewritten when a child report changes adds at most
  0.01 of a re-index per update. Break-evens with these tokens: 185 and 3.5 global
  queries per update.
- **Common-node-set modularity and content coverage (W4).** Table 2 now reports Q on the LCC with unclustered nodes as
  singletons; on clustered nodes only, Zep-style leaf Q is 0.585 (highest) vs. 0.491 on the
  LCC (lowest). Content coverage of never-refreshed stores: re-index + matching 89%, GraphRAG update
  81%, Zep-style 48% (Wikidata 95%, 95%,
  76%).
- **Hierarchical frontier-restricted maintainer (W1, `hier_frontier.py`).** 0.25 (Wikidata
  0.10) of a re-index per update at τ = 0.9; small batches 3.1–10%; leaves above the
  size bound at the final version: 164 vs. 81 for GraphRAG (Wikidata 4,435 vs.
  3,612); partition and nesting checks hold in every version.
- **Synthetic deletion streams (W8, `w8_deletions.py`).** Rollback gains: warm start +0.016 to +0.062,
  frontier +0.004 to +0.029 on arXiv-15, collab-12 and the flat entity graph (gains positive in
  all cases). DF Leiden is not run: our driver of Sahu's code passes insertions only.
- **Materialised vs. virtual view (W9, `w9_virtual_view.py`).** 4,027 measured tokens per report; writing the
  six reports a local question reads costs 24k tokens; keeping reports is cheaper above 0.2–5.6
  local questions per update (QA stream).
- **k-core size bound (W10, `w10_kcore10.py`).** The authors' code with MAX_CLUSTER_SIZE = 10 (ours) rewrites
  2% (Wikidata 8%) after a 0.1 % edge deletion. The authors answer global questions from all
  communities of the leaf or its parent level; our 111-report selection is ours.
- **Leiden seed (W6, `w6_seeds.py`, `w6_summary.py`).** Seeds 1 and 2 for every maintainer and held-out order of Table 5
  (every leidenalg call seeded; seed 0 reproduces the stored results exactly): adding the seeds changes no ranking and no best cell; per-group means
  and 95 % intervals over seeds × orders are in `artifacts/v16/w6_summary.json`.
- **Qwen judge (W7).** Cached verdicts are complete JSON (39 output tokens each), answers are swapped between orders as
  intended and no default label exists; the second-position preference is model behaviour.

## §15 Deletion and lineage in full (paper §9)
Deleting a document (retraction, erasure request) must also invalidate every report derived from it. We deleted a random 1% or 5% of the documents of the entity graph (3 draws each), removed the entities and relationships extracted only from them, and maintained GraphRAG's hierarchical stores (Section (paper)) as their rules prescribe: re-clustering maintainers re-cluster and apply the τ=0.9 rule; GraphRAG's update has no deletion path and Zep-style attachment no removal step, so they keep all reports. A report *leaks* if it is still served and was written for a member set that contains a removed entity, or both endpoints of a removed relationship. With 5% of the documents deleted, 8.4% of GraphRAG's served reports leak and 14.3% of the Zep-style reports (1%: 1.8% and 3.6%); many leaking reports still name the deleted entity in their text (top level: 76% and 46%). Re-clustering maintainers leak only 0.3%, but only because their instability rewrites 47% of the report tokens anyway. These rates are lower bounds: facts from deleted documents can also survive in reports through surviving entities whose descriptions drew on them. Lightfoot  deletes documents exactly from a provenance-tracked graph without community reports, and RelGuard  perturbs or removes relations before indexing for differential privacy; neither maintains reports. Lineage-based invalidation, as proposed for agent memory by Agentic Unlearning , removes the leaks: rewriting every served report that contains a removed entity or relationship costs 8.4% (GraphRAG's update) and 9.8% (Zep-style) of the tokens of a full re-index, and conservative document-level lineage, which rewrites every report containing an entity mentioned in a deleted document, costs 31.7% and 15.3%. Deletion is thus the case where a staleness tolerance is not acceptable.

## §16 Topic-anchored report reuse in full (paper §10)
**Topic-anchored report reuse.** Every report is stored with a topic fingerprint, the mean of its members' mean-centred text embeddings; a community with at least five members reuses the stored report with the most similar fingerprint if the cosine is at least θ=0.95, across identifiers, splits and merges. On the QA stream (Table (paper)) it cuts the reports written for recomputed Leiden from 213 to 91 (39% of the tokens) without a significant loss in local recall. The global judge even prefers it to fresh reports (comprehensiveness 0.73, p=0.002), which we do not read as a gain: the judge rates breadth, not currency; the effect does not survive Holm correction over the 64 tests of the reuse study (p=0.15); 15 of the 16 reuse stores score above 0.5 against the same fresh answers, a shared tilt rather than an effect of reuse; and answer length does not explain it (368 vs. 368 words). The rule is lenient: a control that replaces members at random so that J=0.7 still passes θ in 62% of the communities, so the fingerprint acts largely as a looser overlap rule, and the judge rates 26–45% of the reports reused for a different member set as substantively different from a fresh report (Table (paper)). Combined with rollback and title-only inputs, update tokens fell to 2.5% of recomputation with local recall 0.33.

## §17 Scalability and the Neo4j harness (paper §7)
**Scalability and deployment.** On the full ogbl-citation2 graph (2.9M nodes, 30.4M edges), warm-started, frontier-restricted and rollback maintainers needed 5.4–6.1 minutes per update on one laptop (recomputation 7.8) and at most 8.4 GB of memory; graph computation is negligible next to LLM report writing. In Neo4j, reading identifiers back reproduces the offline evaluation exactly, and storing the Wikipedia stream as one versioned graph needs 6.2× fewer link relationships than per-year snapshots.
