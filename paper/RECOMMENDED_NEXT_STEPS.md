# GeoSignal paper: recommended next steps

**Prepared:** 7 October 2026  
**Decision to make:** Add another case, enlarge the time-series dataset, or both?

## Recommendation

For the current paper, prioritize **more carefully selected event cases and independent evaluation labels**. Do not add Granger or Leontief results just to make the architecture look complete. Expand the historical archive for temporal analysis only as a separate workstream, after deciding exactly what daily geopolitical-risk signal and market outcome the test will evaluate.

The current weakness is not simply “too few rows.” It is that the paper has only two bounded examples and lacks independent labels for relevance and exposure. Adding thousands of unlabeled metadata rows would not by itself fix that weakness.

## Which path answers which question?

| Proposed work | What it could improve | What it cannot fix by itself | Recommendation |
|---|---|---|---|
| Add a third or several additional cases | Shows whether the workflow behavior repeats across events and topics; creates a better basis for a multi-case evaluation | Does not make bad labels or weak exposure mapping reliable | **Highest priority.** Choose cases with a reproducible selection rule and varied event types/dates. |
| Manually label retrieval candidates | Measures whether returned articles are relevant instead of using exact URL identity as a proxy | Does not validate event-to-sector/company mapping | **Do alongside more cases.** Use at least two independent reviewers for a sampled set; document the rubric and disagreements. |
| Validate exposure links | Tests whether event/article evidence maps to the sectors or securities included in market analysis | Does not establish causality of later returns | **Required before claiming system-generated market exposure.** Define the taxonomy and links before looking at returns. |
| Extend daily event/news and market series | Could make a temporal prediction question testable over time | More dates alone do not make a valid risk measure or prove causality | **Secondary, conditional work.** Build a consistent multi-year panel and predefine the analysis first. |
| Run Leontief propagation | Could illustrate modeled cross-sector propagation if data and shocks are sound | A toy/synthetic matrix cannot support empirical claims | **Defer** until an authoritative empirical input–output table and defensible shock mapping are assembled. |

## Suggested staged plan

### Stage 1 — Freeze the question and evaluation protocol

Keep the paper’s present scope as an offline workflow-feasibility study. Write down, before adding cases:

1. Which event types and dates qualify, and why.
2. What counts as a relevant article, with examples for “direct,” “adjacent,” and “unrelated.”
3. How duplicate GDELT event codes and repeated URLs are handled.
4. How event-to-sector/security exposure will be assigned and independently checked.
5. Which metrics will be reported (for example, precision/recall at a chosen cutoff, coverage, and uncertainty), and which results are descriptive only.

This prevents selecting only cases or outcomes that look favorable after analysis.

### Stage 2 — Expand the case set before expanding every data source

Add a small pilot set of varied cases from the available 2024 Q1 archive—such as sanctions/export controls, conflict or shipping, and commodity/supply disruption—using the same inclusion rule. A practical first pilot might be 6–10 cases total, counting the existing two; this is a workflow-development target, **not** a claim that this number guarantees statistical power or generalizability. Ensure each case has an auditable event date, a traceable source, and enough relevant candidate data to evaluate.

Keep event-level units distinct from article-level units. For example, multiple GDELT event-code records pointing to the same article should not be counted as independent relevant articles.

### Stage 3 — Label the cases independently

- Draw a documented, stratified sample of retrieval outputs, including high-ranked candidates and misses/known source links.
- Have at least two reviewers apply the rubric independently, without seeing market returns where feasible.
- Report agreement and resolve disagreements using a documented procedure.
- Separately review sector/security exposure links. Do not use a hand-selected basket as if it were the model’s prediction.

This work directly addresses the current evidence gap: there is one small single-rater relevance audit, and no validated exposure mapping.

### Stage 4 — Re-run the shared pipeline and report the failures too

Run the same frozen retrieval and mapping settings on all included cases. Keep each case’s metrics and then summarize across cases. Include baseline comparisons (at minimum direct URL matching versus lexical ranking) and preserve the negative cases. Report metadata-based retrieval as metadata-based retrieval, not full-text semantic understanding.

### Stage 5 — Decide whether a longer temporal study is worth doing

The existing daily archive is only 2024 Q1 (91 days). The project log calls that a development sample and marks Granger underpowered. A stronger temporal analysis would require:

- a substantially longer, consistently collected daily panel (the project notes propose 2022–2024 as an expansion target; the final span should be justified by data quality and model design);
- a clearly defined daily geopolitical-risk series—not just a few selected events—and aligned sector return series with a stated market and sector taxonomy;
- checks for missing dates, market holidays, event-date lag, stationarity/integration order, and sensitivity to lag choices;
- a correct implementation of the chosen Toda–Yamamoto modified Wald test, with a prespecified hypothesis and adjustment for repeated tests if many sectors/lags are examined; and
- language limited to predictive precedence. Granger results do **not** establish a causal mechanism.

Do not run the current function and present its output as a final Toda–Yamamoto finding without that methodological review. If the research question remains historical workflow feasibility, this temporal analysis can stay future work.

### Stage 6 — Treat Leontief as its own data project

Before reporting empirical propagation, obtain a documented official or research-grade input–output table for the relevant economy/year, validate sector aggregation and coefficient construction, check model stability, and define how a measured event translates into a demand or supply shock vector. Then report sensitivity to the table year, aggregation, and shock assumptions. Until then, any five-sector/toy or synthetic result must be labeled illustrative and kept out of the empirical findings.

## Recommended order for the draft being sent to faculty

1. Keep both unrun methods visible as planned architecture/future work, with the current limitations stated plainly.
2. Finish the strongest low-cost validation step: a documented multi-case sample with independent relevance labels and reviewed exposure links.
3. Use that evidence to decide whether a longer daily temporal panel addresses the paper’s actual question.
4. Pursue Leontief only if the paper’s central contribution requires cross-sector propagation and empirical input–output data can be validated.
5. Before journal submission, apply the selected venue template and resolve the remaining author declarations and reproducibility details.

## Open items for the author

These are the manuscript items that need your factual decision or project choice; they should not be guessed:

- **Code and results availability:** public repository URL, or “available on request,” or another venue-accepted statement.
- **Software versions:** record versions from the final reproducible run.
- **Funding:** funding source and grant number, or confirm no external funding.
- **Conflicts of interest:** disclose any, or confirm none.
- **Next study scope:** choose a multi-case evaluation first, a multi-year temporal study, or a staged combination. The recommendation above is multi-case evaluation first.
- **Target venue:** not selected yet. The draft can remain generic until you receive venue requirements.

## Read alongside

- [Project status study guide](PROJECT_STATUS_STUDY_GUIDE.md)
- [Evidence audit](../docs/PHASE_VI_EVIDENCE_AUDIT.md)
- [Research status](../docs/RESEARCH_STATUS.md)
- [Current manuscript](draft_manuscript.tex)

