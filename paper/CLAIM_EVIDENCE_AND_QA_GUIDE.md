# GeoSignal manuscript: claim evidence and question guide

**Prepared:** 7 October 2026  
**Manuscript covered:** [`draft_manuscript.tex`](draft_manuscript.tex)  
**Purpose:** Help the author explain where each important manuscript claim comes from, why the method or wording was chosen, what it means, and what it does not mean. This guide is an evidence map, not a substitute for inspecting the saved outputs or verifying the final bibliography.

## How to use this guide

Before defending a result, identify which kind of statement it is:

- **Observed result:** a number or output saved from a run.
- **Method choice:** a documented design decision explaining how an analysis was done.
- **Interpretation:** a limited explanation of what the result suggests.
- **Proposed capability:** an intended future part of the architecture, not a completed result.

The central defense of the paper is that it labels these types separately. A saved URL link supports source traceability; it does not prove article relevance. A CAR supports a descriptive comparison for a selected basket; it does not prove the event caused the return.

## 1. Central claim and contribution

### Claim: GeoSignal was evaluated by replaying historical data, not through live deployment

**Where in the manuscript:** Abstract; Introduction; System architecture; Limitations; Conclusion.

**Evidence:** Both case manifests identify the work as an offline historical replay. The run summaries record archived 2024 inputs and saved local outputs; they do not record a live feed, alert service, uptime, or latency experiment. See [`RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md), [`redsea_run_summary_20240112.json`](../results/datasets/redsea_run_summary_20240112.json), and [`experiment_log.json`](../results/experiment_log.json).

**Why this framing:** It matches the work actually performed and avoids implying a production system exists. The paper tests whether selected workflow stages can be exercised on archived inputs.

**If asked:** “The study is a historical replay of a prototype architecture. We have not evaluated a live monitoring service, so I make no claims about live reliability or operational performance.”

### Claim: The contribution is a workflow feasibility and evidence-quality account, including failure cases

**Where:** Abstract; Introduction contribution list; Results summary; Conclusion.

**Evidence:** The saved outputs include both successful source-link joins and poor lexical retrieval, sparse sector labels, descriptive market calculations, and explicit not-run stages. The results can be traced to the case summary files and metrics ledger below.

**Why this contribution:** A chained workflow can fail at the hand-offs between event records, news, exposures, and market series. Reporting which links worked and which did not is more accurate than presenting the architecture diagram as if every block were validated.

**If asked:** “The contribution is diagnostic: we show what the current historical pipeline can connect, quantify where it fails, and distinguish traceability from semantic and economic validation.”

## 2. Data and case-selection claims

### Claim: The archive contains 91 daily GDELT event files and 91 daily GKG files for 2024 Q1

**Where:** Data sources; planned methods explanation; limitations.

**Evidence:** Local archive inventory and the data-source/status documentation: [`RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md), [`DATA_SOURCES.md`](../docs/DATA_SOURCES.md), and [`CONTEXT.md`](context/CONTEXT.md).

**Why this period:** It is the period actually archived and used for the two replays. It was not selected as a representative multi-year sample.

**If asked:** “The 2024 Q1 interval is the available development archive for these cases. It is adequate to replay selected dates, but too short and narrow for broad temporal or population-level claims.”

### Claim: GKG v1 is metadata, not full article text

**Where:** Abstract; Related Work; Data; Methods; Discussion; Limitations.

**Evidence:** The source description and project limitation L-001 explain the fields used: themes, persons, organisations, locations, and related metadata. See [`DATA_SOURCES.md`](../docs/DATA_SOURCES.md) and [`LIMITATIONS.md`](../docs/LIMITATIONS.md).

**Why it matters:** BM25/TF-IDF scores are computed over metadata proxies. They cannot be described as full-text semantic understanding or article-level truth verification.

**If asked:** “Our search corpus contains GDELT metadata fields, not article bodies. We therefore call the result metadata-based lexical retrieval and treat it as a candidate-generation diagnostic.”

### Claim: The cases are purposive examples, not representative samples

**Where:** Data and cases; Limitations.

**Evidence:** The Red Sea selection uses date, CAMEO root, URL/actor keyword filters, a minimum mention threshold, URL deduplication, and a top-30 cap. The semiconductor case is anchored on one VOA source report and its same-day records. Details are in [`run_redsea_case.py`](../scripts/run_redsea_case.py), [`run_chip_controls_case.py`](../scripts/run_chip_controls_case.py), and the saved summaries.

**Why these two cases:** They exercise different structures: a multi-record conflict/transport sample and a policy-news report with a traceable source URL. They permit a workflow comparison but do not establish representativeness.

**If asked why only two:** “These were bounded feasibility cases selected from the archived data. We explicitly do not infer that they represent geopolitical events generally. The next study needs a prespecified, broader multi-event sample.”

## 3. Linkage and retrieval claims

### Claim: Direct URL linkage covers all 30 selected Red Sea event records

**Where:** Abstract; Results, Case 1; Table `redsea-link`; Stage summary.

**Evidence:** [`redsea_run_summary_20240112.json`](../results/datasets/redsea_run_summary_20240112.json) and [`redsea_retrieval_evaluation_20240112.json`](../results/datasets/redsea_retrieval_evaluation_20240112.json): 31 URL-link rows, 30/30 selected event records with at least one direct link.

**Why 31 rows for 30 events:** These are event/article join rows, not a one-to-one count of unique events. A selected event can have multiple linked rows.

**What the claim means:** An event record and a GKG record contained the same normalized source URL under the scripts' canonicalization rules.

**What it does not mean:** It does not establish that the article semantically describes the coded event, that the report is true, or that the event caused a market outcome. The scripts normalize URLs but do not fetch pages to check redirects.

**If asked:** “The URL join is our provenance signal. It tells us that the same source URL appears in both datasets; we do not treat it as a relevance label.”

### Claim: Red Sea lexical retrieval generated 230 candidate pairs for 25 of 30 events

**Where:** Abstract; Results, Case 1; Table `redsea-link`.

**Evidence:** [`redsea_retrieval_evaluation_20240112.json`](../results/datasets/redsea_retrieval_evaluation_20240112.json), [`redsea_hybrid_event_news_pairs_20240112.csv`](../results/datasets/redsea_hybrid_event_news_pairs_20240112.csv), and [`evaluate_redsea_retrieval.py`](../scripts/evaluate_redsea_retrieval.py).

**Method:** Case 1 uses BM25 top-10 retrieval followed by TF-IDF cosine filtering at 0.05 over same-day GKG metadata. The code configuration uses BM25 `k1=1.5` and `b=0.75`. The configured ±3-day band was not exercised in this case.

**Why use both:** The direct URL path preserves an existing source link; lexical ranking attempts to broaden the candidate set when no direct URL is shared. The hybrid design lets the outputs be inspected separately.

**If asked:** “The 83.33% is candidate-generation coverage—25 events got lexical candidates. It is not 83.33% accuracy or relevance.”

### Claim: Only one of the 230 Case 1 lexical pairs was an exact-source match; top-10 exact-source recall was 1/30

**Where:** Abstract; Results; Table `redsea-link`; Discussion.

**Evidence:** The retrieval evaluation JSON records 1 lexical exact-source pair (0.43% of 230) and an exact source in the top ten for 1 of 30 events (3.33%).

**Why compare against URLs:** No independently judged relevance dataset exists for this case. Shared URL identity can serve as a limited **silver signal** to check whether lexical ranking recovers a known source.

**Important interpretation:** Other articles may be relevant even if their URLs differ. Therefore 0.43% is exact-source overlap, not lexical precision; 3.33% is recall of this exact-URL signal in the top ten, not recall of all relevant reporting.

**If asked:** “We use exact URLs only as a narrow reference signal. A relevant article need not share the event record’s URL, so we do not call this a full relevance evaluation.”

### Claim: In Case 2, query enrichment moved the exact article from rank 4,033 to rank 55

**Where:** Abstract; Results, Case 2; Table `semi-rank`.

**Evidence:** [`chip_controls_replay_summary_20240312.json`](../results/datasets/chip_controls_replay_summary_20240312.json) and [`evaluate`/case 2 candidate outputs](../results/datasets/).

**Method:** Same-day TF-IDF cosine ranking across 120,718 metadata rows. The enriched query adds semiconductor, advanced-chip, export-control, and Commerce Department terms. This is not a BM25 result.

**Why include both queries:** The comparison illustrates sensitivity to query wording and tests whether domain terms improve a known article's rank. The exact source remains outside the top 50 even after enrichment.

**If asked:** “Enrichment improves the exact article's rank substantially, but rank 55 is still outside our top 50. This is one diagnostic example, not a general retrieval benchmark.”

### Claim: The top-20 Case 2 audit had 0 direct, 8 adjacent, and 12 unrelated items

**Where:** Results, Case 2; Table `semi-rank`; Limitations.

**Evidence:** [`chip_controls_top20_relevance_audit_20240312.csv`](../results/datasets/chip_controls_top20_relevance_audit_20240312.csv) and [`chip_controls_relevance_audit_summary_20240312.json`](../results/datasets/chip_controls_relevance_audit_summary_20240312.json).

**Why the manuscript qualifies it:** The audit was done by one reviewer, on one query/case, and relied largely on titles, URLs, and GKG metadata. It has no inter-rater agreement and cannot support a general precision claim.

**If asked:** “This is an initial qualitative diagnostic. It signals that the retrieved list was mostly not directly relevant under our rubric, but a multi-rater, multi-case evaluation is needed for a defensible precision estimate.”

## 4. Sector, graph, and market claims

### Claim: The Red Sea sector mapper left 226/230 lexical candidates Unknown

**Where:** Abstract; Results, Case 1; Table `redsea-link`; Appendix on sector rules.

**Evidence:** [`redsea_sector_mapping_counts_20240112.csv`](../results/datasets/redsea_sector_mapping_counts_20240112.csv), [`redsea_run_summary_20240112.json`](../results/datasets/redsea_run_summary_20240112.json), and [`sector_mapping.py`](../src/geosignal/sectors/sector_mapping.py).

**Why rules were used:** Deterministic keyword rules make the current mapping inspectable and repeatable. Unknown is retained rather than forcing a label when no configured rule fires.

**What the result means:** Low coverage of the current rules on this candidate set.

**What it does not mean:** It does not show that only four real exposures existed. Nor do four keyword matches validate those firms/sectors as economically exposed.

**If asked:** “The 226 Unknown result is a limitation of our current rule coverage, not evidence that those candidates had no exposure. We need a labeled exposure taxonomy and independent review.”

### Claim: The Red Sea actor graph has nine actor nodes and ten co-occurrence edges

**Where:** Results, Case 1; Stage status.

**Evidence:** [`redsea_case/network_nodes.csv`](../results/datasets/redsea_case/network_nodes.csv), [`redsea_case/network_edges.csv`](../results/datasets/redsea_case/network_edges.csv), and run summary.

**Why include the graph:** It shows the descriptive actor-co-occurrence output produced by the workflow.

**Boundary:** There are zero sector nodes in this graph. It is not a supply-chain network, causal graph, or demonstrated propagation model.

**If asked:** “This graph summarizes actor co-occurrence in the selected records. We do not interpret its edges as economic transmission links.”

### Claim: Market-model CARs were calculated for selected baskets over 1, 3, 7, 14, and 30 trading days

**Where:** Methods, Market-model event windows and Researcher-defined baskets; Results tables and figures; Appendix raw returns.

**Evidence:** [`case_market_metrics.csv`](context/case_market_metrics.csv), [`redsea_market_model_event_study_20240112.csv`](../results/datasets/redsea_market_model_event_study_20240112.csv), [`chip_controls_market_model_event_study_20240312.csv`](../results/datasets/chip_controls_market_model_event_study_20240312.csv), and the case analysis scripts in [`scripts`](../scripts/).

**Method stated in the draft:** Estimate a market model over 120 pre-event trading sessions, calculate abnormal returns relative to the Nifty 50 for India baskets or S&P 500 for global semiconductors, then sum abnormal log returns over each post-event window. Values are reported in percent as approximate CARs.

**Why this method:** It supplies a standard, transparent descriptive market context for the event windows. The baskets permit a calculation even though GeoSignal's exposure mapper has not validated the event-to-security links.

**Why the manuscript says descriptive:** The baskets were manually selected, the saved analyses report no significance tests, only two cases were run, and the event dates do not isolate the event from other news. A market-model CAR is not automatically a causal effect.

**If asked:** “These calculations show how returns in selected proxy baskets compared with benchmark-based expected returns around the chosen date. They are descriptive, not pipeline predictions and not causal estimates.”

### Claim: Case 1 30-day CAR is +5.58% for ports/logistics and +17.78% for energy

**Evidence:** [`case_market_metrics.csv`](context/case_market_metrics.csv), rows for Red Sea ports/logistics and energy at 30 trading days. The manuscript table reports rounded values; full precision is in the CSV.

**Safe reading:** These are estimates on researcher-defined baskets versus Nifty 50. Their direction and size do not identify an event effect.

### Claim: Case 2 global semiconductor CAR is -5.08% at 7 days and -13.94% at 30 days

**Evidence:** [`case_market_metrics.csv`](context/case_market_metrics.csv), global semiconductor rows versus S&P 500, and [`chip_controls_market_model_event_study_20240312.csv`](../results/datasets/chip_controls_market_model_event_study_20240312.csv).

**Safe reading:** The basket is NVDA, TSM, ASML, and INTC, selected by the researcher. The long window includes potential unrelated news and market movements. It is not an estimate of the article's causal effect.

## 5. Claims about methods that were not run

### Toda–Yamamoto / Granger temporal analysis

**Manuscript status:** Shown as planned and explicitly not run.

**Evidence for not running:** The Red Sea run is one event/news date, the local archive is 90 days, and the experiment log labels the 90-day sample underpowered. There is no validated multi-year daily geopolitical-risk and sector-return panel. See [`experiment_log.json`](../results/experiment_log.json), [`LIMITATIONS.md`](../docs/LIMITATIONS.md), and [`granger_causality.py`](../src/geosignal/analysis/granger_causality.py).

**Why not force a result:** A short series can generate software output without providing a stable or well-justified inference. The code also requires a method-level review before it can be represented as a formal Toda–Yamamoto modified Wald analysis.

**If asked:** “It is a planned component, not a result. We need a much longer and validated daily panel, and we need to verify the test implementation and prespecify the hypotheses before reporting it.”

### Leontief input–output propagation

**Manuscript status:** Shown as planned and explicitly not run as an empirical result.

**Evidence:** [`leontief.py`](../src/geosignal/propagation/leontief.py) labels its built-in five-sector matrix illustrative; [`LIMITATIONS.md`](../docs/LIMITATIONS.md) and [`DECISIONS.md`](../docs/DECISIONS.md) record that no empirical case propagation was produced.

**Why not use the toy matrix as a result:** The numerical output would depend on assumed coefficients rather than observed inter-industry transactions. In addition, a shock vector must be justified independently; a generic event does not itself specify a measured sector demand or supply shock.

**If asked:** “We did not report Leontief propagation because we have neither a validated empirical input–output table nor a defensible measured shock vector. A toy example would be illustrative only.”

## 6. Figures, tables, and reproducibility claims

- The workflow diagram and CAR charts in the current LaTeX source are drawn with TikZ/PGFPlots. The draft contains no `\\includegraphics` command; these paper visuals are code-rendered diagrams/charts, not photographs or imported result-folder images.
- The project does have a saved network graph PNG under `results/figures/redsea_case/`, but the current draft does not embed it as an external image.
- Numerical tables should trace to the relevant saved JSON/CSV outputs, especially [`case_market_metrics.csv`](context/case_market_metrics.csv) and the two case summary JSONs.
- The manuscript states deterministic procedures and no random seed. Before submission, capture the exact package versions from the final environment/run and share the code/results through the availability route chosen by the author.
- The built-in LaTeX compiler previously stopped before source processing with “Unable to find standard directories for platform.” Therefore, compilation/layout is not currently verified by the built-in compiler. This is an environment/compiler setup issue, not evidence that the manuscript has successfully compiled.

## 7. Likely faculty questions and concise answers

**“What is the main finding?”**  
“The prototype can process selected archived records and preserve direct source links, but metadata retrieval and rule-based exposure mapping are weak. The evidence supports a bounded feasibility demonstration, not validated impact prediction.”

**“Why call 83.33% coverage if retrieval was poor?”**  
“That number only means 25 of 30 selected event records received lexical candidates. Exact-source overlap was 1 of 230 pairs, and neither metric is a human relevance accuracy score.”

**“Can you say the Red Sea event caused the positive energy CAR?”**  
“No. The basket was researcher-selected, no significance test was reported, and the event is not isolated from other market news. The estimate is descriptive context only.”

**“Why did the paper include a negative retrieval result?”**  
“It is directly relevant to whether the workflow connects events to reporting. Reporting it transparently shows where the prototype needs work and avoids presenting planned capability as demonstrated performance.”

**“What would make the next version stronger?”**  
“A prespecified multi-event sample; two or more independent reviewers labeling relevance and exposure; a stable evaluation protocol; and market analysis only after the exposure links are validated. A separate longer time-series study would be needed for Granger analysis.”

## 8. Expansion: how to preserve a defensible evidence trail

For every new case, keep a case record with:

1. Event identifier, real-world event date, GDELT detection date, source citations, and inclusion reason.
2. Raw data source, retrieval date, date coverage, row counts, filters, deduplication steps, and exact query text.
3. Code/config version and output filenames so another person can regenerate each metric.
4. Human relevance rubric, sampled items, independent reviewer labels, adjudication, and agreement.
5. Separately reviewed event-to-sector/security links, specified before market outcomes are inspected.
6. Market series, benchmark, estimation/event windows, missing-data handling, uncertainty/significance method, and sensitivity checks.
7. An explicit status for each claim: planned, run, independently checked, or validated across cases.

For a longer temporal study, create and audit the daily risk signal and aligned sector-return panel first. Record why that variable represents geopolitical risk, how gaps and market holidays are handled, and which lags/sectors are tested. For Leontief work, record the source/year of the transaction table, aggregation rules, coefficient construction, stability checks, and the source of each shock value.

The recommended expansion order is in [`RECOMMENDED_NEXT_STEPS.md`](RECOMMENDED_NEXT_STEPS.md). The overall current-state summary is in [`PROJECT_STATUS_STUDY_GUIDE.md`](PROJECT_STATUS_STUDY_GUIDE.md).

## 9. Sources within the project

- [Current manuscript](draft_manuscript.tex)
- [Research status](../docs/RESEARCH_STATUS.md)
- [Evidence audit](../docs/PHASE_VI_EVIDENCE_AUDIT.md)
- [Known limitations](../docs/LIMITATIONS.md)
- [Decision log](../docs/DECISIONS.md)
- [Data sources](../docs/DATA_SOURCES.md)
- [Research context / evidence inventory](context/CONTEXT.md)
- [Case market metrics](context/case_market_metrics.csv)
- [Experiment log](../results/experiment_log.json)

The literature citations in the manuscript provide context for established methods and prior work. Before submission, independently verify every bibliography entry and ensure that each general background claim is supported by the cited primary or authoritative source; the project’s own result files support the GeoSignal-specific claims, not the literature review.

