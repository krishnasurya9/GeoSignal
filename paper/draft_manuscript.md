# GeoSignal: An Offline Historical Replay of a Proposed Geopolitical Monitoring Workflow

**Working manuscript — 7 October 2026**  
**Status:** Initial draft; venue, author details, and reference style not yet set.

## Abstract

Geopolitical developments can affect firms and markets through multiple, time-varying exposure channels, yet monitoring workflows must connect event records, reporting, economic entities, and market data before analysts can assess those channels. This paper examines whether a prototype designed for future live monitoring can exercise parts of that workflow on archived data. We replay two bounded 2024 cases using GDELT event records, GDELT Global Knowledge Graph (GKG) version 1 news metadata, direct source-URL matching, metadata-based lexical retrieval, rule-based sector labels, and descriptive market-model calculations. In the 12 January Red Sea case, 30 event records were linked to 31 same-source rows; lexical retrieval returned 230 candidates for 25 events, but only one exact-source pair appeared among those candidates. In the 12 March semiconductor export-controls case, the source article ranked 4,033rd under a baseline metadata query and 55th after query enrichment. Researcher-defined market baskets produced descriptive cumulative abnormal-return estimates, but neither basket assignment nor the estimates establish exposure or event causality. The replay demonstrates that archived records can be processed and traceable source links retained; it also exposes weak retrieval and unresolved exposure validation. The results do not establish live operational performance, semantic retrieval accuracy, predictive validity, or causal market effects.

**Keywords:** geopolitical risk; event data; news retrieval; historical replay; event study; exposure mapping

## 1. Introduction

Geopolitical events can reach firms and financial markets through trade restrictions, conflict, transport disruption, commodity supply, and other channels. Assessing those pathways from unfolding information requires more than detecting an event: an analytical workflow must connect structured event records to reporting, identify potentially exposed economic entities, and place those hypotheses alongside market observations. Geopolitical-risk indices and event datasets provide established ways to represent parts of this problem, while financial event studies examine market responses around dated information arrivals [CaldaraIacoviello2022; EventDataReference; MacKinlay1997]. The remaining practical challenge addressed here is how to assemble these stages into a traceable monitoring workflow and evaluate what a prototype can do with historical inputs.

We study GeoSignal, a prototype architecture intended for future live geopolitical monitoring. The empirical exercise is an offline replay using archived records; it is not a live deployment. We ask: **To what extent can the prototype connect historical geopolitical event records, news metadata, candidate economic exposures, and market context, and where do those connections fail?** This question focuses the paper on workflow feasibility and evidence quality rather than stock-impact prediction.

The replay covers two bounded cases: Red Sea-related event records dated 12 January 2024 and a semiconductor export-controls report dated 12 March 2024. It tests direct matching by normalized source URL, lexical candidate generation from GKG metadata, rule-based sector labeling, and descriptive market-model calculations on researcher-defined candidate baskets. These components produce different kinds of evidence. A URL match indicates that records refer to the same source URL; it does not verify the content or truth of an article. A lexical candidate indicates retrieval by metadata overlap; it is not a relevance judgment. A market-model estimate for an analyst-selected basket is descriptive and does not validate the event-to-security mapping.

The results show both an operational capability and a clear retrieval weakness. In the Red Sea case, source matching yielded rows for all 30 selected GDELT records, whereas lexical candidates contained only one exact-source pair and sector rules left 226 of 230 candidates unlabeled. In the semiconductor case, the exact article remained outside the top 50 after query enrichment. Market-response calculations can be produced for selected baskets, but those baskets were not generated or validated by the pipeline. Thus, the prototype can process historical records and preserve traceability while its semantic retrieval and exposure-linking stages remain unvalidated.

This paper makes three bounded contributions: (1) it documents an end-to-end historical replay of a proposed geopolitical monitoring workflow; (2) it reports source-linkage and retrieval diagnostics, including negative results; and (3) it separates descriptive market calculations from system-derived exposure claims and causal conclusions. The rest of the paper describes related work and the workflow, presents the replay design and findings, and discusses evidence gaps and next steps.

## 2. Background and related work

### 2.1 Geopolitical-risk measurement and event data

Geopolitical risk has been operationalized through news-based measures and structured event records, which offer complementary representations of conflict and political uncertainty [CaldaraIacoviello2022; PLOVER2023]. GeoSignal uses coded event records as one input to a proposed monitoring workflow. Those records are machine-coded observations of reported events, not a complete or error-free account of geopolitical activity. The present replay therefore treats them as input records and evaluates linkage behavior rather than assuming they are ground truth.

### 2.2 Linking events to reporting and economic exposure

Event extraction and temporal knowledge-graph research provide methods for representing entities, event types, and relations across time [MAVEN2020; MAVENERE2022; TIMERS2021; KnowEvolve2017; REN2020]. GeoSignal's current replay uses a narrower design: direct URL joins provide source traceability, while BM25/TF-IDF ranking over GKG metadata supplies additional candidates. This is a retrieval diagnostic, not a semantic event-news model evaluation. In particular, GKG v1 contains metadata fields rather than article bodies, limiting what lexical overlap can establish.

The intended downstream step is to connect event-related entities to sectors or securities. In the current prototype, sector labels rely on deterministic rules, and candidate market baskets are specified by the researcher. Consequently, the case study examines whether these links can be assembled and inspected; it does not establish that the system identifies the economically exposed firms.

### 2.3 Market response analysis

Event studies estimate security returns around dated information events relative to an expected-return model [MacKinlay1997]. Geopolitical-risk research has also examined heterogeneous responses across sectors and markets [Choi2024; HassanEtAl2022]. GeoSignal computes market-model estimates for selected baskets to illustrate how market data can be brought into the workflow. Because the baskets are not validated pipeline outputs, the event dates are not isolated interventions, and significance testing was not conducted for these case outputs, the estimates are reported as descriptive context only.

> **Drafting note:** Replace the citation keys with a verified bibliography and expand this section after selecting the target venue and scope. The reference-seed inventory marks some candidate works as unverified; do not cite those entries until bibliographic details are checked.

## 3. System and data

### 3.1 Proposed workflow

GeoSignal is organized as a sequence of analytical stages:

1. ingest structured geopolitical event records;
2. retrieve associated news records and retain direct source links where available;
3. generate additional lexical candidates from news metadata;
4. extract and map sector-related entities using deterministic rules; and
5. assemble market data for descriptive event-window analysis.

The architecture is intended to support future live monitoring. The evaluation in this paper replays archived records through selected stages and does not measure streaming latency, uptime, analyst utility, or live reliability. More ambitious temporal-prediction and input-output propagation modules are outside the evaluated results: no validated prediction panel or empirical propagation shock was produced for these cases.

### 3.2 Data sources and cases

The project archives daily GDELT event exports and GDELT GKG v1 metadata exports for 2024 Q1. GDELT event dates are media-detection dates and may differ from the dates on which underlying events occurred. GKG v1 provides metadata such as themes, named entities, locations, and tone; it does not provide full article text in this study.

The first case is a Red Sea/Houthi-related sample dated 12 January 2024. It contains 30 selected GDELT event records and uses a same-day GKG corpus with 108,343 usable metadata rows. The second case is a 12 March 2024 report concerning a U.S. assessment of expanded semiconductor export controls affecting China. Seven GDELT event-code records share the report's source URL; this represents repeated coded records linked to one article, not seven distinct articles.

### 3.3 Linkage and retrieval

For direct linkage, event and GKG source URLs are normalized and joined. The join is interpreted as source identity evidence only. Separately, lexical retrieval uses event/actor query text against GKG metadata. Red Sea results were evaluated against exact source-URL matches as a limited silver signal. In the semiconductor diagnostic, the exact report's rank was recorded under a baseline metadata query and an enriched query that added semiconductor, advanced-chip, export-control, and Commerce Department terms. A single-rater inspection of the baseline top 20 classified candidates as directly relevant, adjacent context, or unrelated using the article title/URL slug and GKG metadata; the labels are diagnostic and do not constitute a broad relevance benchmark.

### 3.4 Sector labels and market calculations

The sector stage applies deterministic entity and keyword rules to retrieved candidates. We report the resulting label counts, including Unknown, as a measure of rule coverage. The two market analyses use researcher-defined candidate baskets, adjusted-close histories, a 120-session pre-event estimation window, and post-event windows of 1, 3, 7, 14, and 30 trading days. A market model is estimated against a broad benchmark and cumulative abnormal return (CAR) is calculated for each window. Basket membership is an analyst-specified proxy, not an output of GeoSignal's event-to-sector/security mapping. The saved case analyses do not report significance tests; all CARs are therefore treated as descriptive and not causally attributed to the event.

## 4. Results

### 4.1 Red Sea case

The direct source-URL join produced 31 event/article rows covering all 30 selected event records. The metadata-based BM25/TF-IDF stage generated 230 candidate pairs for 25 of the 30 events. Against exact source URLs as the silver linkage signal, one exact-source pair appeared in the lexical candidates (0.43% of candidate pairs); the exact source appeared in the top ten for one of 30 events (3.33%). Combining direct links and lexical candidates yielded 260 unique event/article pairs and a path to a linked record for every selected event. These counts describe path coverage and exact-source overlap, not semantic relevance accuracy.

The saved mean and median cosine scores among matched pairs were 0.1818 and 0.1833, respectively, with 9.2 candidate pairs per matched event on average. Rule-based labels assigned Energy to two and Industrials to two of the 230 lexical candidates; the remaining 226 were Unknown. The separate exact-link rows had 29 Unknown and two Consumer Staples labels. An actor-only co-occurrence graph contained nine actor nodes and ten edges, with no sector nodes. These outputs do not supply a validated event-to-sector mapping.

For broad-market context, the Nifty 50 raw cumulative returns were +1.14%, +1.78%, −1.89%, +0.95%, and +2.19% over 1, 3, 7, 14, and 30 trading days, respectively. The market-model calculations for analyst-defined baskets are summarized below. CAR is the saved log-return approximation; all values are percentages.

| Candidate series | Benchmark | CAR at 1/3/7/14/30 trading days (%) |
|---|---|---|
| Ports/logistics | Nifty 50 | −1.77 / −2.74 / −0.11 / +6.70 / +5.58 |
| Energy | Nifty 50 | +0.58 / +3.30 / +3.90 / +13.86 / +17.78 |
| FMCG | Nifty 50 | −0.33 / −0.26 / −1.87 / −2.88 / −5.15 |

The corresponding raw cumulative returns for ports/logistics, energy, and FMCG at those windows were respectively (−0.44, −0.40, −0.66, +10.90, +14.48), (+1.90, +5.66, +2.78, +17.97, +26.67), and (+0.29, +0.66, −3.14, −2.74, −4.75) percent. Nifty benchmark returns were (+1.14, +1.78, −1.89, +0.95, +2.19) percent. The 120-session market-model estimates are single-case descriptive calculations without reported significance tests or causal interpretation. Basket membership is researcher-defined, including the FMCG constituent proxy used because the sector-index history was unavailable.

### 4.2 Semiconductor export-controls case

The exact-source join linked seven coded event records to one same-day GKG row carrying the article URL. In a same-day corpus of 120,718 metadata rows, the exact article ranked 4,033rd under the baseline TF-IDF query (cosine score 0.0692). Enriching the query moved it to rank 55, still outside the top 50. A single-rater review of the baseline top 20 classified 12 items as unrelated and eight as adjacent context; none was directly relevant under the case-specific rubric, giving direct-relevance precision@20 of 0.0 for this diagnostic sample. Because the review relied largely on titles, URL slugs, and metadata, it should be read as a small diagnostic, not a general precision estimate.

For market context, all case estimates used 120 pre-event trading sessions and are descriptive market-model CARs in percent (log-return approximation):

| Candidate/context series | Benchmark | CAR at 1/3/7/14/30 trading days (%) |
|---|---|---|
| Global semiconductor firms (NVDA, TSM, ASML, INTC) | S&P 500 | +1.36 / −2.16 / −5.08 / −4.18 / −13.94 |
| US semiconductor ETFs (SOXX, SMH; context only) | S&P 500 | +0.52 / −2.86 / −5.38 / −3.53 / −9.27 |
| Indian large-cap IT (indirect context) | Nifty 50 | +0.31 / +2.79 / −1.60 / −7.17 / unavailable |

The corresponding raw basket returns for the global semiconductor series were (+3.50, −0.55, −0.32, +2.46, −9.77) percent; S&P 500 benchmark returns were (+1.12, +0.64, +2.08, +2.46, −0.93) percent. For the ETF context series, raw returns were (+2.62, −1.40, −0.97, +2.33, −7.22) percent against the same benchmark. Indian IT raw returns were (+0.39, +2.12, −3.44, −5.29) percent over the first four windows; Nifty benchmark returns were (+0.01, −0.83, −2.21, +0.88) percent. No 30-day Indian IT estimate was present in the saved output. These series were not selected by GeoSignal, and the article did not enter the top 50 lexical candidates after query enrichment. The calculations therefore show market context for analyst-selected proxies, not that the pipeline identified exposed firms or that the report caused the observed returns.

### 4.3 Summary of evidence

Across both cases, direct source links retained traceability when metadata retrieval was weak. The lexical stage generated candidates, but its exact-source diagnostics were poor, and the sector rules left most Red Sea candidates unmapped. Descriptive market calculations were possible for researcher-defined baskets, but the workflow did not produce validated security exposure links. Taken together, the evidence supports a bounded historical processing demonstration and identifies retrieval and exposure mapping as unresolved stages.

## 5. Discussion

The replay clarifies what a prototype workflow can establish before it has a validated relevance set or exposure taxonomy. Direct URL matching is useful for preserving provenance when the same source appears in both event and news records, but it cannot establish that a report is relevant to an event or that its claims are accurate. Lexical retrieval broadens the candidate set, yet the two case diagnostics show that metadata overlap alone did not reliably rank the exact source near the top. In the Red Sea case, the exact source rarely appeared in lexical candidates; in the semiconductor case, query enrichment improved ranking substantially but did not place the article in the top 50.

The downstream results should be interpreted with the same care. Rule-based sector labels are sparse, while the candidate baskets used for return calculations were selected by the researcher. The calculated CARs are sensitive to the chosen basket and window and may include unrelated market movements. They are not evidence of a system prediction, exposure validation, or causal effect. Reporting these estimates alongside their provenance is useful for describing the current workflow, but a stronger market-impact claim requires pre-specified, independently reviewed exposure links, adequate comparison designs, uncertainty estimates, and multiple events.

These findings suggest a practical order for further evaluation: first construct a multi-event set with independent relevance judgements; next validate event-to-sector/security mappings against an explicit taxonomy; then compare retrieval and mapping variants on held-out cases; and only afterward evaluate market-response or predictive claims using appropriate benchmarks and uncertainty analysis. Full-text reporting, explicit event-time verification, and repeatable replay across event types would also address limits of the current metadata-only, two-case design.

## 6. Limitations

This study has several substantive limitations. First, GKG v1 is a metadata source, not article text, so retrieval scores do not measure semantic understanding of full reports. Second, the sample consists of two bounded cases; it is not a representative multi-event evaluation. Third, exact URL identity is only a limited linkage signal, and the semiconductor relevance review is single-rater and small. Fourth, deterministic sector rules have low observed coverage, and no event-to-security mapping is validated. Fifth, the market baskets are researcher-defined, the market calculations are descriptive, and the saved case analyses provide no significance test or basis for causal attribution. Finally, the paper evaluates archived replay only; it does not measure live operation, predictive performance, or empirical cross-sector propagation.

## 7. Conclusion

This paper reports an offline historical replay of selected stages in GeoSignal, a prototype proposed for future geopolitical monitoring. In two 2024 cases, direct source matching preserved traceable links, while metadata-based retrieval performed weakly against exact-source diagnostics and rule-based sector labeling remained sparse. Descriptive market-model estimates could be computed for analyst-defined baskets, but those baskets were not system-generated or validated. The current evidence therefore supports a limited processing-feasibility account and motivates stronger relevance, exposure, and market-impact evaluation before claims about reliable monitoring or prediction can be made.

## References

**To complete after venue and citation style are chosen.** Initial citation keys above point to items in `context/related_work_seeds.csv`; verify each entry and add the full bibliographic record before submission. Add primary references for GDELT/GKG, BM25, TF-IDF, event-study methodology, and the sector taxonomy actually used.

## Author and submission details to fill in

- Author names and affiliations
- Corresponding author details
- Target venue and its current template
- Funding, conflicts of interest, and data/code availability statements
- Verified references and final citation style
