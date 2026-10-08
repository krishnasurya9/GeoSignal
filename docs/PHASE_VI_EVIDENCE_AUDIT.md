# Phase VI Evidence Audit

## Overall finding

The pasted Phase VI text is a proposed case-study narrative, not a report of
results currently produced by this repository. Several numerical statements
are presented as if GeoSignal measured or validated them, but their datasets,
analysis code, and outputs are not present in the current run. Keep those
statements out of the empirical-results section until the source data and
calculations are supplied and reproduced.

The paper's intended system remains a live geopolitical-monitoring architecture.
Historical replay is the offline method used here to test parts of that
architecture; it does not require a live deployment.

## Evidence produced by the current offline replay

The Red Sea case dated 2024-01-12 currently supports these repository-backed
findings:

- 30 structured GDELT event records were selected; the same-day GKG v1 input
  contained 108,343 usable news-metadata records.
- Exact normalized source-URL joins yielded 31 event/article rows covering all
  30 selected events. This establishes source identity linkage, not article
  meaning or truth.
- BM25/TF-IDF yielded 230 candidate pairs for 25/30 events. Against exact source
  URLs as a limited reference signal, one exact source pair appeared in the
  lexical list (0.43% of pairs; top-10 event recall 3.33%). Candidate coverage
  must not be called retrieval accuracy.
- The union of direct source links and lexical candidates contains 260 unique
  event/article pairs covering all 30 events.
- The actor-only graph contains 9 nodes and 10 edges.
- Sector mapping on the lexical candidates leaves 226/230 pairs Unknown; four
  receive low-confidence Energy or Industrials labels.
- Market output is raw Nifty 50 cumulative returns for several post-event
  windows. It is broad-market context, not sector-specific abnormal return or
  evidence of causation.

See `results/datasets/redsea_retrieval_evaluation_20240112.json`,
`results/datasets/redsea_run_summary_20240112.json`, and
`docs/RESEARCH_STATUS.md` for the run details.

## Case Study A: Red Sea logistics crisis

The local GDELT/GKG archives support a historical event/news replay for
2024-01-12. The current outputs do not measure shipping routes, Indian cargo
dependence, port throughput, canal tolls, freight prices, marine insurance,
rice exports, crude flows, or Brent prices. They also do not calculate abnormal
returns for NIFTY FMCG, MSMEs, or any affected sector index.

The pasted numerical claims—including 15–20% cargo dependence, a 50% Suez
throughput decline, a 40% toll-revenue decline, 1,000–3,000% freight increases,
the 2.17% rice-shipment increase, the 45% rice-market share, and quoted crude
freight prices—are not outputs from this repository. Their numbered citations
must be connected to the full bibliography and the claims must be independently
checked before use. The current pipeline must not be described as having
detected, quantified, or validated these outcomes.

An initial source check finds that some pasted figures appear to mix metrics or
periods. UNCTAD reports the Red Sea/Suez disruption contributed 148 percentage
points to a cumulative 120% rise in the China Containerized Freight Index from
October 2023 to June 2024; its February 2024 briefing reports average Shanghai
container spot rates up 122% from early December, and Shanghai–Europe rates up
256% ([UNCTAD freight update](https://unctad.org/es/conference/ntfc-global-forum-2022/news/high-freight-rates-strain-global-supply-chains-threaten-vulnerable-economies),
[UNCTAD rapid assessment](https://unctad.org/publication/navigating-troubled-waters-impact-global-trade-disruption-shipping-routes-red-sea-black)).
Singapore's Ministry of Transport reports war-risk insurance rising from below
0.1% of insured vessel value before the attacks to about 0.5–1.0% as of
February 2024 ([official parliamentary reply](https://www.mot.gov.sg/news-resources/newsroom/written-reply-to-parliamentary-question-on-impact-of-houthi-attacks-on-marine-insurance-premiums/)).
Egypt's State Information Service reports Suez traffic down 30% since the start
of 2024 and foreign-exchange revenues down 40% specifically between January 1
and 11 ([official report](https://sis.gov.eg/en/media-center/news/sca-chief-situation-in-red-sea-may-exacerbate-co2-emissions/));
this does not validate a 50% first-quarter throughput figure. USDA says the
export restrictions on non-basmati white and broken rice represented nearly
45% of India's own rice exports in calendar 2022, not 45% of global rice trade
([USDA FAS report](https://www.fas.usda.gov/data/india-india-bans-export-non-basmati-white-rice)).
The 2.17% rice-shipment figure and 15–20% cargo-dependence figure remain
unverified from the material available here. These independently sourced facts
may be used as historical background with their precise metric and period; they
must remain separate from GeoSignal outputs.

The pasted discussion of Saudi Arabia's 1,200-km East–West pipeline, its
temporary shutdown, Yanbu inventory duration, and sharply different crude
freight quotes also appears to combine later events with the 2024 Red Sea case.
AP's reporting on the pipeline shutdown is dated September 14, 2026 and places
it amid the 2026 Iran war and attacks on Saudi infrastructure ([AP report](https://apnews.com/article/8c18d82c109a8ea91347ce53c0096c53)).
Those 2026 observations must not be presented as outcomes of the January 2024
offline replay. The event date and source date need to be recorded for every
case-study statistic before the narrative is reused.

Safe use in the paper: cite independently verified historical facts as case
background, then report the system's actual event selection, source links,
candidate retrieval, actor graph, sector labels, and broad-market context as
separate prototype results.

## Case Study B: semiconductor export-controls news

A bounded replay now exists for 2024-03-12, using GDELT event records and GKG
v1 metadata around a VOA report on the US assessing expanded chip export
controls affecting China's military. Seven GDELT event-code records share the
same article URL; there is one same-day GKG metadata row and seven normalized
URL joins. This demonstrates event/source linkage and preserves traceability,
not seven independent articles or seven confirmed geopolitical events.

The baseline same-day TF-IDF query ranked the exact article 4,033rd among
120,718 metadata rows. An enriched query that added semiconductor, advanced
chip, restricted-export, and Commerce Department terms moved it to rank 55,
still outside top 50. A single-rater review of the baseline top 20 found 12
unrelated and 8 adjacent-context candidates, none directly relevant under the
event-specific rubric. The audit mostly uses URL title/slug and GKG metadata;
it is a diagnostic, not a general retrieval-accuracy result. GKG v1 is not
full article text.

Historical market data support a descriptive market-model calculation for an
expert-defined basket of NVDA, TSM, ASML, and INTC against the S&P 500. The
estimated CAR is +1.36% over 1 trading day, -5.08% over 7 days, and -13.94%
over 30 days. These returns are not generated by GeoSignal's event/news/sector
mapping, have no significance test, and cannot be attributed to the report.
The 30-day window is exposed to substantial unrelated-news confounding.

The replay therefore demonstrates that archived records and prices can pass
through source-linkage and descriptive market-analysis steps. It does not
validate semantic news ranking, semiconductor exposure mapping, production
effects, inflation, water-use/ZLD/UPW pathways, NIFTY Auto effects, or Leontief
propagation. The toy input-output matrix remains illustrative. Do not claim
statistically significant CAR or industrial outcomes.

## What is needed before those claims can become empirical results

- Full bibliographic entries for the numbered citations in the pasted text.
- A dated, reproducible source dataset for each claimed outcome (shipping,
  freight/insurance, trade/export, energy, semiconductor, production, and
  equity data as applicable).
- Code linking each event/news output to the relevant sector/security and
  outcome series, with a documented event window and comparison baseline.
- Independent relevance/exposure labels or a documented expert review for
  event/news and sector/security matches. The initial Case B top-20 review is
  single-rater and metadata/title based; it is not sufficient as the final
  evaluation protocol.
- A real input-output table and empirically justified shocks before reporting
  Leontief propagation.

When writing from current results, frame them as an offline historical replay
of a prototype designed for future live use. Do not claim live deployment or
reliable stock-impact prediction from this single case.
