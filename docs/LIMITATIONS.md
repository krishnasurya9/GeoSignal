# GeoSignal — Known Limitations

This document is maintained throughout the project. Every significant limitation
must be recorded here before it is reported in any results table or paper section.

---

## L-001: News Text is Metadata Proxy, Not Full Article Body

**Scope:** `src/news_loader.py`, all NLP outputs

GDELT GKG v1 flat files do not contain full article body text.
The `raw_text` field is constructed from:
- V1Themes: GDELT taxonomy strings
- V1Persons: Named persons
- V1Organisations: Named organisations
- V1Locations: Geographic locations

This proxy text enables retrieval but is NOT equivalent to reading the article.
Consequences:
- BM25 and TF-IDF scores reflect metadata overlap, not true semantic alignment
- NER entity extraction is limited to what GDELT already identified
- Context-dependent economic signals may be missed

**Mitigation:** Document in all results tables. Label BM25/cosine scores as "metadata similarity."
**Resolution path:** Full-text retrieval from source URLs (future work).

---

## L-002: GDELT Event Dates Are Detection Dates, Not Occurrence Dates

**Scope:** `src/data_loader.py`, `src/temporal_analysis.py`

GDELT assigns the date when it detected an event in news media.
For events with media lag (e.g., sanctions announced late evening, coverage next day),
the GDELT date may be 1-2 days after the actual event.

**Mitigation:** Use GDELT date consistently as the reference point. Document that event windows
are anchored to media detection date. For key case studies, verify event date manually.

---

## L-003: GDELT False Positive Rate ~15-20%

**Scope:** `src/data_loader.py`, `src/preprocessing.py`

GDELT machine coding generates a substantial rate of false positives and duplicate
event records. Deduplication by `GlobalEventID` removes exact duplicates but not
semantic duplicates (same event, different source article → multiple records).

**Mitigation:** Deduplicate by `event_id`. For analysis, aggregate events by date × event_root
rather than treating each record as a unique event.

---

## L-004: NSE Tickers via Yahoo Finance May Have Data Gaps

**Scope:** `src/sector_mapping.py`

Yahoo Finance is not the official NSE data source. Historical data may have:
- Missing trading days
- Incorrect adjusted prices for some periods
- Discontinued index tickers

**Mitigation:** Validate each ticker's data coverage before analysis. Document gaps.
For missing data, apply `dropna(how="all")` per series.

---

## L-005: Goldstein Scale Fallback is a Root-Code Approximation

**Scope:** `src/preprocessing.py`

When GDELT does not supply a Goldstein value, we fall back to a root-code → score
mapping. This mapping assigns a single score to an entire root category (e.g., all
root-20 events = -10.0), losing sub-code nuance.

**Mitigation:** Use GDELT's supplied GoldsteinScale wherever available.
Document fallback usage rate in preprocessing output.

---

## L-006: Leontief I-O Propagation Uses a Toy Matrix

**Scope:** `src/impact_analysis.py`

The LeontiefPropagator currently uses a 5-sector illustrative matrix.
Results from this model are ILLUSTRATIVE ONLY and must not be presented
as empirical findings.

**Mitigation:** Label all Leontief outputs as "illustrative" or "schematic."
**Resolution path:** Integrate India MoSPI I-O table (2017-18) or WIOD in October.

---

## L-007: Granger Causality ≠ Causation

**Scope:** `src/temporal_analysis.py`, all analysis outputs

Granger causality tests predictive precedence only. A significant Granger result
means that past values of geopolitical risk are informative for forecasting sectoral
returns — it does NOT establish a causal mechanism.

**Mitigation:** All code outputs and paper sections must use language like:
"geo_risk Granger-predicts sector_return" — never "geo_risk causes sector_return."

---

## L-008: Sector Mapping is Rule-Based Keyword Matching

**Scope:** `src/sector_mapping.py`

The current entity → GICS sector mapping uses deterministic keyword lookup.
This is a starting approximation, not a sophisticated ontology.

False positive risks:
- "bank" → Financials (but "river bank" or "data bank" would also match)
- "food" → Consumer Staples (but "food for thought" would also match)

**Mitigation:** EntityRuler ensures NER labels (SEMICONDUCTOR, ENERGY, LOGISTICS, etc.)
are applied to specific multi-word patterns before keyword lookup.
Single-word keyword mapping is a fallback.

---

## L-009: GDELT Event Archive is Q1 2024 Only (90 days)

**Scope:** Full pipeline, Granger causality, case study event counts

The structured GDELT event archive used in this study covers 90 days
(01 Jan – 31 Mar 2024). This affects two things differently:

**Event study (not materially affected):**
The MacKinlay (1997) market model estimation window uses NSE daily returns,
which are downloaded for 2022–2024 via yfinance (independent of the GDELT
archive). All six case studies have 120+ trading days of return history
available before their anchor dates. Event study results are valid within
the cases selected.

**Granger / Toda-Yamamoto (materially affected — not run as a main result):**
Granger causality tests on a daily geo-risk signal require a sustained
time series (typically 2+ years at lag=12 for meaningful power). The
90-day GDELT window is insufficient. The Granger stage in run_all.py
logs an explicit warning and produces no result when observations fall
below 4 × max_lag. A longer 2022–2024 daily panel would support a
pre-specified Toda-Yamamoto analysis and is left for future work.

**Case coverage (adequate for the paper's claim):**
Six cases are registered across all four strata (A/B/C/D), drawing on
the 2024 Q1 archive. The paper's claim is workflow feasibility, not
population-level inference. Six varied cases with honest null reporting
are sufficient for that claim. The Q1 archive is not expanded to avoid
adding unvalidated material under time pressure.

**Mitigation (current):**
- Event study estimation windows are satisfied by the yfinance return history.
- All six cases are anchored within the 90-day GDELT window.
- Granger results are suppressed and documented as insufficient (not omitted silently).
- The cross-case summary table (cross_case_summary.csv / .tex) reports
  all results including null and insufficient outcomes.

**Resolution path (future work):**
Expand the GDELT + GKG archive to 2022–2024 using batch_downloader.py
(`python run_all.py --full --start 2022-01-01 --end 2024-12-31`).
This will enable the Toda-Yamamoto Granger analysis with sufficient power
and extend the case study pool. Leave one sentence in the paper:
"A longer 2022–2024 daily panel would support a pre-specified temporal
analysis (Toda-Yamamoto) and is left for future work."

---

## L-010: Six Cases Are Not a Random Sample

**Scope:** External validity of case study findings

The six registered cases were selected by a deterministic protocol
(case_registry.py §2) after writing inclusion criteria — not after
looking at returns. However, they are not a random sample of geopolitical
events: they were chosen because they have auditable real-world anchors,
clear India exposure channels, and sufficient GDELT volume. Events that
left no trace in the 2024 Q1 archive are necessarily excluded.

**Consequence:** Results (confirmed or null) apply to these six cases.
They do not generalise to the population of geopolitical events, to other
markets, or to other time periods.

**Mitigation:** The paper should state: "Results are reported for six
pre-selected cases satisfying the inclusion protocol. They do not
constitute a representative sample and should not be interpreted as
population-level estimates."

---

## L-011: Propagation Map Layer 2 Uses Deterministic Country Centroids

**Scope:** src/geosignal/visualization/propagation_map.py

The "directly affected countries" layer plots affected_iso2 codes at
hand-curated centroids (COUNTRY_CENTROIDS dict in propagation_map.py).
These are capital-city or geographic-centre coordinates, not
trade-flow-weighted impact centroids. The circle sizes are scaled by
mean |Goldstein| of GDELT events near the anchor date, which is a
media-coverage proxy, not a verified economic exposure measure.

**Mitigation:** The map is a visualisation of the pipeline's reasoning
chain, not a claim about actual geographic impact. Label all map
figures accordingly: "Pipeline-detected propagation pathway
(GDELT-sourced · indicative only)."

---

## L-012: Leontief I-O Results Are Illustrative Unless WIOD CSV Is Supplied

**Scope:** Stage 11 of run_all.py, all case study Leontief figures

Unless the WIOD 2016 India CSV or a MoSPI I-O table is provided via
--io-matrix or auto-downloaded by io_loader.py, the Leontief
propagation uses a synthetic matrix estimated from MoSPI 2022-23 GDP
shares. While the synthetic matrix satisfies the Hawkins-Simon
stability condition and has realistic structural priors, it is not an
empirically derived transaction table.

**Mitigation:** All Leontief outputs are labelled with the matrix source
in the figure title and in leontief_shocks.csv. The paper should note:
"Leontief propagation results are illustrative; the A matrix is
estimated from sectoral GDP shares pending integration of the official
MoSPI 2017-18 I-O table. Same for the case study Leontief figures."
