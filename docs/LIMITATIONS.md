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

## L-009: Dev Sample is Only 90 Days (2024 Q1)

**Scope:** Full pipeline

The initial dev sample covers 90 days. Granger causality tests require sustained
time series (ideally 2+ years for daily data at lag=12). Event study estimation
windows of 120 trading days may exceed the dev sample.

**Mitigation:** NSE index data is downloaded for 2022-2024 (full estimation window).
GDELT events are filtered to the dev period. Granger analysis will only be run
if sufficient observations are confirmed.
**Resolution path:** Expand to 2022-2024 for full analysis in October.
