# GeoSignal — Data Sources Documentation

## Structured Geopolitical Events

### GDELT 1.0 Event Dataset (Primary)
- **Full name:** Global Database of Events, Language, and Tone — Event Export
- **Source:** http://data.gdeltproject.org/events/
- **Format:** Daily ZIP archives containing tab-separated CSV (58 columns)
- **Coverage:** 1979 – present (daily updates)
- **Dev sample:** 2024-01-01 to 2024-03-31 (~90 daily files)
- **Access:** Direct HTTP download, no authentication required
- **Taxonomy:** CAMEO event codes (4-digit) with CAMEO root (2-digit)
- **Goldstein Scale:** Provided directly in `GoldsteinScale` column where coded
- **Limitations:**
  - Machine-coded (BBN ACCENT), false positive rate ~15-20%
  - Duplicate event entries due to multiple news sources per event
  - CAMEO dictionary-based coding, limited recall for non-Western events
- **License:** Public domain / open access
- **Why selected:** Vast historical archive; direct URL linkage to news articles; well-documented schema; no authentication; same infrastructure as GKG

### UCDP GED (Supplementary — conflict only)
- **Full name:** Uppsala Conflict Data Program Georeferenced Event Dataset
- **Source:** https://ucdpapi.pcr.uu.se/api/gedevents
- **Format:** JSON via REST API
- **Coverage:** 1989 – present
- **Access:** REST API, no authentication
- **Scope:** Fatal organised violence only (narrower than GDELT)
- **Limitations:** Only conflict events; not suitable as primary source for trade/diplomatic events
- **Status:** Supplementary (not used in dev sample)

---

## Unstructured News / Text

### GDELT Global Knowledge Graph v1 (GKG)
- **Full name:** GDELT Global Knowledge Graph — Daily Flat Files
- **Source:** http://data.gdeltproject.org/gkg/
- **Format:** Daily ZIP archives containing tab-separated CSV (15 columns)
- **Coverage:** ~2013 – present (daily)
- **Access:** Direct HTTP download, no authentication required
- **Key fields:** Date, Themes, Persons, Organisations, Locations, Tone, Source URLs
- **CRITICAL LIMITATION:** GKG v1 flat files do NOT contain full article body text.
  The `raw_text` field in our news_corpus is a **proxy** constructed from:
  - V1Themes (GDELT theme taxonomy strings)
  - V1Persons (named persons mentioned)
  - V1Organisations (named organisations)
  - V1Locations (geographic locations)
  
  This proxy text enables BM25/TF-IDF retrieval but is NOT equivalent to reading
  the article. This limitation is explicitly acknowledged in all results and
  analysis. Reports must note this as a constraint on NLP quality.
  
- **Why selected:** Same infrastructure as GDELT events; co-referenced event IDs;
  no scraping required; covers the same temporal period; open access

---

## Outcome Data

### NSE Sectoral Indices (yfinance)
- **Full name:** National Stock Exchange of India — Sectoral Indices
- **Source:** Yahoo Finance API via `yfinance`
- **Tickers used:**
  - `^NSEI` — Nifty 50 (broad market benchmark)
  - `^CNXAUTO` — Nifty Auto
  - `^NSEBANK` — Nifty Bank
  - `^CNXFMCG` — Nifty FMCG
  - `^CNXIT` — Nifty IT
  - `^CNXMETAL` — Nifty Metal
  - `^CNXENERGY` — Nifty Energy (verify availability)
  - `^CNXPHARMA` — Nifty Pharma (verify availability)
- **Coverage:** 2022-01-01 to 2024-12-31 (for estimation + event windows)
- **Outcome variable:** Daily percentage return = (Close_t / Close_{t-1}) - 1
- **Non-trading days:** If event falls on weekend/holiday, the event window begins
  on the NEXT trading day. This rule is applied consistently.
- **Limitations:** Yahoo Finance data may have gaps; auto-adjust applied for splits/dividends

### FRED Macroeconomic Indicators
- **Source:** Federal Reserve Economic Data, St. Louis Fed
- **Access:** `fredapi` Python library, API key required
- **Series used:**
  - `DCOILWTICO` — WTI Crude Oil Spot Price (Daily)
  - `DCOILBRENTEU` — Brent Crude Oil Spot Price (Daily)
  - `CPIAUCSL` — US Consumer Price Index (Monthly)
  - `DEXINUS` — USD/INR Exchange Rate (Daily)
- **Note:** `BDIY` (Baltic Dry Index) was discontinued on FRED. Removed.
- **Coverage:** 2022-01-01 to 2024-12-31

---

## What We Are NOT Using

| Dataset | Reason |
|---|---|
| ICEWS | Proprietary ACCENT coder; restricted redistribution |
| GDELT 2.0 GKG via BigQuery | Requires GCP project/billing; not needed for flat CSV approach |
| All the News 2.0 (Kaggle) | Coverage only to 2020; not matching our 2024 dev period |
| CC-News (HuggingFace) | Very large; would require significant preprocessing |
| Live news scraping | Not permitted for this historical proof-of-concept |
