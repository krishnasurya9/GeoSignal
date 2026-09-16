# GeoSignal — Design Decisions Log

Format per entry:
DECISION | WHY | ALTERNATIVES CONSIDERED | REASON FOR CHOICE | LIMITATIONS

---

## DEC-001: Primary Structured Event Source = GDELT 1.0 (not ICEWS, not UCDP)

**DECISION:** Use GDELT 1.0 as the single primary structured event source for the proof-of-concept.

**WHY:** The system needs a broad geopolitical event dataset covering trade restrictions, diplomatic events, conflicts, and supply chain disruptions — not just fatal violence.

**ALTERNATIVES:**
- ICEWS: Proprietary coder; event dictionaries withheld; redistribution restrictions
- UCDP GED: Only fatal violence; too narrow for economic impact research
- GDELT + ICEWS combined: Conflation risk; complicates initial proof-of-concept

**REASON FOR CHOICE:** GDELT covers the widest event taxonomy; open access; same infrastructure as GKG (our news source); co-referenced event IDs; well-documented CAMEO codes

**LIMITATIONS:** GDELT has high false-positive rate (~20% duplicates); machine-coded; recall limited for non-Western events; dictionary-based coding

---

## DEC-002: News Corpus = GDELT GKG v1 Flat Files (not full article scraping)

**DECISION:** Use GDELT GKG v1 metadata as a proxy for news text.

**WHY:** The pipeline requires unstructured news associated with events. Scraping live URLs is (a) forbidden for this historical POC and (b) many URLs from 2024 return 404.

**ALTERNATIVES:**
- Scraping SOURCEURL from GDELT events: ~40% URL survival rate for 2024 articles
- All the News 2.0 (Kaggle): Only covers to 2020; temporal mismatch with dev sample
- CC-News (HuggingFace): Very large; complex pipeline to subset by date/topic
- GDELT GKG v2 via BigQuery: Requires GCP billing account

**REASON FOR CHOICE:** GKG v1 daily flat files require no authentication; co-referenced with GDELT events; cover the same temporal period; provide structured metadata (themes, persons, orgs) usable for BM25/TF-IDF retrieval

**LIMITATIONS:** GKG v1 does NOT contain full article body text. The raw_text proxy (themes + persons + orgs + locations) is a metadata-based representation. This limits NER quality (no full article context) and BM25 precision. This is explicitly documented as a methodology constraint and must be acknowledged in the paper.

---

## DEC-003: Event Study Anchored to Geopolitical Event Date (not article date)

**DECISION:** Use `event_date` from `processed_events.parquet` (GDELT event timestamp) as the anchor for all temporal windows — NOT the news article publication date.

**WHY:** The research question concerns the economic effect of the geopolitical event itself. News coverage may lag by hours or days. Using article_date would misalign the measurement window.

**ALTERNATIVES:** Using article_date (simpler, already in pairs table) — rejected because methodologically incorrect.

**REASON FOR CHOICE:** MacKinlay (1997) event study methodology explicitly requires the event date as the reference point. GDELT assigns a date to the event it detects.

**LIMITATIONS:** GDELT event dates are detection dates (when the news source reported the event), not necessarily when the event physically occurred. For breaking events, this may be the same day. For slow-developing situations, the distinction matters. We document this as a known limitation.

---

## DEC-004: Goldstein Scale — Use GDELT Supplied Value, Fall Back to Root-Code Map

**DECISION:** Use `GoldsteinScale` directly from GDELT where available. Fall back to a root-code lookup table only when the field is empty.

**WHY:** GDELT includes a machine-assigned Goldstein Scale value per event. This is more precise than a root-code aggregate.

**LIMITATIONS:** GDELT Goldstein values are assigned by the ACCENT coder, not by the canonical Goldstein (1992) table. Minor discrepancies exist. The fallback root-code map is a coarse approximation.

---

## DEC-005: Text Pipeline — Separate raw_text from cleaned_text

**DECISION:** Maintain two text representations: `raw_text` (original, for NER) and `text` (cleaned, for BM25/TF-IDF retrieval).

**WHY:** spaCy NER depends heavily on capitalisation and punctuation to identify proper nouns (organisations, persons, locations). Lowercasing before NER destroys this signal. The existing `clean_text()` function must NOT be applied before NER.

**ALTERNATIVES:** Single cleaned text — rejected (documented bug in original code)

**REASON FOR CHOICE:** Standard NLP pipeline practice. NLTK/spaCy documentation explicitly warns against lowercasing before NER.

---

## DEC-006: Outcome Metric = NSE Sectoral Index Daily Returns

**DECISION:** Use daily percentage returns of NSE sectoral indices as the primary outcome variable for Indian market impact validation.

**WHY:** Sector indices aggregate company-level effects into a single measurable time series. High-frequency (daily) data allows precise temporal alignment. NSE is the relevant market for the India-focused case studies.

**ALTERNATIVES:**
- Individual stock returns: Too noisy; survivorship bias risk
- Commodity prices: Less market-specific; harder to link to specific event types
- Trade volume data: Monthly frequency; insufficient for short-window event studies

**LIMITATIONS:** Sectoral indices are float-weighted; they reflect market expectations, not physical economic activity. A significant CAR means investors responded, not necessarily that production was affected.

---

## DEC-007: Granger Causality — Only Run if Stationarity and Data Volume Justify It

**DECISION:** Run Granger causality tests with ADF pre-check and Toda-Yamamoto adjustment only if: (a) sufficient observations exist (>4× max_lag), (b) ADF test results are interpretable.

**WHY:** Granger causality tests are invalid on non-stationary series without correction. With a 90-day dev sample, the daily return series may have insufficient observations at lag=12.

**LIMITATIONS:** Granger causality identifies predictive precedence, not causal mechanism. All results must be described as "Granger-predictive" not "causal." This constraint is enforced throughout the codebase.

---

## DEC-008: No Full GNN / Graph Model in Initial Implementation

**DECISION:** Use NetworkX for relationship representation and optional graph-derived features. No Graph Neural Network in the September proof-of-concept.

**WHY:** GNN requires sufficient labelled training data and a validated graph schema. Neither exists at this stage. Adding a GNN without empirical justification would be over-engineering.

**PLAN:** If the event study and Granger analysis produce validated results, the graph layer can be introduced in October as an extension.

---

## DEC-009: Leontief I-O Matrix = Illustrative Toy Matrix (not real data)

**DECISION:** The `LeontiefPropagator` in `impact_analysis.py` uses a 5-sector toy matrix for demonstration only.

**WHY:** Real I-O tables (WIOD, OECD ICIO, India MoSPI) require licensing and significant preprocessing. Integrating them correctly is a separate research task.

**PLAN:** Replace toy matrix with India-specific I-O data in the October extension. Document the toy matrix clearly in all outputs.
