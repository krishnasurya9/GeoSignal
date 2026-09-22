# GeoSignal

**Geopolitical Event → Economic Impact Analysis Pipeline**

A reproducible research pipeline that links structured geopolitical event data to measurable downstream economic impacts on Indian sectoral markets, using NLP, event studies, Granger causality, and Leontief Input-Output propagation.

---

## 1. Overview

GeoSignal operationalises the research question:

> *Do geopolitical events (trade restrictions, shipping disruptions, diplomatic tensions, armed conflict) produce statistically detectable economic impacts in Indian sectoral markets?*

The pipeline works by:
1. Acquiring structured geopolitical events from GDELT 1.0
2. Retrieving associated news metadata from GDELT GKG v1
3. Aligning events to news articles via BM25 retrieval + TF-IDF cosine verification
4. Extracting industry/sector entities via spaCy NER + custom EntityRuler
5. Mapping entities to GICS sectors and NSE sectoral indices
6. Measuring market responses via MacKinlay (1997) event studies (CAR/AR)
7. Testing predictive precedence via Toda-Yamamoto Granger causality
8. Modelling cross-sector propagation via Leontief I-O analysis

---

## 2. Research Objective

```
Geopolitical event (GDELT structured)
    ↓ CAMEO coding + Goldstein intensity scoring
News/information response (GDELT GKG v1 metadata proxy)
    ↓ BM25 retrieval + TF-IDF cosine alignment
Sector exposure (spaCy NER + GICS keyword mapping)
    ↓ Entity → NSE sectoral index mapping
Market outcome (NSE sectoral index daily returns via yfinance)
    ↓ MacKinlay (1997) market model
Abnormal Returns (AR) + Cumulative Abnormal Returns (CAR)
    ↓ Toda-Yamamoto Granger causality
Temporal/causal analysis (predictive precedence only — not causation)
    ↓ Leontief I-O (I-A)⁻¹
Cross-sector propagation (illustrative — toy matrix pending real I-O data)
```

**Implemented:** All stages from data acquisition through integrated dataset assembly.  
**Partially implemented:** Leontief propagation (toy matrix — real India MoSPI / WIOD I-O data not yet integrated).  
**Planned:** Expanded 2022-2024 dataset (current dev sample is 90 days, Q1 2024 only).

---

## 3. Repository Structure

```
GeoSignal/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── settings.py              Central configuration (paths, windows, thresholds)
│   ├── entity_rules.jsonl       Custom spaCy EntityRuler patterns
│   ├── gics_taxonomy.csv        GICS sector reference
│   ├── nic2008_taxonomy.csv     NIC-2008 sector reference
│   └── cameo_goldstein_map.csv  CAMEO code → Goldstein score mapping
│
├── data/
│   ├── raw/                     Immutable downloaded files (GDELT ZIPs, GKG ZIPs)
│   ├── interim/                 Stage outputs not yet final
│   │   ├── event_news_pairs.parquet
│   │   └── event_news_sector.parquet
│   └── processed/               Final cleaned datasets consumed by analysis
│       ├── processed_events.parquet
│       ├── news_corpus.parquet
│       └── nse_returns.parquet
│
├── src/
│   └── geosignal/               Installable research package
│       ├── data/
│       │   ├── loader.py        GDELT 1.0 + UCDP GED acquisition
│       │   └── preprocessing.py Harmonization, Goldstein mapping, text cleaning
│       ├── nlp/
│       │   ├── news_loader.py   GDELT GKG v1 corpus acquisition
│       │   ├── alignment.py     BM25 + TF-IDF alignment (corpus-level vectorizer)
│       │   └── entity_extraction.py  spaCy NER + EntityRuler
│       ├── sectors/
│       │   └── sector_mapping.py     GICS keyword + label mapping
│       ├── markets/
│       │   └── market_data.py        NSE index download (yfinance) + FRED macro
│       ├── analysis/
│       │   ├── event_study.py        MacKinlay (1997) CAR / AR / t-test
│       │   ├── granger_causality.py  ADF + Toda-Yamamoto Granger tests
│       │   └── impact_analysis.py    Integrated dataset assembly + validation labels
│       └── propagation/
│           └── leontief.py           Leontief I-O shock propagation
│
├── scripts/
│   ├── run_pipeline.py          Full pipeline orchestrator (stage-skipping, --force)
│   ├── run_stage.py             Run a single named stage
│   ├── smoke_test.py            Fast in-memory smoke test (<60s)
│   └── verify_data.py           Read-only data quality diagnostic
│
├── experiments/
│   ├── case_studies/
│   │   ├── redsea.py            Red Sea / Houthi crisis (Dec 2023 – Q1 2024)
│   │   └── semiconductor.py     Semiconductor export controls (2022-2024)
│   └── exploratory/             Analysis scripts (01–07) for interactive investigation
│       ├── 01_data_exploration.py
│       ├── 02_event_processing.py
│       ├── 03_nlp_alignment.py
│       ├── 04_sector_mapping.py
│       ├── 05_event_study.py
│       ├── 06_granger_causality.py
│       └── 07_leontief_io.py
│
├── tests/
│   ├── conftest.py              Shared fixtures (tiny in-memory only)
│   ├── test_data.py             preprocessing + text cleaning
│   ├── test_nlp.py              NewsAligner + entity extraction
│   ├── test_sector_mapping.py   GICS mapping
│   └── test_analysis.py        EventStudy, Granger, Leontief, impact assembly
│
├── docs/
│   ├── DATA_SOURCES.md
│   ├── DECISIONS.md
│   └── LIMITATIONS.md
│
└── results/
    ├── figures/                 Generated plots
    ├── tables/                  Generated LaTeX / CSV tables
    ├── datasets/                Final integrated validation dataset
    ├── logs/                    Pipeline run logs
    └── experiment_log.json      Experiment tracking
```

---

## 4. Data

| File | Description | Size (approx) |
|------|-------------|---------------|
| `data/raw/gdelt_raw.parquet` | Raw GDELT 1.0 events (91 days, 2024 Q1) | ~913 MB |
| `data/raw/nse_sector_prices.parquet` | NSE sectoral index prices (2022-2024) | ~16 KB |
| `data/raw/gkg/*.zip` | GDELT GKG v1 daily ZIPs | ~2 GB |
| `data/processed/processed_events.parquet` | Harmonized canonical events | ~2.6 MB |
| `data/processed/news_corpus.parquet` | GKG metadata proxy corpus (53k articles) | ~68 MB |
| `data/processed/nse_returns.parquet` | Daily sectoral returns | ~18 KB |
| `data/interim/event_news_pairs.parquet` | BM25/TF-IDF aligned event–article pairs | generated |
| `data/interim/event_news_sector.parquet` | Sector-enriched pairs | generated |
| `data/processed/temporal_results.parquet` | Event study + raw impact results | generated |
| `results/datasets/integrated_validation_dataset.parquet` | Final research output | generated |

Raw data is not committed (see `.gitignore`). Large files are download-cached automatically.

---

## 5. Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For FRED macro data, add your API key to `.env` (copy from `config/.env.example`):
```
FRED_API_KEY=your_key_here
```

---

## 6. Running the Pipeline

### Data verification (read-only, no side effects)
```bash
python scripts/verify_data.py
```

### Smoke test (fast, in-memory fixtures, <60s)
```bash
python scripts/smoke_test.py
```

### Run the full pipeline (dev sample: 2024 Q1)
```bash
python scripts/run_pipeline.py --start 2024-01-01 --end 2024-03-31
```

### Run the full research window (2022-2024)
```bash
python scripts/run_pipeline.py --start 2022-01-01 --end 2024-12-31
```

### Skip expensive stages you've already run
```bash
python scripts/run_pipeline.py --skip-download   # reuse cached raw data
python scripts/run_pipeline.py --skip-nlp        # reuse existing event_news_pairs.parquet
```

### Force rerun of all stages
```bash
python scripts/run_pipeline.py --force
```

### Run a single stage
```bash
python scripts/run_stage.py --stage preprocess
python scripts/run_stage.py --stage nlp
python scripts/run_stage.py --stage temporal
python scripts/run_stage.py --stage impact
```

---

## 7. Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Download GDELT | `geosignal.data.loader` | network | `data/raw/gdelt_raw.parquet` |
| 2. Download GKG | `geosignal.nlp.news_loader` | network | `data/processed/news_corpus.parquet` |
| 3. Preprocess events | `geosignal.data.preprocessing` | `gdelt_raw.parquet` | `data/processed/processed_events.parquet` |
| 4. NLP alignment | `geosignal.nlp.alignment` | events + news | `data/interim/event_news_pairs.parquet` |
| 5. Sector mapping | `geosignal.sectors.sector_mapping` | pairs | `data/interim/event_news_sector.parquet` |
| 6. Market data | `geosignal.markets.market_data` | network | `data/processed/nse_returns.parquet` |
| 7. Temporal analysis | `geosignal.analysis.event_study` | pairs + returns | `data/processed/temporal_results.parquet` |
| 8. Impact assembly | `geosignal.analysis.impact_analysis` | all above | `results/datasets/integrated_validation_dataset.parquet` |

Each stage is independently rerunnable. Outputs from one stage are the inputs to the next.

---

## 8. Experiments

Case studies and exploratory analysis scripts are in `experiments/`:

```bash
# Red Sea / Houthi crisis
python experiments/case_studies/redsea.py

# Semiconductor export controls
python experiments/case_studies/semiconductor.py

# Exploratory analysis (requires pipeline outputs)
python experiments/exploratory/05_event_study.py
```

---

## 9. Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests use small in-memory fixtures only — no full corpus is loaded during testing.

---

## 10. Results

Generated outputs are written to:
- `results/figures/`  — matplotlib plots
- `results/tables/`   — CSV/LaTeX tables
- `results/datasets/` — integrated validation dataset
- `results/logs/`     — pipeline run logs
- `results/experiment_log.json` — experiment tracking

---

## 11. Reproducibility

- All parameters are centralised in `config/settings.py`
- Random seeds: NumPy and statsmodels use deterministic procedures; no random seed is currently required
- Experiment IDs are tracked in `results/experiment_log.json`
- Raw data is downloaded deterministically from public GDELT URLs
- Intermediate outputs are Parquet files with stable schemas

---

## 12. Research Methodology

**NLP alignment:** BM25Okapi retrieval over a ±3-day temporal window, filtered by TF-IDF cosine similarity (threshold = 0.05). The TF-IDF vectorizer is fitted once over the full corpus.

**Entity extraction:** spaCy `en_core_web_sm` with custom EntityRuler patterns prepended (SEMICONDUCTOR, ENERGY, LOGISTICS, AGRICULTURE, CRITICAL_MINERAL). NER runs on unmodified `raw_text` to preserve capitalisation.

**Sector mapping:** Deterministic keyword lookup with EntityRuler label priority. Known limitation: keyword matching, not ontological reasoning (L-008).

**Event study:** MacKinlay (1997) market model. OLS regression of sector returns on Nifty50 over a 120-trading-day estimation window. AR = actual − predicted. CAR = cumsum(AR). Significance: two-tailed t-test on CAR.

**Granger causality:** ADF stationarity pre-check. Toda-Yamamoto correction applied for non-stationary series (augments max_lag by integration order d). **Language requirement:** results must be described as *"geo_risk Granger-predicts sector_return"* — never as "causes."

**Leontief I-O propagation:** Leontief inverse L = (I − A)⁻¹. Shock propagation: Δx = L · Δd. **Current A matrix is a 5-sector toy matrix — illustrative only** (L-006, DEC-009). Replace with India MoSPI 2017-18 or WIOD tables before reporting empirical results.

---

## 13. Known Limitations

| ID | Scope | Description |
|----|-------|-------------|
| L-001 | NLP | GKG proxy text ≠ full article body. BM25/cosine scores reflect metadata overlap. |
| L-002 | Temporal | GDELT dates = media detection dates, not event occurrence dates (up to 2-day lag). |
| L-003 | Data | GDELT false positive rate ~15-20%; semantic duplicates may remain. |
| L-004 | Market | Yahoo Finance may have NSE data gaps. |
| L-005 | Goldstein | Root-code fallback is a coarse approximation. |
| L-006 | Leontief | Current I-O matrix is a toy — not empirical. |
| L-007 | Granger | Granger ≠ causation. Predictive precedence only. |
| L-008 | Sectors | Sector mapping is keyword-based, not ontological. |
| L-009 | Dev sample | 90-day dev sample is underpowered for Granger at lag=12. |

Full details: `docs/LIMITATIONS.md`

---

## 14. Current Status

| Component | Status |
|-----------|--------|
| GDELT 1.0 acquisition | ✅ Implemented |
| GKG v1 news corpus | ✅ Implemented |
| Event harmonization + Goldstein | ✅ Implemented |
| BM25 + TF-IDF alignment (corpus-level vectorizer) | ✅ Implemented |
| spaCy NER + EntityRuler | ✅ Implemented |
| GICS sector mapping | ✅ Implemented |
| NSE market data | ✅ Implemented |
| MacKinlay (1997) event study | ✅ Implemented |
| Toda-Yamamoto Granger causality | ✅ Implemented |
| Integrated validation dataset | ✅ Implemented |
| Leontief I-O propagation | ⚠️ Illustrative only (toy matrix) |
| Real India I-O data (MoSPI / WIOD) | 🔲 Planned |
| Full 2022-2024 dataset | 🔲 Planned |
| NetworkX graph layer | 🔲 Planned |
