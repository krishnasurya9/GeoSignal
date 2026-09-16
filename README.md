# GeoSignal: Geopolitical Event → Economic Impact Pipeline

A rigorous, empirical pipeline that links structured geopolitical event data to measurable downstream economic impacts using NLP, event studies, Granger causality, and Leontief Input-Output modeling.

---

## Architecture Overview

```
Raw Geopolitical Data (GDELT/ICEWS/UCDP)
        ↓
  Data Harmonization + CAMEO/Goldstein Mapping
        ↓
  Unstructured News Corpus (BM25 + TF-IDF Alignment)
        ↓
  NER + EntityRuler (spaCy) → Sector Mapping (GICS / NIC-2008)
        ↓
  Outcome Data (NSE Nifty Indices via yfinance, FRED Macroeconomics)
        ↓
  Temporal Alignment → Event Study (CAR / Abnormal Returns)
        ↓
  Granger Causality (VAR / Toda-Yamamoto)
        ↓
  Leontief I-O Propagation → Integrated Validation Dataset
```

## Project Structure

```
GeoSignal/
├── data/
│   ├── raw/          # Immutable original datasets
│   └── processed/    # Harmonized, deduplicated outputs
├── notebooks/        # Sequential exploration notebooks
├── src/              # Reusable Python modules
├── results/          # Figures, tables, experiment logs
├── config/           # API keys, parameters, taxonomy maps
└── tests/            # Unit tests
```

## Phases

| Phase | Description |
|-------|-------------|
| I     | Structured geopolitical data acquisition (GDELT primary, ICEWS/UCDP supplementary) |
| II    | Unstructured news NLP (BM25, TF-IDF, spaCy NER + EntityRuler) |
| III   | Sectoral taxonomy (GICS, NIC-2008) + outcome data (yfinance, FRED) |
| IV    | Temporal alignment + Event Study (MacKinlay 1997) + Granger Causality |
| V     | Integration → Validated event-impact dataset |
| VI    | Case studies: Red Sea Crisis, Semiconductor Export Controls |

## Quick Start

```bash
pip install -r requirements.txt
python src/data_loader.py       # Phase I: Download GDELT data
python src/preprocessing.py     # Phase II: Clean + align news
python src/sector_mapping.py    # Phase III: Map entities to GICS/NIC
python src/temporal_analysis.py # Phase IV: Event study + Granger
python src/impact_analysis.py   # Phase V: Build validation dataset
```
