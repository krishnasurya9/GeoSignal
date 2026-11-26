# Implementation status

GeoSignal is an offline Python system for processing geopolitical event records,
news metadata, exposure candidates, and market data. The application workflow
and its supporting modules are implemented. Two algorithm areas remain
unfinished: Granger/Toda-Yamamoto analysis and empirical Leontief
input-output propagation.

## Implemented components

- GDELT event and GKG metadata acquisition and preprocessing.
- Source-URL linkage and BM25/TF-IDF candidate retrieval.
- Entity extraction and rule-based sector mapping.
- Market data retrieval and event-study calculations.
- Registered case studies, cross-case summaries, and propagation maps.
- Actor network construction and a local Streamlit dashboard.
- Data verification and staged pipeline runners.

These components are available in the repository and can be run on local or
downloaded historical inputs. Their presence describes implementation status;
it does not imply that every method has been validated for prediction or causal
inference.

## Remaining algorithm work

### Granger / Toda-Yamamoto analysis

The analysis module and exploratory code exist, but a complete, validated
end-to-end algorithm and adequate time-series input are still outstanding.
The available 2024 Q1 event window is too short for the intended daily
geopolitical-risk time-series analysis.

### Empirical Leontief propagation

The propagation module and toy-matrix examples exist. The empirical algorithm
is incomplete until the project has a documented input-output matrix and a
validated mapping from event exposure to sector shocks. Toy or synthetic
matrices are illustrative only.

## Runtime boundary

The project currently processes archived data and writes local files. It does
not run a live ingestion service or provide an operational alerting endpoint.
The event/news, exposure, market, and case-summary outputs are generated under
`data/` and `results/`; those locations are ignored by Git by default.

## Main entry points

- `scripts/verify_data.py` — inspect local input availability.
- `scripts/run_pipeline.py` — run the staged data pipeline.
- `experiments/case_studies/run_pipeline.py` — calculate the six registered
  case summaries and maps.
- `run_all.py` — run the extended pipeline and generate local summaries.
- `dashboard.py` — browse local pipeline outputs.
