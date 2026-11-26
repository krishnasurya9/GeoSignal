# GeoSignal

GeoSignal is a Python research system for connecting geopolitical events,
related reporting, economic exposure, and market data. It processes archived
data and produces traceable analysis outputs for historical case studies.

The current implementation is an offline system. It does not consume live news
or provide production alerts. The end-to-end data and case-study workflows are
implemented. The two unfinished algorithm areas are Granger/Toda-Yamamoto
analysis and data-backed Leontief input-output propagation. Exploratory code
and toy examples exist for both, but they are not complete empirical
implementations.

## Capabilities

- Ingest and preprocess GDELT event records and GDELT GKG news metadata.
- Link records by source URL and retrieve additional event/news candidates
  with BM25 and TF-IDF.
- Extract entities, map candidate exposures to sectors, and build actor graphs.
- Retrieve market data and calculate event-study returns for registered cases.
- Produce case summaries, figures, and a local dashboard for generated outputs.

The GKG source contains metadata rather than full article bodies. Sector and
security exposure requires review, and event-study results are descriptive.
See [System Status](docs/IMPLEMENTATION_STATUS.md) and
[Limitations](docs/LIMITATIONS.md) before using results.

## Architecture

```text
GDELT events ─────┐
                  ├─> preprocessing and event/news linkage
GDELT GKG ────────┘              │
                                 v
                   entity and sector exposure
                                 │
                                 v
                    market data and event study
                                 │
                                 v
                   case outputs, figures, dashboard
```

Reusable code lives in `src/geosignal/`. Shared settings and taxonomies are in
`config/`. Case definitions are centralized in
`experiments/case_studies/case_registry.py`.

## Get started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional FRED access uses `FRED_API_KEY`. Copy `config/.env.example` to `.env`
and add the key if needed. Keep credentials out of source control.

## Run

Check which inputs are available locally:

```bash
python scripts/verify_data.py
```

Run the six registered historical cases:

```bash
python experiments/case_studies/run_pipeline.py
```

Run the broader staged data pipeline:

```bash
python scripts/run_pipeline.py --start 2024-01-01 --end 2024-03-31
```

The broader pipeline can download data and may take longer. Use `--skip-download`
to reuse local raw data or `--force` to regenerate stage outputs.

Run the extended end-to-end pipeline, including graph generation and the run
summary:

```bash
python run_all.py --skip-download
```

To browse local generated outputs with Streamlit:

```bash
python -m pip install streamlit
streamlit run dashboard.py
```

Run the unit tests with:

```bash
python -m pytest
```

The first run may require downloading source data. Raw and generated data are
excluded from Git by default; see `docs/DATA_SOURCES.md` for the sources and
local input expectations.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/geosignal/` | Reusable pipeline and analysis modules |
| `config/` | Settings, taxonomy tables, and entity rules |
| `scripts/` | Pipeline stages, data checks, and case-specific utilities |
| `experiments/case_studies/` | Case registry and historical case runners |
| `experiments/exploratory/` | Numbered exploratory analysis scripts |
| `data/` | Local raw, interim, and processed data; ignored by Git |
| `results/` | Generated datasets, figures, and logs; mostly ignored by Git |
| `docs/` | Data sources, research decisions, limitations, and evidence audit |
| `tests/` | Unit tests for reusable components |

## Research notes

GeoSignal currently replays historical records rather than evaluating a live
service. Exact source links establish traceability, not article relevance.
Candidate exposure labels and security baskets require independent review.
Market windows do not establish that an event caused a return. Granger results
must be described as predictive precedence, not causation. Leontief results
require a documented empirical input-output matrix; toy data are illustrative.

See [System Status](docs/IMPLEMENTATION_STATUS.md),
[Data Sources](docs/DATA_SOURCES.md), [Design Decisions](docs/DECISIONS.md),
and [Limitations](docs/LIMITATIONS.md) for details.
