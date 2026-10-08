# GeoSignal project context pack

Read-only inventory of the project and saved outputs as of 2026-10-05. This is a research/evidence map, not manuscript text. Each claim is tagged as **MEASURED** (present in a saved output or source artifact), **SCRIPTED-BUT-UNRUN** (implementation exists but a result was not produced for this project run), or **PLANNED** (proposal/document only). Do not treat code paths as evidence that a method ran.

## 1. Project question and scope

- **MEASURED:** The project proposes an architecture for live geopolitical monitoring and uses historical replay to exercise parts of it offline; it has not evaluated live operation. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md), [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json)
- **MEASURED:** Current runs establish archived event/news metadata processing, exact source-URL linkage, weak lexical candidate retrieval, limited rule-based labels, descriptive actor graph, and descriptive market-model computations for two case dates. They do not establish semantic relevance accuracy, validated exposure mapping, predictive accuracy, causal impact, or operational live reliability. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md), [`docs/PHASE_VI_EVIDENCE_AUDIT.md`](../../docs/PHASE_VI_EVIDENCE_AUDIT.md)
- **PLANNED:** The defensible paper question is whether a prototype can exercise a historical/offline version of the proposed live workflow, not whether a complete live service already works. A full live deployment is outside the saved evaluation. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md)

## 2. Architecture and implementation map

| Stage | Evidence status | Data flow / implementation | Evidence or limitation |
|---|---|---|---|
| Historical structured-event ingestion | MEASURED for case scripts; general downloader also exists | GDELT event ZIPs → harmonized event table. `src/geosignal/data/loader.py` (`download_gdelt_range`); `src/geosignal/data/preprocessing.py` (`harmonize_gdelt`, `clean_text`, `deduplicate_news`); case orchestration in `scripts/run_redsea_case.py` and `scripts/run_chip_controls_case.py`. | Case event CSV/Parquet outputs listed below. Event records are machine coded and can contain semantic duplicates. [`docs/DECISIONS.md`](../../docs/DECISIONS.md), [`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md) |
| News metadata ingestion | MEASURED for case scripts | GDELT GKG v1 daily metadata ZIP → harmonized rows. `src/geosignal/nlp/news_loader.py` (`download_gkg_range`, `_harmonize_gkg`). | GKG v1 contains metadata fields such as themes/entities/locations/tone, not article bodies. [`docs/DATA_SOURCES.md`](../../docs/DATA_SOURCES.md), [`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md) |
| Direct event–article source link | MEASURED | Normalize URL fields and join event source URLs to GKG URLs. Per-case logic in `scripts/run_redsea_case.py` / `scripts/run_chip_controls_case.py`; supporting evaluation/audit scripts. | Establishes same-source identity only; not semantic relevance or truth. Outputs: `results/datasets/redsea_exact_source_links_20240112.csv`, `chip_controls_exact_source_links_20240312.csv`. [`docs/DECISIONS.md`](../../docs/DECISIONS.md) |
| Lexical retrieval / alignment | MEASURED in Red Sea and chip-control case artifacts; methods differ slightly by experiment | `src/geosignal/nlp/alignment.py`: sparse corpus-level BM25 index and TF-IDF vectorization; `NewsAligner.retrieve`, `cosine_scores`, and `align_all`. Default config: `BM25_TOP_N=10`, `MIN_COSINE_SIM=0.05`, `NEWS_TEMPORAL_BAND=3`; TF-IDF uses 1–2 grams and max 20,000 features in class default. Red Sea case used 2024-01-12 same-day metadata only, so it did not exercise the configured ±3-day band. Chip rank diagnostic uses same-day TF-IDF metadata ranking and a separately enriched query. | Exact run metrics in §5. BM25/TF-IDF candidate coverage is not precision. No full article text or independent relevance labels for Red Sea. [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json), [`results/datasets/chip_controls_replay_summary_20240312.json`](../../results/datasets/chip_controls_replay_summary_20240312.json), [`config/settings.py`](../../config/settings.py) |
| Entity extraction | SCRIPTED-BUT-UNRUN as a validated case metric | `src/geosignal/nlp/entity_extraction.py`: `build_nlp_pipeline`, `_load_entity_patterns`, `extract_entities`; spaCy English small model plus project EntityRuler patterns in `config/entity_rules.jsonl`. The raw-text/cleaned-text split is intentional so NER retains capitalization. | No saved case-level entity extraction precision/recall or independent entity labels. [`docs/DECISIONS.md`](../../docs/DECISIONS.md) |
| Sector mapping | MEASURED for Red Sea lexical candidates only | `src/geosignal/sectors/sector_mapping.py`: `map_entity_to_gics`, `enrich_pairs_with_sectors`; configuration in `config/gics_taxonomy.csv`, `config/nic2008_taxonomy.csv`, and `config/entity_rules.jsonl`. Mapping is keyword/rule based. | Red Sea lexical results are 226 Unknown / 230. The chip case has no system sector-mapping result. Candidate exposure baskets are researcher-selected, not pipeline predictions. [`results/datasets/redsea_sector_mapping_counts_20240112.csv`](../../results/datasets/redsea_sector_mapping_counts_20240112.csv), [`results/datasets/case_exposure_proxy_register.csv`](../../results/datasets/case_exposure_proxy_register.csv) |
| Market data / returns | MEASURED for saved case price data and calculations | `src/geosignal/markets/market_data.py`: `download_nse_indices`, `compute_daily_returns`, `download_fred_series`; saved cases use Yahoo Finance adjusted-close series and case analysis scripts. | Red Sea broad Nifty context and both cases' candidate basket studies are descriptive. Market basket membership is researcher-defined, not a system-generated match. [`docs/DECISIONS.md`](../../docs/DECISIONS.md), output files in §5 |
| Event study | MEASURED for candidate-basket analyses; broad-market context also measured | `src/geosignal/analysis/event_study.py` (`compute_raw_impact`, `EventStudy`); case-specific market model calculations in `scripts/analyze_redsea_market_response.py` and `scripts/analyze_chip_controls_market_response.py`. | 120 pre-event trading-session estimation window; 1/3/7/14/30 trading-day windows; OLS market model; benchmarks and candidate series described in §4/§5. No significance testing or causal attribution. [`results/datasets/redsea_market_event_study_20240112.json`](../../results/datasets/redsea_market_event_study_20240112.json), [`results/datasets/chip_controls_market_event_study_20240312.json`](../../results/datasets/chip_controls_market_event_study_20240312.json) |
| Actor graph | MEASURED for Red Sea graph | `src/geosignal/graph/network.py` (`GeoSignalGraph.build`); saved actor co-occurrence graph in `results/datasets/redsea_case/network_nodes.csv` and `network_edges.csv`. | 9 actor nodes, 10 co-occurrence edges, zero sector nodes. No evidence of validated propagation or causal structure. [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json) |
| Integrated validation dataset | SCRIPTED-BUT-UNRUN as an evaluated result | `src/geosignal/analysis/impact_analysis.py` (`assign_validation_label`, `build_integrated_dataset`). | No validated exposure labels/evaluation output saved. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md) |
| Granger/time-series analysis | SCRIPTED-BUT-UNRUN for these cases | `src/geosignal/analysis/granger_causality.py` (`adf_stationarity`, `granger_causality_test`). | Current case logs mark it not run; a single event case does not support it. Granger prediction would not establish causation. [`docs/DECISIONS.md`](../../docs/DECISIONS.md), [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json) |
| Leontief propagation | SCRIPTED-BUT-UNRUN as an empirical result | `src/geosignal/propagation/leontief.py` (`LeontiefPropagator`); `src/geosignal/propagation/io_loader.py`. | Current 5-sector matrix is illustrative/toy, not empirical; no case propagation output. [`docs/DECISIONS.md`](../../docs/DECISIONS.md) |
| Forecasting, explanation, live operation | PLANNED | Described at a high level in the research plan/synopsis, not evidenced by the two saved case outputs. | No saved forecasts, explanation output, live feed evaluation, latency, uptime, or operational evaluation. [`GeoSignal Pipeline Research Plan.pdf`](../../GeoSignal%20Pipeline%20Research%20Plan.pdf), [`GeoSignal_Research_Paper_Synopsis.pdf`](../../GeoSignal_Research_Paper_Synopsis.pdf), [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md) |

## 3. Data inventory

| Dataset | Evidence status | Local source / range / rows | Known quality or interpretation limits |
|---|---|---|---|
| GDELT 1.0 Events | MEASURED | `data/raw/gdelt/`; 91 daily archive ZIPs covering 2024-01-01–2024-03-31. Case outputs: `redsea_events_20240112.csv` (30 rows; same case Parquet) and `chip_controls_events_20240312.csv` (7 event-code rows). | GDELT event date is media detection/reporting timing rather than guaranteed real-world occurrence time; machine coding and duplicate/ambiguous event records. [`docs/DATA_SOURCES.md`](../../docs/DATA_SOURCES.md), [`docs/DECISIONS.md`](../../docs/DECISIONS.md), case CSVs |
| GDELT GKG v1 | MEASURED | `data/raw/gkg/`; 91 daily ZIPs for same 2024-01-01–2024-03-31 interval. Red Sea same-day usable corpus 108,343 rows (`redsea_news_corpus_20240112.parquet`); chip-controls same-day metadata rows 120,718 (run summary). | Metadata proxy rather than full text; this limits retrieval and human relevance assessment. [`docs/DATA_SOURCES.md`](../../docs/DATA_SOURCES.md), `results/datasets/redsea_run_summary_20240112.json`, `results/datasets/chip_controls_replay_summary_20240312.json` |
| Exact source URL links | MEASURED | Red Sea: 31 event/article rows covering 30 selected events. Chip controls: 7 joined event-code rows to 1 unique same-day GKG article/source URL. | A shared URL demonstrates provenance/source identity, not semantic match quality. Chip rows are repeated event codes, not seven unique articles. [`results/datasets/redsea_retrieval_evaluation_20240112.json`](../../results/datasets/redsea_retrieval_evaluation_20240112.json), [`results/datasets/chip_controls_replay_summary_20240312.json`](../../results/datasets/chip_controls_replay_summary_20240312.json) |
| NSE / US market data | MEASURED | Saved case prices requested for 2023-06-01–2024-04-30. Red Sea Nifty series: `redsea_nifty50_prices_20230601_20240430.parquet`, 224 observations; candidate security CSV/Parquet also 224 rows. Chip union of candidate series: `chip_controls_candidate_prices_20230601_20240430.csv` / Parquet, 237 date rows. Sector-index history: `historical_nse_sector_indices_20230601_20240430.csv` / Parquet, 224 rows. | Yahoo Finance/NSE data availability gaps: ^CNXFMCG history unavailable (Red Sea FMCG constituent proxy used); HPCL.NS missing and excluded; TATAMOTORS.NS missing. Chip histories mix US and India session calendars; 30-day Indian IT output is absent. [`results/datasets/redsea_market_event_study_20240112.json`](../../results/datasets/redsea_market_event_study_20240112.json), [`results/datasets/chip_controls_market_event_study_20240312.json`](../../results/datasets/chip_controls_market_event_study_20240312.json) |
| Old processed/archive outputs | MEASURED but stale; exclude from current case result reporting | `data/processed/archive_stale_20261005/`: `news_corpus.parquet` 53,795 rows, `nse_returns.parquet` 254 rows, `processed_events.parquet` 67,623 rows. | Marked stale in local research status; date coverage does not match current case inputs. Do not use as the current two-case corpus/results. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md) |
| GPR/FRED / other optional sources | PLANNED or unused for these results | `src/geosignal/markets/market_data.py` contains FRED helper; configuration has optional credentials but they are not used in saved case outputs. | Not part of reported metrics. Do not include secrets or credentials in the paper/context pack. [`docs/DATA_SOURCES.md`](../../docs/DATA_SOURCES.md), [`config/settings.py`](../../config/settings.py) |

## 4. Methods and run parameters actually used

- **Red Sea event selection — MEASURED:** event date 2024-01-12; GDELT root code `19`; URL keyword filter `houth|red.?sea|yemen|shipping|ship|vessel|suez|cargo`; deduplicate on GDELT source URL; maximum 30 selected event rows; one same-day GKG corpus. [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json)
- **Red Sea text retrieval — MEASURED:** sparse corpus-level BM25 followed by corpus-fitted TF-IDF cosine threshold; case corpus is event-day only, despite a default config temporal band of 3 days. Configured top-N 10, cosine threshold 0.05, TF-IDF 1–2 grams, max features 20,000. The case metric output records retrieval pairs/coverage but does not make them human-relevance scores. [`src/geosignal/nlp/alignment.py`](../../src/geosignal/nlp/alignment.py), [`config/settings.py`](../../config/settings.py), [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json)
- **Chip controls retrieval — MEASURED:** on 2024-03-12, same-day GKG metadata TF-IDF ranking; compare baseline event query with enriched terms for semiconductor, advanced chips, restricted exports, and Commerce Department. Top-50 retained; exact source rank measured against all same-day metadata rows. [`results/datasets/chip_controls_replay_summary_20240312.json`](../../results/datasets/chip_controls_replay_summary_20240312.json)
- **Entity extraction — SCRIPTED-BUT-UNRUN as an independently evaluated measure:** spaCy English small model + custom EntityRuler patterns; no saved precision/recall labels for the case. [`src/geosignal/nlp/entity_extraction.py`](../../src/geosignal/nlp/entity_extraction.py), [`config/entity_rules.jsonl`](../../config/entity_rules.jsonl)
- **Sector mapping — MEASURED for Red Sea candidate metadata only:** configured GICS/NIC/entity rules with keyword signals. 226 of 230 lexical candidate rows are Unknown; two Energy and two Industrials. Exact-source linked rows separately have 29 Unknown and 2 Consumer Staples in run summary. This is not a validated exposure taxonomy. [`src/geosignal/sectors/sector_mapping.py`](../../src/geosignal/sectors/sector_mapping.py), [`results/datasets/redsea_run_summary_20240112.json`](../../results/datasets/redsea_run_summary_20240112.json)
- **Event study — MEASURED as descriptive estimate:** OLS market model estimated on 120 trading sessions before event; event windows are 1, 3, 7, 14, 30 trading days. Red Sea uses ^NSEI; chip global semiconductors and ETFs use ^GSPC, Indian IT indirect context uses ^NSEI. Candidate baskets were manually selected in `scripts/create_case_exposure_register.py`; no system-generated or independently validated event-security assignments. The exact output metric column is `market_model_CAR_pct_log_approx`. [`scripts/analyze_redsea_market_response.py`](../../scripts/analyze_redsea_market_response.py), [`scripts/analyze_chip_controls_market_response.py`](../../scripts/analyze_chip_controls_market_response.py), [`results/datasets/redsea_market_event_study_20240112.json`](../../results/datasets/redsea_market_event_study_20240112.json), [`results/datasets/chip_controls_market_event_study_20240312.json`](../../results/datasets/chip_controls_market_event_study_20240312.json)
- **Actor graph — MEASURED:** Red Sea NetworkX actor co-occurrence graph; graph has actor nodes only, no sector nodes. [`src/geosignal/graph/network.py`](../../src/geosignal/graph/network.py), [`results/datasets/redsea_case/network_nodes.csv`](../../results/datasets/redsea_case/network_nodes.csv), [`results/datasets/redsea_case/network_edges.csv`](../../results/datasets/redsea_case/network_edges.csv)
- **Methods not run:** Granger, empirical Leontief propagation, end-to-end forecast, sector-level causal/event attribution, model ablations, multi-case statistical evaluation, and live system evaluation. Code exists for some analysis helpers, but saved case run summaries mark key analyses not run. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md), [`docs/DECISIONS.md`](../../docs/DECISIONS.md)

## 5. Saved case results (every value below is from a saved output)

### Red Sea / Houthi coverage — event date 2024-01-12

| Metric | Saved value | Output source |
|---|---:|---|
| Selected structured event records | 30 | `results/datasets/redsea_retrieval_evaluation_20240112.json` |
| Usable same-day GKG metadata rows | 108,343 | `results/datasets/redsea_run_summary_20240112.json` |
| Exact URL join rows / events covered | 31 / 30 of 30 (100.0%) | `results/datasets/redsea_retrieval_evaluation_20240112.json` |
| Lexical candidate event–article rows / events with candidates | 230 / 25 of 30 (83.33% candidate coverage) | `results/datasets/redsea_retrieval_evaluation_20240112.json` |
| Exact same-source URL among lexical candidates | 1 of 230 (0.43% by saved metric); URL in top 10 for 1 of 30 events (3.33% recall@10) | `results/datasets/redsea_retrieval_evaluation_20240112.json` |
| Hybrid union pairs / events covered | 260 / 30 of 30 (100.0%) | `results/datasets/redsea_retrieval_evaluation_20240112.json` |
| Similarity diagnostic | mean cosine 0.1818; median 0.1833; 9.2 pairs per matched event | `results/datasets/redsea_run_summary_20240112.json` |
| Sector labels on lexical pairs | Unknown 226; Energy 2; Industrials 2 | `results/datasets/redsea_sector_mapping_counts_20240112.csv`; `redsea_run_summary_20240112.json` |
| Separate exact-link sector labels | Unknown 29; Consumer Staples 2 | `results/datasets/redsea_run_summary_20240112.json` |
| Nifty 50 (^NSEI) raw cumulative return | 1d +1.1426%; 3d +1.7790%; 7d -1.8866%; 14d +0.9544%; 30d +2.1936% | `results/datasets/redsea_market_context_20240112.json` |
| Actor graph | 9 nodes (all actors; 0 sector nodes), 10 actor co-occurrence edges | `results/datasets/redsea_run_summary_20240112.json`; `results/datasets/redsea_case/network_nodes.csv`; `network_edges.csv` |

Market model aggregate rows: exact saved numeric strings for all 29 aggregate/window combinations from both case outputs are also indexed in [`case_market_metrics.csv`](case_market_metrics.csv), with the original output path in each row. The original Red Sea CSV has 80 rows (individual constituents and portfolio aggregates); benchmark is ^NSEI. The original chip CSV has 60 rows (individual securities and series aggregates). Estimation window is 120 pre-event trading sessions. The summary tables below are rounded to 2 decimals; use the indexed CSV/original outputs for exact values. All are descriptive, have no significance test, and use researcher-defined baskets.

| Candidate basket | Trading days | Basket raw return % | Nifty raw return % | CAR % (log approximation) |
|---|---:|---:|---:|---:|
| Ports/logistics | 1 | -0.44 | 1.14 | -1.77 |
| Ports/logistics | 3 | -0.40 | 1.78 | -2.74 |
| Ports/logistics | 7 | -0.66 | -1.89 | -0.11 |
| Ports/logistics | 14 | 10.90 | 0.95 | 6.70 |
| Ports/logistics | 30 | 14.48 | 2.19 | 5.58 |
| Energy | 1 | 1.90 | 1.14 | 0.58 |
| Energy | 3 | 5.66 | 1.78 | 3.30 |
| Energy | 7 | 2.78 | -1.89 | 3.90 |
| Energy | 14 | 17.97 | 0.95 | 13.86 |
| Energy | 30 | 26.67 | 2.19 | 17.78 |
| FMCG | 1 | 0.29 | 1.14 | -0.33 |
| FMCG | 3 | 0.66 | 1.78 | -0.26 |
| FMCG | 7 | -3.14 | -1.89 | -1.87 |
| FMCG | 14 | -2.74 | 0.95 | -2.88 |
| FMCG | 30 | -4.75 | 2.19 | -5.15 |

### Chip export-controls news — event/report date 2024-03-12

| Metric | Saved value | Output source |
|---|---:|---|
| Event-code records sharing VOA URL | 7 | `results/datasets/chip_controls_replay_summary_20240312.json` |
| Same-day GKG corpus / exact article rows / exact URL join rows | 120,718 / 1 / 7 | `results/datasets/chip_controls_replay_summary_20240312.json` |
| Selected event record | ID 1163145352, CAMEO code 0015, root 01, NumArticles 1324 | `results/datasets/chip_controls_replay_summary_20240312.json` |
| Baseline exact article rank / cosine | 4,033 / 0.06919658281630936 | `results/datasets/chip_controls_replay_summary_20240312.json` |
| Enriched exact article rank | 55 (outside top 50) | `results/datasets/chip_controls_replay_summary_20240312.json` |
| Single-rater baseline top-20 labels | 12 unrelated, 8 adjacent, 0 directly relevant; direct relevance precision@20 = 0.0 | `results/datasets/chip_controls_relevance_audit_summary_20240312.json` |

Market model CSV is `results/datasets/chip_controls_market_model_event_study_20240312.csv` (60 rows; includes individual securities and series aggregates), estimation window 120 trading sessions, windows 1/3/7/14/30. For aggregates below, percentages are rounded to 2 decimals; [`case_market_metrics.csv`](case_market_metrics.csv) and the source CSV are authoritative for exact values. No significance test/causal attribution; the event day is a report date which may not equal first public information time.

| Candidate/context series | Benchmark | Window (trading days) | Basket raw return % | Benchmark raw return % | CAR % (log approximation) |
|---|---|---:|---:|---:|---:|
| Global semiconductor firms (NVDA, TSM, ASML, INTC) | ^GSPC | 1 | 3.50 | 1.12 | 1.36 |
| Global semiconductor firms | ^GSPC | 3 | -0.55 | 0.64 | -2.16 |
| Global semiconductor firms | ^GSPC | 7 | -0.32 | 2.08 | -5.08 |
| Global semiconductor firms | ^GSPC | 14 | 2.46 | 2.46 | -4.18 |
| Global semiconductor firms | ^GSPC | 30 | -9.77 | -0.93 | -13.94 |
| US semiconductor ETFs (SOXX, SMH; context only) | ^GSPC | 1 | 2.62 | 1.12 | 0.52 |
| US semiconductor ETFs | ^GSPC | 3 | -1.40 | 0.64 | -2.86 |
| US semiconductor ETFs | ^GSPC | 7 | -0.97 | 2.08 | -5.38 |
| US semiconductor ETFs | ^GSPC | 14 | 2.33 | 2.46 | -3.53 |
| US semiconductor ETFs | ^GSPC | 30 | -7.22 | -0.93 | -9.27 |
| Indian large-cap IT (indirect context) | ^NSEI | 1 | 0.39 | 0.01 | 0.31 |
| Indian large-cap IT | ^NSEI | 3 | 2.12 | -0.83 | 2.79 |
| Indian large-cap IT | ^NSEI | 7 | -3.44 | -2.21 | -1.60 |
| Indian large-cap IT | ^NSEI | 14 | -5.29 | 0.88 | -7.17 |
| Indian large-cap IT | ^NSEI | 30 | — | — | unavailable in saved output |

## 6. Figures, tables, and generated artifacts

- **MEASURED existing figure:** `results/figures/redsea_case/network_graph.png` (actor-only graph; regenerate if graph node/edge data or styling changes). Source graph CSVs: `results/datasets/redsea_case/network_nodes.csv` and `network_edges.csv`.
- **MEASURED output tables/data:** all saved CSV/Parquet metric sources named in §5. There are no files under `results/tables/` at inventory time.
- **SCRIPTED-BUT-UNRUN / regenerate if needed:** case scripts can regenerate outputs and overwrite their output artifacts; do not rerun over saved evidence unless outputs have been backed up. `scripts/run_redsea_case.py`, `scripts/run_chip_controls_case.py`, retrieval/relevance audit scripts, and market-response scripts are the sources for those outputs.
- **PLANNED figures/tables:** no saved end-to-end architecture diagram, multi-case comparison figure, retrieval performance chart with judged relevance labels, uncertainty/significance table, or validated exposure table was found. Create only after methods/evaluation are settled.

## 7. Decisions, limitations, and evidence audit

- **MEASURED/documented decisions:** GDELT 1.0 is the structured-event source; GKG v1 metadata is used as a news proxy; date is GDELT detection/reporting date; supplied Goldstein values are used with a coarse fallback; preserve raw text for NER and cleaned text for retrieval; NSE sector indices are intended as market outcomes; Granger is conditional on adequate stationary data and is predictive precedence, not causation; no GNN in initial prototype; current Leontief matrix is a five-sector toy. [`docs/DECISIONS.md`](../../docs/DECISIONS.md)
- **MEASURED/documented limitations:** no article bodies; reporting-date lag; machine-coded semantic noise; missing Yahoo/NSE series; coarse Goldstein fallback; toy input-output matrix; keyword mapping false positives; short development interval underpowered for Granger; daily index returns measure investor expectations, not physical output. [`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md)
- **MEASURED/documented status:** Red Sea exact linkage and hybrid candidate generation complete; retrieval low against URL-identity proxy; sector mapping weak; broad Nifty and descriptive candidate-basket models exist; semantic evaluation, sector event study as pipeline output, Granger, empirical I-O, and live operation not evaluated. Chip case has weak baseline/enriched ranks and single-rater top-20 review. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md)
- **MEASURED/documented evidence audit:** the pasted Phase VI claims about freight, ports, rice, energy flows, physical production, or detailed semiconductor supply-chain impacts are not outputs of the saved run; keep them separate from project metrics and validate their independent sources before use. The report’s case CARs are descriptive basket calculations, not effects caused by the event. [`docs/PHASE_VI_EVIDENCE_AUDIT.md`](../../docs/PHASE_VI_EVIDENCE_AUDIT.md)

## 8. Related-work seeds

See [`related_work_seeds.csv`](related_work_seeds.csv). Titles seeded from the synopsis/work plan are only starting points. **VERIFIED** means a direct publisher/proceedings/official record was checked for title, authors, year, and venue; items with incomplete source metadata or only a vague project-topic pointer remain **UNVERIFIED**. The verified references have URLs recorded in the CSV. In particular, generic ICEWS and supply-chain/graph references in the source workbook need exact bibliographic identification before citation.

## 9. Reproducibility snapshot

- Python runtime available to this workspace: 3.12.14. The project metadata allows Python >=3.11. [`pyproject.toml`](../../pyproject.toml)
- Declared packages/ranges: see [`requirements.txt`](../../requirements.txt); no dependency lock file was found in the repository root inventory. Environment package versions beyond the Python version have not been pinned in this context pack.
- Git commit at inspection: `618675ca251dd516d7af68f673d7a4e36c15083b`.
- Run from repository root `A:\projects\paper\geo\GeoSignal` in PowerShell. The workspace runtime executable was `C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`; the `python` command itself was not found in the active shell during this inspection. After installing dependencies and the spaCy `en_core_web_sm` model (where needed), use these exact commands:
  - Red Sea case: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_redsea_case.py`
  - Red Sea URL-link evaluation: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/evaluate_redsea_retrieval.py`
  - Red Sea market model: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/analyze_redsea_market_response.py`
  - Chip case: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/run_chip_controls_case.py`
  - Chip top-20 audit: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/create_chip_relevance_audit.py`
  - Chip market model: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/analyze_chip_controls_market_response.py`
  - Candidate exposure proxy register: `& 'C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/create_case_exposure_register.py`
- These commands are reproduction instructions from local status/docs and script outputs, not a claim that they were rerun during this inspection. They may need network access for historical market downloads and may overwrite existing result files. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md), script files listed above.

## 10. Paper-evidence gaps

- No multi-event/multi-date replay across event types to establish repeatability. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md)
- No independent, pre-specified relevance labels; current chip audit is single-rater, top-20, and largely based on URL title/slug plus metadata. [`results/datasets/chip_controls_relevance_audit_summary_20240312.json`](../../results/datasets/chip_controls_relevance_audit_summary_20240312.json)
- No validated event → sector/company exposure taxonomy; the two market-basket studies use researcher-selected baskets. [`results/datasets/case_exposure_proxy_register.csv`](../../results/datasets/case_exposure_proxy_register.csv)
- No reported uncertainty/significance tests, suitable benchmark comparisons, temporal ablation, or baseline system comparison for the current event/news/impact workflow. [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md)
- No reliable semantic retrieval result: exact URL reference retrieval is a limited silver signal; Red Sea metrics are weak, and chip exact article remains rank 55 after query enrichment. [`results/datasets/redsea_retrieval_evaluation_20240112.json`](../../results/datasets/redsea_retrieval_evaluation_20240112.json), [`results/datasets/chip_controls_replay_summary_20240312.json`](../../results/datasets/chip_controls_replay_summary_20240312.json)
- No pipeline-produced validated impact labels or evidence that the event caused the observed basket returns. [`docs/PHASE_VI_EVIDENCE_AUDIT.md`](../../docs/PHASE_VI_EVIDENCE_AUDIT.md)
- No full-text news ingestion, sector-event study from system-derived mappings, empirical I-O data, live streaming/latency/uptime evaluation, or live deployment. [`docs/DECISIONS.md`](../../docs/DECISIONS.md), [`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md), [`docs/RESEARCH_STATUS.md`](../../docs/RESEARCH_STATUS.md)
- Venue template, page/layout/citation rules, and forms remain unconfirmed; see [`../template/README.md`](../template/README.md).

## Source inventory note

The researcher-provided plan and synopsis are [`GeoSignal Pipeline Research Plan.docx`](../../GeoSignal%20Pipeline%20Research%20Plan.docx), [`GeoSignal Pipeline Research Plan.pdf`](../../GeoSignal%20Pipeline%20Research%20Plan.pdf), [`GeoSignal_Research_Paper_Synopsis.pdf`](../../GeoSignal_Research_Paper_Synopsis.pdf), and [`GeoSignal_Research_Paper_Synopsis.xlsx`](../../GeoSignal_Research_Paper_Synopsis.xlsx). The plan/synopsis describe broader proposed stages than the two measured case runs; do not use their planned stages as completed results.
