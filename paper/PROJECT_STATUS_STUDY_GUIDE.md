# GeoSignal paper: project status study guide

**Prepared:** 7 October 2026  
**Purpose:** A plain-language record of what the project has actually done, what remains incomplete, why those choices were made, and what the current evidence can support.

This guide is based on the saved code, run summaries, metrics, and manuscript in this project. It distinguishes **measured** work (there is a saved output), **partial/diagnostic** work (some evidence exists but does not validate the method), and **planned/not run** work. Code that exists is not treated as a result.

## 1. What the paper is currently about

GeoSignal is a prototype for a proposed workflow that would monitor geopolitical events, connect them to news and economic entities, and show relevant market context. The work completed so far is an **offline historical replay**: archived data from 2024 are processed after the fact. It is not a live monitoring service, and the study does not show that GeoSignal can predict which stocks will move.

The defensible paper question is:

> How far can this prototype connect historical event records, news metadata, possible economic exposures, and market observations—and where do those connections fail?

That framing makes the weak retrieval and uncertain exposure links useful findings. It does not hide them or claim that the full proposed architecture has been validated.

## 2. What has been done

### Data and case runs

- The project has 91 daily GDELT event archives and 91 GDELT GKG v1 metadata archives for **1 January–31 March 2024**.
- Two bounded historical cases have saved outputs: Red Sea-related records on **12 January 2024**, and semiconductor export-controls reporting on **12 March 2024**.
- The market-price request covers **1 June 2023–30 April 2024**. Price histories were downloaded from Yahoo Finance through `yfinance`; the run manifest records 5 October 2026 as the download date.
- GDELT GKG v1 supplies metadata (themes, entities, locations, tone), not the article body. The study therefore tests matching and retrieval over metadata, not full-text understanding.

### Case 1: Red Sea, 12 January 2024

- Selected 30 structured GDELT event records and read 108,343 usable same-day news-metadata rows.
- Exact normalized source-URL matching created 31 event/article rows covering all 30 selected event records. This shows traceability to a shared URL; it does not verify that an article is relevant or true.
- BM25/TF-IDF produced 230 additional candidate pairs for 25 of the 30 events (83.33% candidate-generation coverage). Only one pair shared an exact source URL with the event record: 0.43% of candidate pairs; the exact source was in the top ten for 1 of 30 events (3.33%). These are linkage diagnostics, **not** relevance accuracy.
- Combining direct source links and lexical candidates produced 260 unique event/article pairs and at least one path for all 30 selected records.
- Rule-based labels left 226 of 230 lexical candidates as **Unknown**. Four were labelled Energy or Industrials, but those few labels are not validated exposures.
- The actor graph has 9 nodes and 10 edges. It is descriptive actor co-occurrence, not an economic propagation network.
- Broad Nifty 50 returns and analyst-defined candidate baskets were calculated. For example, the 30-trading-day market-model CARs were +5.58% for the ports/logistics candidate basket and +17.78% for the energy basket. These are descriptive estimates for researcher-selected baskets, without significance tests or causal attribution.

### Case 2: semiconductor export-controls report, 12 March 2024

- Seven GDELT event-code rows shared a VOA article URL. The same-day GKG input had 120,718 metadata rows and one matching article row. Seven event codes are not seven separate articles.
- The exact article ranked 4,033rd in the baseline same-day TF-IDF search. An enriched query moved it to rank 55, still outside the top 50.
- A single-reviewer look at the baseline top 20 found 12 unrelated candidates, 8 adjacent-context candidates, and no directly relevant candidate. This is a small diagnostic, not a reliable general precision estimate; judgments relied mostly on titles/URLs and GKG metadata.
- For a researcher-selected global semiconductor basket, the 7- and 30-day market-model CAR estimates were -5.08% and -13.94%. The basket was not selected by GeoSignal, the event-to-company exposure was not validated, and unrelated news can affect these long windows.

### Manuscript work

- The open draft is a generic two-column article, since there is no target journal or conference yet. It can be converted to a venue template after one is chosen.
- The draft identifies the temporal Granger and Leontief stages as proposed but not run, and describes the market estimates as descriptive.
- The author block includes P. Krishna Surya, School of Advanced Sciences, VIT-AP University, Amaravati, Andhra Pradesh 522241, India, and the supplied email address.
- Wide tables were changed to span both columns where appropriate. The built-in LaTeX compiler currently stops with **“Unable to find standard directories for platform”** before it processes the source, so the revised table layout has not been visually checked in a compiled PDF.

## 3. What is partial, weak, or not done

| Component | Status | Why the current evidence is limited |
|---|---|---|
| Event-to-source linkage | Measured in both cases | URL identity is useful provenance, but matching a URL does not assess article meaning, event truth, or economic impact. |
| News retrieval | Measured as a diagnostic | GKG is metadata, not full text. Retrieval is weak against the exact-source proxy, and the only manual review is small and single-rater. No broad labeled relevance set exists. |
| Sector/company exposure | Partial | The rule mapper leaves almost all Red Sea candidates unknown. Price baskets were chosen by the researcher, not generated and validated by the pipeline. |
| Market response | Descriptive calculations exist | Only two bounded cases; exposure is not validated; no significance/uncertainty analysis or causal design is reported. A 30-day window can include unrelated events. |
| Actor graph | Descriptive output exists for Red Sea | Nine actors and ten links do not establish supply-chain propagation or sector impact. |
| Toda–Yamamoto/temporal analysis | **Not run** | The event/news archive covers only 90 days, and the project has no validated aligned daily geopolitical-risk and sector-return panel. One or two event dates cannot stand in for a long time series. The existing function also needs methodological review before its output could be called a valid Toda–Yamamoto result. |
| Leontief input–output propagation | **Not run as an empirical result** | The available five-sector matrix is explicitly illustrative. There is no validated empirical input–output matrix plus defensible event-to-sector shock vector in the saved run. Synthetic or toy coefficients could only support a clearly labeled demonstration. |
| Live operation / forecasting | **Not evaluated** | The run replays archived data; it has no live-feed latency, uptime, alert quality, or forecast evaluation. |
| Independent validation | **Not complete** | There is no multi-event test set with independent relevance/exposure labels and reviewer agreement. |
| Reproducibility statements | **Still open** | The draft asks for the code repository/availability decision and final software versions. Funding and conflict-of-interest statements need author confirmation. References and final venue style also need review. |

## 4. Why these choices were made

### Why use an offline replay?

The available evidence is archived. Replaying it makes the workflow inspectable and repeatable without claiming a live service that was never evaluated. This is a scope decision that keeps the paper aligned with the data actually available.

### Why use GDELT and URL matching?

The project already has public GDELT event and GKG archives. Direct source URL joins provide a transparent provenance check. Lexical ranking was added to find possible additional reporting beyond exact links. The results show the strengths and weaknesses of both paths: URLs preserve traceability, while metadata-based retrieval misses even the exact case article in the semiconductor example.

### Why report weak retrieval and unknown sector labels?

They are the measured outcome. Calling candidate coverage “accuracy,” or treating unknown labels as confirmed exposures, would overstate what the output proves. The negative findings help identify what should be improved next.

### Why calculate market returns at all?

The calculations demonstrate that market data can be attached to a historical case and analyzed descriptively. Because the baskets were selected by the researcher rather than produced and validated by the system, the estimates are context—not proof that the event caused a return or that GeoSignal found the exposed firms.

### Why not add Granger just to complete the architecture?

An analysis is not evidence merely because the architecture planned it. The current 90-day archive is a development sample, and the necessary daily risk series and aligned market panel have not been validated. In addition, the current code needs review for the formal Toda–Yamamoto procedure. A rushed result could reduce credibility instead of improving it.

### Why not show a Leontief result using the existing matrix?

The five-sector matrix is labeled toy/illustrative. Its output would reflect chosen coefficients, not observed Indian inter-industry transactions. It can be used for a diagram or explicitly illustrative example, but it should not be presented as an empirical result.

### Why not select a publisher format yet?

There is no target venue. Keeping a generic two-column draft is reasonable for faculty feedback; publisher-specific margins, page limits, citation style, and declarations should be applied only after the venue is selected.

## 5. What the current paper can responsibly claim

It can claim that, for two bounded historical cases, the prototype processed archived event records and metadata, preserved direct source links, generated additional lexical candidates, produced a descriptive actor graph for one case, and calculated market context for researcher-defined baskets. It can also report the observed retrieval weakness and the low sector-label coverage.

It cannot yet claim reliable semantic news retrieval, validated sector/company identification, prediction of security returns, causal event effects, empirical supply-chain propagation, representative performance across geopolitical events, or successful live monitoring.

## 6. Source files to inspect

- [Research status and saved findings](../docs/RESEARCH_STATUS.md)
- [Evidence audit, including proposed claims that are not measured](../docs/PHASE_VI_EVIDENCE_AUDIT.md)
- [Known limitations](../docs/LIMITATIONS.md)
- [Project decisions and their rationale](../docs/DECISIONS.md)
- [Case market metrics](context/case_market_metrics.csv)
- [Research context and evidence inventory](context/CONTEXT.md)
- [Current manuscript](draft_manuscript.tex)

