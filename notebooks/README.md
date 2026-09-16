# GeoSignal Notebooks

Sequential Jupyter notebooks for exploration, aligned with pipeline phases.

| Notebook | Phase | Purpose |
|---|---|---|
| `01_data_exploration.ipynb` | I | Inspect raw GDELT/UCDP data, check distributions |
| `02_event_processing.ipynb` | I→II | Harmonize events, visualize Goldstein scores |
| `03_nlp_alignment.ipynb`    | II | BM25 retrieval demo, cosine sim heatmaps |
| `04_sector_mapping.ipynb`   | III | NER output inspection, GICS mapping validation |
| `05_event_study.ipynb`      | IV | CAR plots, MacKinlay event study walkthrough |
| `06_granger_causality.ipynb`| IV | VAR model, Granger test results |
| `07_leontief_io.ipynb`      | V | I-O shock propagation visualization |
| `08_case_study_redsea.ipynb`| VI | Full Red Sea case study end-to-end |
| `09_case_study_semicon.ipynb`| VI | Semiconductor export controls case study |

Run notebooks in order. Each notebook reads from `data/processed/` outputs of the preceding step.
