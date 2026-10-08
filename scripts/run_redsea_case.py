"""Run a bounded, historical GeoSignal proof-of-concept for the Red Sea case.

This deliberately writes case-labeled artifacts under results/datasets and
does not replace the project's general processed-data files.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import DATA_RAW, RESULTS_DIR, NIFTY50_TICKER
from geosignal.data.loader import GDELT_COLUMNS
from geosignal.nlp.news_loader import GKG_COLUMNS, _harmonize_gkg
from geosignal.nlp.alignment import NewsAligner
from geosignal.nlp.entity_extraction import build_nlp_pipeline, extract_entities
from geosignal.sectors.sector_mapping import enrich_pairs_with_sectors


OUT = RESULTS_DIR / "datasets"
DATE = "2024-01-12"
DATE_TAG = "20240112"
MAX_EVENTS = 30
KEYWORDS = re.compile(r"houth|red.?sea|yemen|shipping|ship|vessel|suez|cargo", re.I)


def read_archive(path: Path, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_name = next(name for name in archive.namelist()
                        if name.lower().endswith((".csv", ".export")))
        return pd.read_csv(archive.open(csv_name), sep="\t", header=None,
                           names=columns, dtype=str, low_memory=False,
                           on_bad_lines="skip")


def load_events() -> pd.DataFrame:
    path = DATA_RAW / "gdelt" / f"{DATE_TAG}.export.CSV.zip"
    raw = read_archive(path, GDELT_COLUMNS)
    raw = raw[(raw["Day"] == DATE_TAG) & (raw["EventRootCode"] == "19")].copy()
    raw["_mentions"] = pd.to_numeric(raw["NumArticles"], errors="coerce").fillna(0)
    url_match = raw["SOURCEURL"].fillna("").str.contains(KEYWORDS)
    actor_text = (raw["Actor1Name"].fillna("") + " " + raw["Actor2Name"].fillna(""))
    actor_match = actor_text.str.contains(r"YEMEN|HOUTHI|NAVY|SHIP|VESSEL|ADEN", case=False, regex=True)
    selected = raw[url_match & actor_match & (raw["_mentions"] >= 10)]
    # One event record per article URL: keep the record with the most mentions.
    selected = (selected.sort_values("_mentions", ascending=False)
                .drop_duplicates("SOURCEURL").head(MAX_EVENTS).copy())
    events = pd.DataFrame({
        "event_id": selected["GlobalEventID"].astype(str),
        "date": pd.to_datetime(selected["Day"], format="%Y%m%d"),
        "event_code": selected["EventCode"].fillna("").astype(str).str.zfill(4),
        "event_root": selected["EventRootCode"].astype(str).str.zfill(2),
        "goldstein": pd.to_numeric(selected["GoldsteinScale"], errors="coerce"),
        "actor1": selected["Actor1Name"].fillna(selected["Actor1CountryCode"]).fillna(""),
        "actor2": selected["Actor2Name"].fillna(selected["Actor2CountryCode"]).fillna(""),
        "action_country": selected["ActionGeo_CountryCode"].fillna(""),
        "action_lat": pd.to_numeric(selected["ActionGeo_Lat"], errors="coerce"),
        "action_lon": pd.to_numeric(selected["ActionGeo_Long"], errors="coerce"),
        "avg_tone": pd.to_numeric(selected["AvgTone"], errors="coerce"),
        "num_articles": selected["_mentions"].astype(int),
        "source_url": selected["SOURCEURL"].fillna(""),
    })
    return events.reset_index(drop=True)


def load_news() -> pd.DataFrame:
    path = DATA_RAW / "gkg" / f"{DATE_TAG}.gkg.csv.zip"
    raw = read_archive(path, GKG_COLUMNS)
    news = _harmonize_gkg(raw)
    return news[news["date"] == pd.Timestamp(DATE)].reset_index(drop=True)


def canonical_url(value: object) -> str:
    """Normalize URLs for cross-source identity matching, without fetching them."""
    try:
        parsed = urlsplit(str(value).strip())
        host = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"/+$", "", parsed.path).lower()
        return f"{host}{path}" if host and path else ""
    except Exception:
        return ""


def exact_source_links(events: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Join GDELT event URLs to same-day GKG source URLs and enrich those rows."""
    e = events.copy()
    n = news.copy()
    e["_url_key"] = e["source_url"].map(canonical_url)
    n["_url_key"] = n["source_url"].map(canonical_url)
    n = n.rename(columns={
        "article_id": "article_id", "source_url": "article_source_url",
        "date": "article_date", "raw_text": "raw_text", "text": "text",
    })
    cols = ["_url_key", "article_id", "article_source_url", "article_date",
            "themes", "persons", "orgs", "locations", "tone", "raw_text", "text"]
    linked = e.merge(n[cols], on="_url_key", how="left").dropna(subset=["article_id"])
    linked = linked.drop_duplicates(subset=["event_id", "article_id"]).copy()
    nlp = build_nlp_pipeline()
    linked["entities"] = linked["raw_text"].fillna("").map(
        lambda value: json.dumps(extract_entities(value, nlp))
    )
    linked["text_snippet"] = linked["text"].fillna("").str.slice(0, 300)
    linked["match_type"] = "exact_source_url"
    return enrich_pairs_with_sectors(linked.drop(columns=["_url_key"]))


def market_context() -> tuple[pd.DataFrame, dict]:
    start, end = "2023-06-01", "2024-05-01"
    downloaded = yf.download([NIFTY50_TICKER], start=start, end=end,
                              auto_adjust=True, progress=False, threads=False)
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no Nifty 50 prices.")
    prices = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=NIFTY50_TICKER)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.rename(columns={"Close": NIFTY50_TICKER})
    prices.to_parquet(OUT / "redsea_nifty50_prices_20230601_20240430.parquet")
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns.to_parquet(OUT / "redsea_nifty50_returns_20230601_20240430.parquet")

    event_day = pd.Timestamp(DATE)
    observed = returns.loc[returns.index >= event_day, NIFTY50_TICKER].dropna()
    out = {"outcome": "raw Nifty 50 cumulative return; descriptive, not sector-matched or causal",
           "ticker": NIFTY50_TICKER, "event_date": DATE,
           "price_rows": int(prices[NIFTY50_TICKER].notna().sum()),
           "price_start": str(prices[NIFTY50_TICKER].first_valid_index().date()),
           "price_end": str(prices[NIFTY50_TICKER].last_valid_index().date()),
           "windows": {}}
    for horizon in (1, 3, 7, 14, 30):
        window = observed.iloc[:horizon]
        value = float((1 + window).prod() - 1) * 100 if len(window) == horizon else None
        out["windows"][f"{horizon}_trading_days_pct"] = round(value, 4) if value is not None else None
    return returns, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = RESULTS_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    events = load_events()
    news = load_news()
    exact_links = exact_source_links(events, news)
    events.to_parquet(OUT / "redsea_events_20240112.parquet", index=False)
    events.to_csv(OUT / "redsea_events_20240112.csv", index=False)
    news.to_parquet(OUT / "redsea_news_corpus_20240112.parquet", index=False)
    exact_links.to_parquet(OUT / "redsea_exact_source_links_20240112.parquet", index=False)
    exact_links.to_csv(OUT / "redsea_exact_source_links_20240112.csv", index=False)

    aligner = NewsAligner(news)
    pairs = aligner.align_all(events)
    pairs = pairs.merge(events[["event_id", "date", "event_code", "event_root",
                                "goldstein", "actor1", "actor2", "num_articles",
                                "source_url"]].rename(columns={"source_url": "event_source_url"}),
                        on="event_id", how="left")
    pairs = enrich_pairs_with_sectors(pairs)
    pairs.to_parquet(OUT / "redsea_event_news_sector_pairs_20240112.parquet", index=False)
    pairs.to_csv(OUT / "redsea_event_news_sector_pairs_20240112.csv", index=False)
    (pairs.groupby("sector", dropna=False).size().rename("pair_count")
     .reset_index().sort_values("pair_count", ascending=False)
     .to_csv(OUT / "redsea_sector_mapping_counts_20240112.csv", index=False))

    # Actor graph is supported by the structured event sample. Sector graph edges
    # are omitted because this sample has no validated sector assignment.
    try:
        from geosignal.graph.network import GeoSignalGraph
        graph = GeoSignalGraph().build(events)
        graph.compute_centrality()
        graph_dir = OUT / "redsea_case"
        graph.export(graph_dir)
        graph.visualise(figures / "redsea_case")
        graph_summary = graph.summary()
    except Exception as exc:
        graph_summary = {"status": "not_generated", "reason": str(exc)}

    returns, market = market_context()
    (OUT / "redsea_market_context_20240112.json").write_text(
        json.dumps(market, indent=2), encoding="utf-8")

    unknown_count = int(pairs["sector"].eq("Unknown").sum()) if "sector" in pairs else 0
    valid_sectors = sorted(set(pairs.get("sector", pd.Series(dtype=str))) - {"Unknown"})
    matched_event_count = int(pairs["event_id"].nunique()) if not pairs.empty else 0
    alignment_metrics = {
        "matched_events": matched_event_count,
        "selected_events": int(len(events)),
        "event_coverage_pct": round(100 * matched_event_count / len(events), 2) if len(events) else 0.0,
        "pairs_per_matched_event": round(len(pairs) / matched_event_count, 2) if matched_event_count else 0.0,
        "mean_cosine_similarity": round(float(pairs["cosine_sim"].mean()), 4) if len(pairs) else None,
        "median_cosine_similarity": round(float(pairs["cosine_sim"].median()), 4) if len(pairs) else None,
        "event_url_available_pairs": int(pairs["event_source_url"].fillna("").ne("").sum()) if "event_source_url" in pairs else 0,
        "article_url_available_pairs": int(pairs["article_source_url"].fillna("").ne("").sum()) if "article_source_url" in pairs else 0,
        "interpretation": "retrieval coverage and similarity diagnostics only; not a manually validated precision score",
    }
    summary = {
        "run_id": "redsea_20240112_poc",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_scope": "historical feasibility demonstration; not a live system or causal claim",
        "case": "US/UK strikes on Houthi targets / Red Sea shipping coverage, 2024-01-12",
        "scope": {"event_date": DATE, "event_root": "19", "source_url_keyword_filter": KEYWORDS.pattern,
                  "deduplicated_by": "GDELT source URL", "max_events": MAX_EVENTS,
                  "news_corpus": "GDELT GKG v1 metadata for 2024-01-12 only",
                  "alignment": "sparse corpus-level BM25 retrieval followed by corpus-fitted TF-IDF cosine threshold"},
        "stages": {
            "historical_structured_events": {"status": "complete", "records": len(events)},
            "historical_news_metadata": {"status": "complete", "records": len(news),
                                         "full_article_text": False},
            "event_news_source_linkage": {
                "status": "complete" if exact_links["event_id"].nunique() == len(events) else "partial",
                "exact_source_url_links": len(exact_links), "selected_events": len(events),
                "events_with_exact_source_link": int(exact_links["event_id"].nunique()),
                "event_coverage_pct": round(100 * exact_links["event_id"].nunique() / len(events), 2) if len(events) else 0.0,
                "interpretation": "same source URL appears in GDELT event and GKG metadata; this verifies source linkage, not article meaning",
            },
            "lexical_news_retrieval": {"status": "complete" if len(pairs) else "no_matches",
                                     "pairs": len(pairs),
                                     **alignment_metrics,
                                     "metadata_similarity_only": True},
            "sector_mapping": {"status": "partial_low_confidence",
                               "unknown_pairs": unknown_count, "mapped_sectors": valid_sectors,
                               "mapped_pairs": int(len(pairs) - unknown_count),
                               "exact_source_link_sector_counts": exact_links["sector"].value_counts().to_dict(),
                               "reason": "Only four metadata keyword matches (two Oil-price/Energy theme signals and two Suez/Industrials mentions) were found; these are sparse, rule-based proxy signals, not validated sector exposures."},
            "market_response": {"status": "descriptive_broad_market_only", **market},
            "sector_event_study": {"status": "not_run",
                                   "reason": "No usable case-matched index history: Industrials has no configured NSE index, and the configured ^CNXENERGY Yahoo Finance history returned no rows. No proxy index was substituted."},
            "granger_analysis": {"status": "not_run", "reason": "One-day event/news case cannot support a time-series test."},
            "input_output_propagation": {"status": "not_run", "reason": "Project decision log identifies the current I-O matrix as illustrative only."},
            "actor_network": {"status": "complete" if graph_summary else "partial", **graph_summary},
        },
        "limitations": [
            "GKG v1 is metadata (themes, named entities, locations), not article body text.",
            "One event-date corpus is used; the intended +/- 3 day news retrieval band is not covered in this case run.",
            "Event records are machine-coded and may include residual semantic duplicates despite URL deduplication.",
            "Nifty 50 returns are a broad-market description, not a sector-matched or causal impact estimate.",
            "Sector-event study, Granger, and empirical I-O propagation remain unvalidated/not run.",
        ],
        "outputs": [
            "redsea_events_20240112.parquet", "redsea_events_20240112.csv",
            "redsea_news_corpus_20240112.parquet",
            "redsea_exact_source_links_20240112.parquet", "redsea_exact_source_links_20240112.csv",
            "redsea_event_news_sector_pairs_20240112.parquet",
            "redsea_event_news_sector_pairs_20240112.csv",
            "redsea_sector_mapping_counts_20240112.csv",
            "redsea_case/network_edges.csv", "redsea_case/network_nodes.csv",
            "redsea_nifty50_prices_20230601_20240430.parquet",
            "redsea_nifty50_returns_20230601_20240430.parquet",
            "redsea_market_context_20240112.json", "../figures/redsea_case/network_graph.png",
        ],
    }
    (OUT / "redsea_run_summary_20240112.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    log_path = RESULTS_DIR / "experiment_log.json"
    log_data = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {
        "schema_version": "2", "experiments": []
    }
    log_data.setdefault("experiments", [])
    for prior in log_data["experiments"]:
        if prior.get("experiment_id") == "EXP-001":
            prior["status"] = "superseded_stale_data_outputs"
            note = "Original generated processed tables were found to have incorrect date coverage; do not use them as Q1 results."
            if note not in prior.get("notes", ""):
                prior["notes"] = f"{prior.get('notes', '').strip()} {note}".strip()
    log_data["experiments"] = [x for x in log_data["experiments"]
                                if x.get("experiment_id") != "EXP-002"]
    log_data["experiments"].append({
        "experiment_id": "EXP-002",
        "description": "Red Sea historical feasibility run: event-to-news alignment and actor network",
        "created": datetime.now(timezone.utc).date().isoformat(),
        "status": "poc_complete_sector_impact_pending",
        "config": summary["scope"],
        "stages_completed": [name for name, stage in summary["stages"].items()
                             if stage.get("status") in ("complete", "descriptive_broad_market_only")],
        "stages_partial": [name for name, stage in summary["stages"].items()
                            if stage.get("status", "").startswith("partial")],
        "stages_pending": [name for name, stage in summary["stages"].items()
                           if stage.get("status") == "not_run"],
        "results": {"event_records": len(events), "news_metadata_records": len(news),
                    "event_news_pairs": len(pairs), "sector_counts": pairs["sector"].value_counts().to_dict(),
                    "alignment_metrics": alignment_metrics,
                    "nifty50_cumulative_returns_pct": market["windows"]},
        "outputs": summary["outputs"],
        "notes": "Historical feasibility only. GKG is metadata, sector assignments are sparse/low-confidence; no sector event study, Granger, prediction evaluation, or empirical I-O result.",
    })
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    print(json.dumps({"event_records": len(events), "news_records": len(news),
                      "matched_pairs": len(pairs), "sector_counts": pairs["sector"].value_counts().to_dict(),
                      "market_windows": market["windows"],
                      "summary_path": str(OUT / "redsea_run_summary_20240112.json")}, indent=2))


if __name__ == "__main__":
    main()
