"""Reproduce a second historical event/news replay for chip export controls.

The replay uses only 2024-03-12 GDELT/GKG archives already stored in data/raw.
GKG text is metadata-derived, not article body text. Exact URL links are a
source-identity silver signal; TF-IDF rankings are candidate generation only.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_RAW = ROOT / "data" / "raw"
GDELT_COLUMNS = ["GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code",
    "Actor1Type2Code", "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code", "IsRootEvent", "EventCode",
    "EventBaseCode", "EventRootCode", "QuadClass", "GoldsteinScale", "NumMentions",
    "NumSources", "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"]
GKG_COLUMNS = ["Date", "NumArticles", "V1Counts", "V1Themes", "V1Locations", "V1Persons",
    "V1Organisations", "V1Tone", "V1Cameoeventids", "V1Sources", "V1SourceURLs",
    "V1GKGRecordId", "V1Extras", "V1Quotations", "V1AllNames"]


def harmonize_gkg(raw: pd.DataFrame) -> pd.DataFrame:
    get = lambda name: raw[name].astype("string").fillna("")
    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(get("Date").str[:8], format="%Y%m%d", errors="coerce")
    for src, dst in [("V1Themes", "themes"), ("V1Persons", "persons"),
                     ("V1Organisations", "orgs"), ("V1Locations", "locations")]:
        out[dst] = get(src)
    urls = get("V1SourceURLs").mask(get("V1SourceURLs").str.strip().eq(""), get("V1Sources"))
    out["source_url"] = urls.str.split(";", n=1).str[0].str.strip().fillna("")
    out["tone"] = pd.to_numeric(get("V1Tone").str.split(",", n=1).str[0], errors="coerce").fillna(0.0)
    clean = lambda s: s.str.replace(";", " ", regex=False).str.replace(",", " ", regex=False).str.strip()
    out["text"] = clean(out.themes) + " " + clean(out.persons) + " " + clean(out.orgs) + " " + clean(out.locations)
    out["raw_text"] = out.text
    out["article_id"] = get("V1GKGRecordId")
    return out

TAG = "20240312"
TARGET_URL = "https://www.voanews.com/a/us-assessing-expansion-of-chip-export-controls-impacting-china-s-military/7523381.html"
DATA = ROOT / "results" / "datasets"


def read_zip(path: Path, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        item = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".export")))
        return pd.read_csv(zf.open(item), sep="\t", header=None, names=names,
                           dtype=str, low_memory=False, on_bad_lines="skip")


def url_key(value: object) -> str:
    p = urlsplit(str(value).strip())
    host = p.netloc.lower().removeprefix("www.")
    path = re.sub(r"/$", "", p.path).lower()
    return f"{host}{path}" if host and path else ""


def main() -> None:
    events = read_zip(DATA_RAW / "gdelt" / f"{TAG}.export.CSV.zip", GDELT_COLUMNS)
    events = events[events.Day.eq(TAG)].copy()
    target_key = url_key(TARGET_URL)
    matching = events[events.SOURCEURL.map(url_key).eq(target_key)].copy()
    if matching.empty:
        raise RuntimeError("The specified VOA event URL was not found in the cached GDELT export.")
    matching["mentions"] = pd.to_numeric(matching.NumArticles, errors="coerce").fillna(0)
    matching = matching.sort_values("mentions", ascending=False)
    gkg_raw = read_zip(DATA_RAW / "gkg" / f"{TAG}.gkg.csv.zip", GKG_COLUMNS)
    news = harmonize_gkg(gkg_raw)
    news = news[news.date.eq(pd.Timestamp("2024-03-12"))].reset_index(drop=True)
    news["url_key"] = news.source_url.map(url_key)
    exact = news[news.url_key.eq(target_key)].copy()

    # One target event record (largest article count) plus same-day news metadata.
    event = matching.iloc[0]
    matching["url_key"] = matching.SOURCEURL.map(url_key)
    linked = matching.merge(exact, on="url_key", how="inner", suffixes=("_event", "_article"))
    actor_query = " ".join(str(event.get(col, "")) for col in [
        "Actor1Name", "Actor2Name", "Actor1CountryCode", "Actor2CountryCode",
        "Actor1Type1Code", "Actor2Type1Code", "EventCode", "EventRootCode",
    ]) + " US assessing expansion chip export controls China military March 12 2024"
    enriched_query = actor_query + " semiconductors advanced chips technology restricted exports commerce department"
    texts = news.text.fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                 token_pattern=r"(?u)\b\w+\b")
    matrix = vectorizer.fit_transform(texts + [actor_query, enriched_query])
    baseline_scores = cosine_similarity(matrix[-2], matrix[:-2]).ravel()
    enriched_scores = cosine_similarity(matrix[-1], matrix[:-2]).ravel()
    news["tfidf_cosine"] = baseline_scores
    news["rank"] = news.tfidf_cosine.rank(method="first", ascending=False).astype(int)
    exact_rows = news[news.url_key.eq(target_key)]
    exact_rank = int(exact_rows["rank"].min()) if len(exact_rows) else None
    exact_score = float(exact_rows["tfidf_cosine"].max()) if len(exact_rows) else None
    news["enriched_tfidf_cosine"] = enriched_scores
    news["enriched_rank"] = news.enriched_tfidf_cosine.rank(method="first", ascending=False).astype(int)
    enriched_exact = news[news.url_key.eq(target_key)]
    enriched_exact_rank = int(enriched_exact["enriched_rank"].min()) if len(enriched_exact) else None
    enriched_top = news.sort_values("enriched_tfidf_cosine", ascending=False).head(50).copy()
    enriched_top["baseline_rank"] = enriched_top["rank"]
    enriched_top["rank"] = enriched_top["enriched_rank"]
    enriched_top["event_id"] = str(event.GlobalEventID)
    enriched_top["event_url"] = event.SOURCEURL
    enriched_top["is_exact_source_url"] = enriched_top.url_key.eq(target_key)
    top = news.sort_values("tfidf_cosine", ascending=False).head(50).copy()
    top["event_id"] = str(event.GlobalEventID)
    top["event_url"] = event.SOURCEURL
    top["event_code"] = str(event.EventCode).zfill(4)
    top["event_root"] = str(event.EventRootCode).zfill(2)
    top["event_num_articles"] = int(event.mentions)
    top["is_exact_source_url"] = top.url_key.eq(target_key)

    # Retain every same-day GDELT event row on the article source, and its exact GKG match.
    event_rows = matching[["GlobalEventID", "Day", "EventCode", "EventRootCode", "GoldsteinScale",
                           "Actor1Name", "Actor2Name", "NumArticles", "AvgTone", "SOURCEURL"]].copy()
    event_rows.to_csv(DATA / f"chip_controls_events_{TAG}.csv", index=False)
    linked.to_csv(DATA / f"chip_controls_exact_source_links_{TAG}.csv", index=False)
    top.to_csv(DATA / f"chip_controls_top50_metadata_candidates_{TAG}.csv", index=False)
    enriched_top.to_csv(DATA / f"chip_controls_top50_enriched_candidates_{TAG}.csv", index=False)
    result = {
        "case": "US assessing expansion of semiconductor export controls affecting China military",
        "event_date": "2024-03-12",
        "source_event_url": TARGET_URL,
        "event_records_for_same_url": int(len(matching)),
        "same_day_gkg_metadata_rows": int(len(news)),
        "same_day_exact_gkg_article_rows": int(len(exact)),
        "event_to_article_exact_url_link_rows": int(len(linked)),
        "selected_event_id": str(event.GlobalEventID),
        "selected_event_code": str(event.EventCode).zfill(4),
        "selected_event_root": str(event.EventRootCode).zfill(2),
        "selected_event_num_articles": int(event.mentions),
        "candidate_method": "same-day metadata TF-IDF cosine; top 50 retained for relevance audit",
        "exact_url_rank_among_all_same_day_metadata": exact_rank,
        "exact_url_cosine_score": exact_score,
        "enriched_query": "actor/event fields plus semiconductor, advanced chips, restricted exports, Commerce Department terms",
        "enriched_exact_url_rank_among_all_same_day_metadata": enriched_exact_rank,
        "limitations": ["GKG V1 supplies metadata proxies, not article bodies.",
                        "URL identity is not semantic relevance.",
                        "One event and one date do not establish repeatability or predictive validity."],
        "outputs": [f"chip_controls_events_{TAG}.csv", f"chip_controls_exact_source_links_{TAG}.csv",
                    f"chip_controls_top50_metadata_candidates_{TAG}.csv",
                    f"chip_controls_top50_enriched_candidates_{TAG}.csv"],
    }
    (DATA / f"chip_controls_replay_summary_{TAG}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("\nTOP CANDIDATES")
    print(top[["rank", "tfidf_cosine", "source_url", "themes", "persons", "orgs", "is_exact_source_url"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
