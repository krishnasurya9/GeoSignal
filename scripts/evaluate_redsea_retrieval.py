"""Evaluate the saved Red Sea event/news outputs against exact source URL links.

The exact URL cross-links are a silver-standard linkage signal, not human
judgements of article relevance. This script writes transparent retrieval
metrics and a union table that preserves both exact and lexical pathways.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "datasets"
TAG = "20240112"


def canonical_url(value: object) -> str:
    try:
        parsed = urlsplit(str(value).strip())
        host = parsed.netloc.lower().removeprefix("www.")
        path = re.sub(r"/+$", "", parsed.path).lower()
        return f"{host}{path}" if host and path else ""
    except Exception:
        return ""


def main() -> None:
    pairs = pd.read_csv(DATA / f"redsea_event_news_sector_pairs_{TAG}.csv", low_memory=False)
    exact = pd.read_csv(DATA / f"redsea_exact_source_links_{TAG}.csv", low_memory=False)
    events = pd.read_csv(DATA / f"redsea_events_{TAG}.csv", low_memory=False)

    pairs["_event_url_key"] = pairs["event_source_url"].map(canonical_url)
    pairs["_article_url_key"] = pairs["article_source_url"].map(canonical_url)
    pairs["retrieval_rank"] = pairs.groupby("event_id")["cosine_sim"].rank(
        method="first", ascending=False
    ).astype(int)
    exact["_event_url_key"] = exact["source_url"].map(canonical_url)
    exact["_article_url_key"] = exact["article_source_url"].map(canonical_url)

    labels: dict[str, set[str]] = {}
    for event_id, group in exact.groupby("event_id"):
        labels[str(event_id)] = set(group["_article_url_key"]) - {""}

    lexical_exact_rows = pairs[
        pairs["_event_url_key"].ne("")
        & pairs["_event_url_key"].eq(pairs["_article_url_key"])
    ].copy()
    per_event = []
    for event_id, targets in labels.items():
        group = pairs[pairs["event_id"].astype(str).eq(event_id)].sort_values(
            "cosine_sim", ascending=False
        )
        exact_hits = lexical_exact_rows[lexical_exact_rows["event_id"].astype(str).eq(event_id)]
        ranks = exact_hits["retrieval_rank"].tolist()
        row = {
            "event_id": event_id,
            "exact_source_url_links": int((exact["event_id"].astype(str) == event_id).sum()),
            "distinct_exact_source_urls": len(targets),
            "lexical_candidate_pairs": len(group),
            "lexical_candidate_generated": bool(len(group)),
            "exact_source_url_in_lexical_candidates": bool(ranks),
            "best_exact_source_url_rank": min(ranks) if ranks else None,
            "exact_source_hit_at_10": bool(any(rank <= 10 for rank in ranks)),
        }
        per_event.append(row)
    event_eval = pd.DataFrame(per_event).sort_values("event_id")

    # Build the hybrid table from the exact source joins plus lexical candidates.
    exact_out = exact.copy().rename(columns={"source_url": "event_source_url"})
    exact_out["retrieval_method"] = "exact_source_url"
    pairs_out = pairs.copy()
    pairs_out["retrieval_method"] = "lexical_bm25_tfidf"
    exact_out["is_exact_source_url_match"] = True
    pairs_out["is_exact_source_url_match"] = pairs_out["_event_url_key"].ne("") & pairs_out[
        "_event_url_key"
    ].eq(pairs_out["_article_url_key"])
    combined = pd.concat([exact_out, pairs_out], ignore_index=True, sort=False)
    combined["_pair_key"] = combined["event_id"].astype(str) + "|" + combined["article_id"].astype(str)
    methods = combined.groupby("_pair_key")["retrieval_method"].agg(
        lambda values: "+".join(sorted(set(values)))
    )
    exact_flags = combined.groupby("_pair_key")["is_exact_source_url_match"].any()
    combined = combined.groupby("_pair_key", sort=False, as_index=False).first()
    combined["retrieval_method"] = combined["_pair_key"].map(methods)
    combined["is_exact_source_url_match"] = combined["_pair_key"].map(exact_flags).fillna(False)
    combined = combined.drop(columns=[c for c in combined.columns if c.startswith("_")])

    exact_events = len(labels)
    selected_events = int(events["event_id"].nunique())
    lexical_hit_events = int(event_eval["lexical_candidate_generated"].sum())
    exact_hits_in_lexical = int(event_eval["exact_source_url_in_lexical_candidates"].sum())
    hit_at_10 = int(event_eval["exact_source_hit_at_10"].sum())
    exact_pair_recall_at_10 = float(hit_at_10 / exact_events) if exact_events else 0.0
    exact_candidates = int(lexical_exact_rows.shape[0])
    metrics = {
        "evaluation": "single-case historical offline replay",
        "case_date": "2024-01-12",
        "reference_label": "same normalized source URL in GDELT event and GKG metadata; silver standard only",
        "selected_event_count": selected_events,
        "exact_url_link_rows": int(len(exact)),
        "events_with_exact_url_link": exact_events,
        "exact_url_event_coverage_pct": round(100 * exact_events / max(1, selected_events), 2),
        "lexical_candidate_pairs": int(len(pairs)),
        "events_with_lexical_candidates": lexical_hit_events,
        "lexical_candidate_event_coverage_pct": round(100 * lexical_hit_events / max(1, exact_events), 2),
        "lexical_exact_source_url_pairs": exact_candidates,
        "lexical_exact_source_url_pair_precision_pct": round(100 * exact_candidates / max(1, len(pairs)), 2),
        "events_with_exact_source_url_in_lexical_list": exact_hits_in_lexical,
        "exact_source_url_event_recall_at_10_pct": round(100 * exact_pair_recall_at_10, 2),
        "hybrid_unique_event_article_pairs": int(len(combined)),
        "hybrid_events_with_exact_or_lexical_path": int(combined["event_id"].nunique()),
        "hybrid_event_coverage_pct": round(100 * combined["event_id"].nunique() / max(1, exact_events), 2),
        "interpretation": (
            "Direct URL linkage gives complete event coverage in this selected case. "
            "Lexical candidates broaden context but rarely recover the same-source URL; "
            "no human relevance labels or full article text are available."
        ),
    }

    event_eval.to_csv(DATA / f"redsea_retrieval_event_audit_{TAG}.csv", index=False)
    combined.to_csv(DATA / f"redsea_hybrid_event_news_pairs_{TAG}.csv", index=False)
    (DATA / f"redsea_retrieval_evaluation_{TAG}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    summary_path = DATA / f"redsea_run_summary_{TAG}.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["research_scope"] = (
            "offline historical replay of an architecture proposed for live geopolitical monitoring; "
            "live operation was not evaluated"
        )
        summary["stages"]["hybrid_event_news_output"] = {
            "status": "complete_for_selected_case",
            "unique_event_article_pairs": metrics["hybrid_unique_event_article_pairs"],
            "events_with_exact_or_lexical_path": metrics["hybrid_events_with_exact_or_lexical_path"],
            "selected_events": metrics["selected_event_count"],
            "event_coverage_pct": metrics["hybrid_event_coverage_pct"],
            "interpretation": metrics["interpretation"],
        }
        summary["stages"]["semantic_relevance_validation"] = {
            "status": "not_run",
            "reason": "GKG v1 contains metadata rather than article bodies; no independent human relevance labels were available.",
        }
        summary["retrieval_evaluation"] = metrics
        for name in [
            f"redsea_retrieval_event_audit_{TAG}.csv",
            f"redsea_hybrid_event_news_pairs_{TAG}.csv",
            f"redsea_retrieval_evaluation_{TAG}.json",
        ]:
            if name not in summary["outputs"]:
                summary["outputs"].append(name)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_path = ROOT / "results" / "experiment_log.json"
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        for experiment in log.get("experiments", []):
            if experiment.get("experiment_id") == "EXP-002":
                experiment["status"] = "offline_hybrid_retrieval_evaluated_sector_impact_pending"
                experiment.setdefault("results", {})["retrieval_evaluation"] = metrics
                completed = experiment.setdefault("stages_completed", [])
                if "hybrid_event_news_output" not in completed:
                    completed.append("hybrid_event_news_output")
                outputs = experiment.setdefault("outputs", [])
                for name in [
                    f"redsea_retrieval_event_audit_{TAG}.csv",
                    f"redsea_hybrid_event_news_pairs_{TAG}.csv",
                    f"redsea_retrieval_evaluation_{TAG}.json",
                ]:
                    if name not in outputs:
                        outputs.append(name)
                pending = experiment.setdefault("stages_pending", [])
                if "manual_semantic_relevance_validation" not in pending:
                    pending.append("manual_semantic_relevance_validation")
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
