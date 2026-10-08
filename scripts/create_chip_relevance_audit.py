"""Write the analyst relevance audit for the top 20 March 12 candidates."""
from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "datasets"

# 0 unrelated, 1 adjacent context only, 2 directly relevant to the controls event.
# Decisions are based on visible URL titles/slugs and GKG metadata; this is a
# small analyst audit, not a blinded or multi-rater gold standard.
LABELS = {
 1:(0,"Offshore oil-platform construction; no chip-control relation."),
 2:(1,"Quantum-sector investment is technology context but not export controls or semiconductors."),
 3:(0,"Provincial GDP/economic outlook; no controls or chip supply-chain relation."),
 4:(0,"Tibetan Plateau rainfall research; unrelated to the event."),
 5:(1,"Chinese political profile/editorial context; not the chip-controls event."),
 6:(1,"Syndicated/profile version of China leadership story; no chip controls."),
 7:(1,"Syndicated/profile version of China leadership story; no chip controls."),
 8:(0,"Tibetan hydropower protest; unrelated to chip export controls."),
 9:(0,"Qinghai environmental/green coverage story; unrelated."),
10:(0,"Qinghai environmental/green coverage story; unrelated."),
11:(0,"Syndicated environmental/green coverage story; unrelated."),
12:(0,"Chinese film/entertainment story based on metadata entities; unrelated."),
13:(0,"Syndicated Chinese film/entertainment story; unrelated."),
14:(0,"Syndicated Chinese film/entertainment story; unrelated."),
15:(0,"Syndicated Chinese film/entertainment story; unrelated."),
16:(1,"Chinese naval/military story; broad geopolitical context, not chip controls."),
17:(1,"Syndicated Chinese naval/military story; broad geopolitical context, not chip controls."),
18:(1,"China East China Sea drilling / naval-context story; not chip controls."),
19:(0,"Yangtze River/water infrastructure metadata; no chip-control relation."),
20:(1,"China industrial-metals production outlook; broad industrial context, no specific chip-control link."),
}

def main():
    candidates = pd.read_csv(DATA / "chip_controls_top50_metadata_candidates_20240312.csv")
    audit = candidates[candidates["rank"].le(20)].copy()
    audit["manual_relevance_label"] = audit["rank"].map(lambda n: LABELS[int(n)][0])
    audit["label_name"] = audit.manual_relevance_label.map({0:"unrelated",1:"adjacent_context",2:"directly_relevant"})
    audit["review_basis"] = "URL title/slug plus same-day GKG themes, people, organizations, and location metadata"
    audit["review_note"] = audit["rank"].map(lambda n: LABELS[int(n)][1])
    audit.to_csv(DATA / "chip_controls_top20_relevance_audit_20240312.csv", index=False)
    exact = pd.read_csv(DATA / "chip_controls_exact_source_links_20240312.csv")
    exact["manual_relevance_label"] = 2
    exact["label_name"] = "directly_relevant"
    exact["review_basis"] = "VOA article title and same normalized source URL; article directly reports US assessment of chip export controls affecting China’s military"
    exact.to_csv(DATA / "chip_controls_exact_link_relevance_review_20240312.csv", index=False)
    counts = audit.label_name.value_counts().to_dict()
    replay = json.loads((DATA / "chip_controls_replay_summary_20240312.json").read_text(encoding="utf-8"))
    summary = {
      "sample": "top 20 same-day TF-IDF candidates, plus separately examined exact source link",
      "rubric": {"0":"unrelated", "1":"adjacent context only", "2":"directly relevant to the semiconductor export-controls event"},
      "top20_counts": {k:int(v) for k,v in counts.items()},
      "direct_relevance_precision_at_20": 0.0,
      "gold_article_rank_among_same_day_metadata": replay["exact_url_rank_among_all_same_day_metadata"],
      "gold_article_rank_after_enriched_query": replay["enriched_exact_url_rank_among_all_same_day_metadata"],
      "gold_article_in_top_50": False,
      "review_limits": ["Analyst single-rater audit.", "Most candidates were judged from URL title/slug and GKG metadata, not full article text.", "Labels are a small, event-specific audit and should not be generalized as a performance estimate."],
      "outputs": ["chip_controls_top20_relevance_audit_20240312.csv", "chip_controls_exact_link_relevance_review_20240312.csv"]
    }
    (DATA / "chip_controls_relevance_audit_summary_20240312.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__ == "__main__": main()
