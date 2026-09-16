"""
src/nlp_pipeline.py
Phase II: Unstructured Context Processing

Implements:
  1. BM25 retrieval – finds top-N news articles for each event
  2. TF-IDF + cosine similarity – verifies alignment
  3. spaCy NER + EntityRuler – extracts industry/commodity entities

Usage:
    python src/nlp_pipeline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.language import Language

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    DATA_PROC, BM25_TOP_N, MIN_COSINE_SIM,
    NEWS_TEMPORAL_BAND, ENTITY_RULES_PATH,
)
# Use local import (both work when CWD=project root or when running as module)
try:
    from preprocessing import clean_text
except ImportError:
    from src.preprocessing import clean_text


# ── BM25 Retrieval ────────────────────────────────────────────────────────────

class NewsRetriever:
    """
    BM25-based news retriever aligned to a structured event corpus.
    
    The retriever constructs an inverted index over the news corpus.
    For each geopolitical event, it queries using the event description
    (actor1 + actor2 + event_root) and retrieves top-N articles within
    a tight temporal window.
    """

    def __init__(self, news_df: pd.DataFrame,
                 text_col: str = "text",
                 date_col: str = "date"):
        self.news_df  = news_df.copy()
        self.text_col = text_col
        self.date_col = date_col

        # Tokenize corpus for BM25
        logger.info("Building BM25 index...")
        tokenized = [doc.split() for doc in news_df[text_col].fillna("").tolist()]
        self.bm25 = BM25Okapi(tokenized)
        logger.success(f"BM25 index built over {len(tokenized):,} articles.")

    def retrieve(self, query: str, event_date: pd.Timestamp,
                 top_n: int = BM25_TOP_N,
                 temporal_band: int = NEWS_TEMPORAL_BAND) -> pd.DataFrame:
        """
        Retrieve top-N articles most relevant to `query` within ±temporal_band days
        of `event_date`.

        Args:
            query:         Combined event description string.
            event_date:    The date of the geopolitical event.
            top_n:         Number of articles to retrieve.
            temporal_band: Days before/after event date to search.

        Returns:
            DataFrame of matched articles with BM25 scores.
        """
        window_start = event_date - pd.Timedelta(days=temporal_band)
        window_end   = event_date + pd.Timedelta(days=temporal_band)

        mask    = (self.news_df[self.date_col] >= window_start) & \
                  (self.news_df[self.date_col] <= window_end)
        subset  = self.news_df[mask].copy()

        if subset.empty:
            return pd.DataFrame()

        # Re-score only within temporal subset
        tokenized_subset = [doc.split() for doc in subset[self.text_col].fillna("").tolist()]
        bm25_sub = BM25Okapi(tokenized_subset)

        scores   = bm25_sub.get_scores(query.split())
        top_idx  = np.argsort(scores)[::-1][:top_n]

        result             = subset.iloc[top_idx].copy().reset_index(drop=True)
        result["bm25_score"] = scores[top_idx]
        return result


# ── TF-IDF Cosine Verification ────────────────────────────────────────────────

def verify_alignment(event_desc: str,
                     candidate_texts: list[str],
                     threshold: float = MIN_COSINE_SIM) -> list[float]:
    """
    Compute TF-IDF cosine similarity between event description and candidate articles.

    Args:
        event_desc:       Synthesized event description query string.
        candidate_texts:  List of news article texts.
        threshold:        Minimum cosine score to consider aligned (for logging).

    Returns:
        List of cosine similarity scores, one per candidate.
    """
    if not candidate_texts:
        return []

    corpus  = [event_desc] + candidate_texts
    vec     = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000)
    matrix  = vec.fit_transform(corpus)

    # Similarity of event description (row 0) against all articles (rows 1+)
    scores  = cosine_similarity(matrix[0:1], matrix[1:])[0].tolist()
    aligned = sum(1 for s in scores if s >= threshold)
    logger.debug(f"TF-IDF: {aligned}/{len(scores)} articles above threshold ({threshold})")
    return scores


# ── spaCy NER + EntityRuler ───────────────────────────────────────────────────

def build_nlp_pipeline() -> Language:
    """
    Build a spaCy pipeline with:
      - Standard en_core_web_sm NER
      - Custom EntityRuler prepended with industry-specific terminology
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.warning("spaCy model 'en_core_web_sm' not found. "
                       "Run: python -m spacy download en_core_web_sm")
        raise

    # Load custom entity rules if available
    patterns = _load_entity_patterns()

    # EntityRuler runs BEFORE the statistical NER to give priority to custom patterns
    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": False})
    ruler.add_patterns(patterns)

    logger.info(f"EntityRuler loaded with {len(patterns)} custom patterns.")
    return nlp


def _load_entity_patterns() -> list[dict]:
    """
    Load industry-specific entity patterns from the JSONL config file.
    Falls back to a minimal built-in set if the file does not exist.
    """
    if ENTITY_RULES_PATH.exists():
        patterns = []
        with open(ENTITY_RULES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip blank lines and # comment lines (not valid JSON)
                if not line or line.startswith("#"):
                    continue
                try:
                    patterns.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid EntityRuler line: {line!r} ({e})")
        return patterns

    # Built-in fallback patterns
    logger.warning(f"EntityRuler JSONL not found at {ENTITY_RULES_PATH}. Using built-in fallback.")
    return [
        # Semiconductor
        {"label": "SEMICONDUCTOR", "pattern": "EUV lithography"},
        {"label": "SEMICONDUCTOR", "pattern": "NAND flash"},
        {"label": "SEMICONDUCTOR", "pattern": "advanced silicon"},
        {"label": "SEMICONDUCTOR", "pattern": "fabrication plant"},
        {"label": "SEMICONDUCTOR", "pattern": "microcontroller"},
        # Energy / Logistics
        {"label": "ENERGY",        "pattern": "Brent crude"},
        {"label": "ENERGY",        "pattern": "Petroline"},
        {"label": "LOGISTICS",     "pattern": "Suez Canal"},
        {"label": "LOGISTICS",     "pattern": "Cape of Good Hope"},
        {"label": "LOGISTICS",     "pattern": "freight surcharge"},
        {"label": "LOGISTICS",     "pattern": "container throughput"},
        # Agriculture
        {"label": "AGRICULTURE",   "pattern": "rice shipment"},
        {"label": "AGRICULTURE",   "pattern": "food inflation"},
        # Critical Minerals
        {"label": "CRITICAL_MINERAL", "pattern": "lithium"},
        {"label": "CRITICAL_MINERAL", "pattern": "cobalt"},
        {"label": "CRITICAL_MINERAL", "pattern": "rare earth"},
    ]


def extract_entities(text: str, nlp: Language) -> list[dict]:
    """
    Run the full NLP pipeline on a text and return extracted entities.

    Returns:
        List of dicts: {"text": ..., "label": ..., "start": ..., "end": ...}
    """
    doc = nlp(text[:100_000])  # spaCy max length guard
    return [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]


# ── End-to-end pipeline runner ────────────────────────────────────────────────

def run_nlp_pipeline(events_df: pd.DataFrame,
                     news_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event, retrieve aligned news articles and extract entities.

    Args:
        events_df: Processed events (canonical schema from preprocessing.py).
        news_df:   Cleaned news corpus with 'date' and 'text' columns.

    Returns:
        DataFrame with columns: event_id, bm25_score, cosine_sim, entities_json
    """
    retriever  = NewsRetriever(news_df)
    nlp        = build_nlp_pipeline()
    records    = []

    for _, event in events_df.iterrows():
        query   = f"{event['actor1']} {event['actor2']} {event['event_root']}"
        query   = clean_text(query)
        matches = retriever.retrieve(query, pd.Timestamp(event["date"]))

        if matches.empty:
            continue

        texts   = matches["text"].fillna("").tolist()
        cos_sims = verify_alignment(query, texts)

        for i, (_, row) in enumerate(matches.iterrows()):
            cos = cos_sims[i] if i < len(cos_sims) else 0.0
            if cos < MIN_COSINE_SIM:
                continue
            # ── NER must run on raw_text (preserves case/punctuation for spaCy) ──
            # If raw_text column exists use it; otherwise fall back to text with a warning.
            ner_text = row.get("raw_text") or row.get("text", "")
            if not row.get("raw_text"):
                logger.debug("raw_text column missing; NER running on cleaned text — "
                             "entity recall may be reduced.")
            entities = extract_entities(ner_text, nlp)
            records.append({
                "event_id"    : event["event_id"],
                "article_date": row["date"],
                "bm25_score"  : row["bm25_score"],
                "cosine_sim"  : cos,
                "entities"    : json.dumps(entities),
                "text_snippet": row["text"][:300],
            })

    result = pd.DataFrame(records)
    logger.success(f"NLP pipeline complete: {len(result):,} event-article pairs extracted.")
    return result


def main():
    events_path = DATA_PROC / "processed_events.parquet"
    news_path   = DATA_PROC / "news_corpus.parquet"     # populated separately

    if not events_path.exists():
        logger.error("Run preprocessing.py first.")
        return
    if not news_path.exists():
        logger.error("News corpus parquet not found. Populate data/processed/news_corpus.parquet.")
        return

    events_df = pd.read_parquet(events_path)
    news_df   = pd.read_parquet(news_path)

    result    = run_nlp_pipeline(events_df, news_df)
    out_path  = DATA_PROC / "event_news_pairs.parquet"
    result.to_parquet(out_path, index=False)
    logger.success(f"Saved event-news pairs → {out_path}")


if __name__ == "__main__":
    main()
