"""
geosignal.nlp.alignment
BM25 retrieval + TF-IDF cosine verification for event–news alignment.

KEY DESIGN: The TF-IDF vectorizer is fitted ONCE over the entire corpus
at construction time (corpus-level fit). Queries are transformed using
the pre-fitted vectorizer — no per-event refitting. This eliminates the
O(n_events × corpus_size) bottleneck in the original nlp_pipeline.py.

Usage pattern:
    aligner = NewsAligner(news_df)           # fit once
    pairs   = aligner.align_all(events_df)  # query for each event
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    BM25_TOP_N, MIN_COSINE_SIM, NEWS_TEMPORAL_BAND,
)
from geosignal.data.preprocessing import clean_text


class NewsAligner:
    """
    Align geopolitical events to news articles via BM25 + TF-IDF cosine similarity.

    Bottleneck fix: TfidfVectorizer is fitted once on the full corpus text
    at __init__ time. Each query uses transform() (not fit_transform()),
    making per-event cost O(vocab_lookup) rather than O(corpus_fit).
    """

    def __init__(self, news_df: pd.DataFrame,
                 text_col: str = "text",
                 date_col: str = "date",
                 max_tfidf_features: int = 20_000):
        self.news_df  = news_df.reset_index(drop=True)
        self.text_col = text_col
        self.date_col = date_col

        texts = news_df[text_col].fillna("").tolist()

        # ── BM25 index (full corpus) ──────────────────────────────────────────
        logger.info(f"Building BM25 index over {len(texts):,} articles …")
        self.bm25 = BM25Okapi([t.split() for t in texts])
        logger.success("BM25 index ready.")

        # ── TF-IDF vectorizer fitted ONCE on corpus ───────────────────────────
        logger.info("Fitting TF-IDF vectorizer on full corpus …")
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_tfidf_features)
        self.corpus_matrix = self.tfidf.fit_transform(texts)  # (n_docs, vocab)
        logger.success(f"TF-IDF fitted: {self.corpus_matrix.shape[1]:,} features.")

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str,
                 event_date: pd.Timestamp,
                 top_n: int = BM25_TOP_N,
                 temporal_band: int = NEWS_TEMPORAL_BAND) -> pd.DataFrame:
        """
        Return top-N articles by BM25 within ±temporal_band days of event_date.
        Re-scores only the temporal subset (avoids full-corpus scoring).
        """
        lo = event_date - pd.Timedelta(days=temporal_band)
        hi = event_date + pd.Timedelta(days=temporal_band)
        mask   = (self.news_df[self.date_col] >= lo) & (self.news_df[self.date_col] <= hi)
        subset = self.news_df[mask].copy()

        if subset.empty:
            return pd.DataFrame()

        tokenized = [t.split() for t in subset[self.text_col].fillna("").tolist()]
        bm25_sub  = BM25Okapi(tokenized)
        scores    = bm25_sub.get_scores(query.split())
        top_idx   = np.argsort(scores)[::-1][:top_n]

        result = subset.iloc[top_idx].copy().reset_index(drop=True)
        result["bm25_score"] = scores[top_idx]
        return result

    # ── TF-IDF cosine verification ────────────────────────────────────────────

    def cosine_scores(self, query: str, candidate_texts: list[str]) -> list[float]:
        """
        Compute cosine similarity between query and candidates using the
        pre-fitted corpus vectorizer. O(query_transform) — no refitting.
        """
        if not candidate_texts:
            return []
        all_texts = [query] + candidate_texts
        matrix    = self.tfidf.transform(all_texts)          # transform only
        scores    = cosine_similarity(matrix[0:1], matrix[1:])[0].tolist()
        return scores

    # ── End-to-end alignment ──────────────────────────────────────────────────

    def align_all(self, events_df: pd.DataFrame,
                  top_n: int = BM25_TOP_N,
                  cosine_threshold: float = MIN_COSINE_SIM) -> pd.DataFrame:
        """
        For every event in events_df, retrieve candidate articles and filter
        by cosine similarity threshold. Returns event–article pairs DataFrame.

        Output columns:
            event_id, article_date, bm25_score, cosine_sim,
            entities (JSON str), text_snippet
        """
        from geosignal.nlp.entity_extraction import build_nlp_pipeline, extract_entities
        nlp     = build_nlp_pipeline()
        records = []
        n_total = len(events_df)

        for i, (_, event) in enumerate(events_df.iterrows()):
            if i % 500 == 0:
                logger.info(f"Aligning event {i}/{n_total} …")

            query   = clean_text(f"{event['actor1']} {event['actor2']} {event['event_root']}")
            matches = self.retrieve(query, pd.Timestamp(event["date"]), top_n=top_n)
            if matches.empty:
                continue

            texts   = matches[self.text_col].fillna("").tolist()
            cos_sims = self.cosine_scores(query, texts)

            for j, (_, row) in enumerate(matches.iterrows()):
                cos = cos_sims[j] if j < len(cos_sims) else 0.0
                if cos < cosine_threshold:
                    continue
                ner_text = row.get("raw_text") or row.get(self.text_col, "")
                entities = extract_entities(ner_text, nlp)
                records.append({
                    "event_id"    : event["event_id"],
                    "article_date": row["date"],
                    "bm25_score"  : row["bm25_score"],
                    "cosine_sim"  : cos,
                    "entities"    : json.dumps(entities),
                    "text_snippet": row[self.text_col][:300],
                })

        result = pd.DataFrame(records)
        logger.success(f"Alignment complete: {len(result):,} event–article pairs.")
        return result
