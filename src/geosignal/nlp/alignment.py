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


import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import (
    BM25_TOP_N, MIN_COSINE_SIM, NEWS_TEMPORAL_BAND,
)
from geosignal.data.preprocessing import clean_text


class _SparseBM25:
    """Corpus-level BM25 scores backed by a sparse term-frequency matrix."""

    def __init__(self, counts, vocabulary: dict[str, int], k1: float = 1.5,
                 b: float = 0.75, epsilon: float = 0.25):
        self.counts = counts.tocsr()
        self.vocabulary = vocabulary
        self.k1 = k1
        self.b = b
        self.doc_len = np.asarray(self.counts.sum(axis=1)).ravel().astype(float)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        n_docs = self.counts.shape[0]
        doc_freq = np.asarray(self.counts.getnnz(axis=0)).ravel().astype(float)
        idf = np.log(n_docs - doc_freq + 0.5) - np.log(doc_freq + 0.5)
        if len(idf):
            avg_idf = float(idf.mean())
            idf[idf < 0] = epsilon * avg_idf
        self.idf = idf

    def get_scores(self, query_tokens: list[str],
                   candidate_indices: np.ndarray | None = None) -> np.ndarray:
        if candidate_indices is None:
            candidate_indices = np.arange(self.counts.shape[0])
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        scores = np.zeros(len(candidate_indices), dtype=float)
        if not len(candidate_indices) or self.avgdl <= 0:
            return scores

        length_norm = self.k1 * (
            1 - self.b + self.b * self.doc_len[candidate_indices] / self.avgdl
        )
        for token in query_tokens:
            term_index = self.vocabulary.get(token)
            if term_index is None:
                continue
            term_freq = np.asarray(
                self.counts[candidate_indices, term_index].toarray()
            ).ravel()
            scores += (
                self.idf[term_index] * term_freq * (self.k1 + 1)
                / (term_freq + length_norm)
            )
        return scores


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

        # Store BM25 term frequencies sparsely instead of retaining one Python
        # token list per article. This keeps corpus-level BM25 feasible for
        # historical GKG windows with hundreds of thousands of records.
        logger.info(f"Building sparse BM25 index over {len(texts):,} articles …")
        bm25_vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            max_features=200_000,
            dtype=np.int32,
        )
        bm25_counts = bm25_vectorizer.fit_transform(texts)
        self.bm25 = _SparseBM25(bm25_counts, bm25_vectorizer.vocabulary_)
        del bm25_counts
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

        The BM25 index is built once at construction time and reused for every
        event. Temporal filtering is applied to the corpus-level BM25 scores,
        avoiding repeated BM25 index construction.
        """
        lo = event_date - pd.Timedelta(days=temporal_band)
        hi = event_date + pd.Timedelta(days=temporal_band)

        mask = (
                (self.news_df[self.date_col] >= lo)
                & (self.news_df[self.date_col] <= hi)
        )

        candidate_indices = np.flatnonzero(mask.to_numpy())

        if len(candidate_indices) == 0:
            return pd.DataFrame()

        # Score only documents inside the temporal band while retaining
        # corpus-level inverse document frequencies.
        subset_scores = self.bm25.get_scores(query.split(), candidate_indices)

        top_local_idx = np.argsort(subset_scores)[::-1][:top_n]

        result = self.news_df.iloc[
            candidate_indices[top_local_idx]
        ].copy().reset_index(drop=True)

        result["bm25_score"] = subset_scores[top_local_idx]

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
                    "article_id"  : row.get("article_id", ""),
                    "article_source_url": row.get("source_url", ""),
                    "article_date": row["date"],
                    "bm25_score"  : row["bm25_score"],
                    "cosine_sim"  : cos,
                    "entities"    : json.dumps(entities),
                    "text_snippet": row[self.text_col][:300],
                })

        result = pd.DataFrame(records)
        logger.success(f"Alignment complete: {len(result):,} event–article pairs.")
        return result
