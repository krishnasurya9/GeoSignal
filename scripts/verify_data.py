"""
scripts/verify_data.py
Verification script for downloaded GDELT and GKG news datasets.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def verify():
    gdelt_path = Path("data/raw/gdelt_raw.parquet")
    news_path = Path("data/processed/news_corpus.parquet")

    print("=" * 60)
    print("1. VERIFYING GDELT RAW DATA")
    print("=" * 60)
    if gdelt_path.exists():
        df_g = pd.read_parquet(gdelt_path)
        print(f"File: {gdelt_path}")
        print(f"Total Rows: {len(df_g):,}")
        print(f"Total Columns: {len(df_g.columns)}")
        print(f"Memory: {df_g.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print(f"Day range: {df_g['Day'].min()} to {df_g['Day'].max()}")
        print("\nNull percentages in key columns:")
        key_cols = [
            "GlobalEventID", "Day", "EventCode", "EventRootCode",
            "GoldsteinScale", "Actor1CountryCode", "Actor2CountryCode",
            "ActionGeo_CountryCode", "AvgTone", "NumArticles", "SOURCEURL"
        ]
        for c in key_cols:
            if c in df_g.columns:
                null_pct = df_g[c].isna().mean() * 100
                print(f"  {c:25s}: {null_pct:5.1f}% null")
        print("\nTop 5 EventRootCodes:")
        print(df_g["EventRootCode"].value_counts().head(5).to_string())
        print("\nTop 5 ActionGeo Countries:")
        print(df_g["ActionGeo_CountryCode"].value_counts().head(5).to_string())
        print("\nGoldsteinScale summary:")
        print(pd.to_numeric(df_g["GoldsteinScale"], errors="coerce").describe().round(2).to_string())
    else:
        print(f"ERROR: {gdelt_path} does not exist!")

    print("\n" + "=" * 60)
    print("2. VERIFYING GKG NEWS CORPUS")
    print("=" * 60)
    if news_path.exists():
        df_n = pd.read_parquet(news_path)
        print(f"File: {news_path}")
        print(f"Total Rows: {len(df_n):,}")
        print(f"Total Columns: {len(df_n.columns)}")
        print(f"Memory: {df_n.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
        print(f"Date range: {df_n['date'].min()} to {df_n['date'].max()}")
        print(f"Columns: {list(df_n.columns)}")
        print("\nNull percentages in news columns:")
        for c in df_n.columns:
            null_pct = df_n[c].isna().mean() * 100
            print(f"  {c:20s}: {null_pct:5.1f}% null")
        print("\nTone summary:")
        print(df_n["tone"].describe().round(2).to_string())
        print("\nText character length summary:")
        lens = df_n["text"].str.len()
        print(lens.describe().round(0).to_string())
        print("\nSample records:")
        for idx, row in df_n.head(2).iterrows():
            print(f"  [{idx}] URL: {row['source_url']}")
            print(f"      Tone: {row['tone']}")
            print(f"      Text: {row['text'][:120]}...")
    else:
        print(f"ERROR: {news_path} does not exist!")

if __name__ == "__main__":
    verify()
