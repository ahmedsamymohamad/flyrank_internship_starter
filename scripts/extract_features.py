import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

# Load .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def extract_features():
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{HF_TOKEN}')")
    
    # We will use two time periods:
    # TRAIN: features from 2026-03, label from 2026-04
    # TEST: features from 2026-05, label from 2026-06

    print("Extracting training features (month=2026-03)...")
    train_feat = con.sql("""
        SELECT client_hash_id, content_hash_id, 
               SUM(gsc_clicks) AS clicks_feat, 
               SUM(gsc_impressions) AS impressions_feat,
               AVG(gsc_avg_position) AS avg_pos_feat
        FROM read_parquet('hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-03/*.parquet')
        GROUP BY client_hash_id, content_hash_id
    """).df()

    print("Extracting training labels (month=2026-04)...")
    train_label = con.sql("""
        SELECT client_hash_id, content_hash_id, 
               SUM(gsc_clicks) AS clicks_label,
               SUM(gsc_impressions) AS impressions_label
        FROM read_parquet('hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-04/*.parquet')
        GROUP BY client_hash_id, content_hash_id
    """).df()

    print("Extracting test features (month=2026-05)...")
    test_feat = con.sql("""
        SELECT client_hash_id, content_hash_id, 
               SUM(gsc_clicks) AS clicks_feat, 
               SUM(gsc_impressions) AS impressions_feat,
               AVG(gsc_avg_position) AS avg_pos_feat
        FROM read_parquet('hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-05/*.parquet')
        GROUP BY client_hash_id, content_hash_id
    """).df()

    print("Extracting test labels (month=2026-06)...")
    test_label = con.sql("""
        SELECT client_hash_id, content_hash_id, 
               SUM(gsc_clicks) AS clicks_label,
               SUM(gsc_impressions) AS impressions_label
        FROM read_parquet('hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-06/*.parquet')
        GROUP BY client_hash_id, content_hash_id
    """).df()

    # Get static content dimensions
    print("Extracting dim_content...")
    dim_content = con.sql("""
        SELECT content_hash_id, content_type, search_volume, competition, cpc, main_intent, char_count, word_count
        FROM read_parquet('hf://datasets/FlyRank/internship-warehouse/dim_content.parquet')
    """).df()

    print("Merging data...")
    # Train
    train_df = pd.merge(train_feat, train_label, on=["client_hash_id", "content_hash_id"], how="left")
    train_df["split"] = "train"
    
    # Test
    test_df = pd.merge(test_feat, test_label, on=["client_hash_id", "content_hash_id"], how="left")
    test_df["split"] = "test"
    
    # Combine
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Merge dim_content
    df = pd.merge(df, dim_content, on="content_hash_id", how="left")
    
    # Fill missing labels with 0
    df["clicks_label"] = df["clicks_label"].fillna(0)
    df["impressions_label"] = df["impressions_label"].fillna(0)
    
    # Define label: is_declining = clicks dropped by at least 20% compared to feature window, AND minimum clicks in feature window > 10 (to avoid noise)
    df["is_declining_label"] = ((df["clicks_label"] < 0.8 * df["clicks_feat"]) & (df["clicks_feat"] >= 10)).astype(int)

    os.makedirs("work/outputs", exist_ok=True)
    output_path = "work/outputs/capstone_features.parquet"
    df.to_parquet(output_path)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    extract_features()
