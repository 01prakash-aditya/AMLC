"""
Stage 2: Text Preprocessing & Normalization
===========================================
Amazon ML Challenge - Business Entity Resolution

DEEP LEARNING EDITION:
Unlike the previous TF-IDF pipeline, Deep Learning models (Cross-Encoders)
perform worse when text is aggressively lowercased and stripped of punctuation.
This script simply handles NaN values, formats a clean text string, and saves
as Parquet to ensure the Transformers get the raw contextual clues they need.
"""

import pandas as pd
from pathlib import Path
import os
import gc

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"

def preprocess_for_dl(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.copy()
    
    # Fill missing values with empty strings
    out_df['business_name'] = out_df['business_name'].fillna("").astype(str).str.strip()
    out_df['business_address'] = out_df['business_address'].fillna("").astype(str).str.strip()
    out_df['country'] = out_df['country'].fillna("").astype(str).str.strip()
    
    # Create the concatenated text for FAISS semantic embedding & Cross-Encoder
    out_df['full_text'] = out_df.apply(
        lambda row: f"{row['business_name']}, {row['business_address']} ({row['country']})".replace(" ,", ",").replace(" ()", ""),
        axis=1
    )
    
    # Clean up multi-spaces
    out_df['full_text'] = out_df['full_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return out_df

def process_all_files():
    """Process all raw TSVs and save them as parquet files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    files_to_process = [
        ("train", "train_source1.tsv"),
        ("train", "train_source2.tsv"),
        ("train", "train_source3.tsv"),
        ("test", "test_source1.tsv"),
        ("test", "test_source2.tsv"),
        ("test", "test_source3.tsv")
    ]
    
    for split, filename in files_to_process:
        in_path = DATASET_DIR / split / filename
        out_name = filename.replace('.tsv', '_cleaned.parquet')
        out_path = PROCESSED_DIR / out_name
        
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"  Loaded {len(df):,} rows.")
        
        df_clean = preprocess_for_dl(df)
        
        print(f"  Saving to {out_path}...")
        df_clean.to_parquet(out_path, index=False)
        print("  Done.")
        
        del df, df_clean
        gc.collect()

if __name__ == "__main__":
    process_all_files()
