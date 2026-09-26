"""
Stage 3: Multi-Signal Candidate Generation (Blocking)
=====================================================
Amazon ML Challenge - Business Entity Resolution

Generates candidate pairs for Source 1 entities against Source 2 and Source 3.
Uses multiple overlapping blocking signals to maximize recall:
    Signal 1: TF-IDF Char N-grams on Full Text (Name + Address)
    Signal 2: TF-IDF Word N-grams on Name Only
    Signal 3: TF-IDF Char N-grams on Name Only (For differing addresses)

Utilizes fast sparse matrix multiplication (sparse_dot_topn).
Loads data from the preprocessed Parquet files generated in Stage 2.
"""

import os
import sys
import gc
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sparse

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:
    print("Please install sparse_dot_topn: pip install sparse_dot_topn")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"

def load_data(is_test=False):
    """Load the preprocessed Parquet data."""
    prefix = "test" if is_test else "train"
    print(f"Loading preprocessed {prefix} data...")
    
    df1 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source1_cleaned.parquet")
    df2 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source2_cleaned.parquet")
    df3 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source3_cleaned.parquet")
    
    return df1, df2, df3

def build_candidates(df1: pd.DataFrame, target_df: pd.DataFrame, text_col: str, 
                     analyzer: str = 'char_wb', ngram_range: tuple = (3,4), 
                     top_k: int = 15, threshold: float = 0.35, max_features: int = 100000):
    """
    Find top-k matches for df1 in target_df using TF-IDF on `text_col`.
    Uses chunking to avoid RAM explosions on huge dense matrices.
    """
    print(f"  Building TF-IDF ({analyzer} {ngram_range}) on '{text_col}' for {len(df1):,} queries vs {len(target_df):,} targets...")
    
    # max_features restricts the vocabulary to the most frequent n-grams, drastically reducing matrix size
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=3, max_df=0.8, max_features=max_features)
    vectorizer.fit(target_df[text_col])
    
    target_tfidf = vectorizer.transform(target_df[text_col])
    query_tfidf = vectorizer.transform(df1[text_col])
    
    print(f"  Computing top {top_k} nearest neighbors (threshold {threshold})...")
    
    target_tfidf_t = target_tfidf.transpose().tocsr()
    query_tfidf = query_tfidf.tocsr()
    
    # Process queries in chunks to prevent memory explosion
    chunk_size = 250000
    n_queries = query_tfidf.shape[0]
    results_list = []
    
    s1_ids = df1['entity_id'].values
    target_ids = target_df['entity_id'].values
    
    for start_idx in range(0, n_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, n_queries)
        print(f"    -> Chunk {start_idx:,} to {end_idx:,} / {n_queries:,}")
        
        q_chunk = query_tfidf[start_idx:end_idx]
        
        # Using the modern sparse_dot_topn API (1 thread per chunk usually safest in Jupyter)
        matches = sp_matmul_topn(
            q_chunk, 
            target_tfidf_t, 
            top_n=top_k, 
            threshold=threshold,
            n_threads=1 
        ).tocoo()
        
        chunk_s1_ids = s1_ids[start_idx:end_idx]
        
        results_list.append(pd.DataFrame({
            's1_id': chunk_s1_ids[matches.row],
            'cand_id': target_ids[matches.col],
            'score': matches.data
        }))
        
    results = pd.concat(results_list, ignore_index=True) if results_list else pd.DataFrame()
    
    del query_tfidf, target_tfidf, target_tfidf_t, vectorizer, results_list
    gc.collect()
    
    return results

def run_multi_signal_blocking(is_test=False):
    t0 = time.time()
    
    try:
        df1, df2, df3 = load_data(is_test=is_test)
    except FileNotFoundError:
        print("Preprocessed parquet files not found! Please run Stage 2 first.")
        sys.exit(1)
    
    all_candidates = []
    
    # -------------------------------------------------------------------------
    # SIGNAL 1: Char N-Grams on Full Text (Name + Address)
    # High recall for fuzzy misspellings across both fields
    # -------------------------------------------------------------------------
    print("\n[SIGNAL 1] TF-IDF Char N-Grams on Full Text")
    c1_s2 = build_candidates(df1, df2, text_col='full_text', analyzer='char_wb', ngram_range=(3,4), top_k=10, threshold=0.35, max_features=100000)
    c1_s3 = build_candidates(df1, df3, text_col='full_text', analyzer='char_wb', ngram_range=(3,4), top_k=10, threshold=0.35, max_features=100000)
    all_candidates.extend([c1_s2, c1_s3])
    del c1_s2, c1_s3; gc.collect()
    
    # -------------------------------------------------------------------------
    # SIGNAL 2: Word N-Grams on Name Only
    # Catches exact/near-exact names even if addresses are completely different
    # -------------------------------------------------------------------------
    print("\n[SIGNAL 2] TF-IDF Word N-Grams on Name Only")
    c2_s2 = build_candidates(df1, df2, text_col='name_norm', analyzer='word', ngram_range=(1,2), top_k=5, threshold=0.45, max_features=80000)
    c2_s3 = build_candidates(df1, df3, text_col='name_norm', analyzer='word', ngram_range=(1,2), top_k=5, threshold=0.45, max_features=80000)
    all_candidates.extend([c2_s2, c2_s3])
    del c2_s2, c2_s3; gc.collect()

    # -------------------------------------------------------------------------
    # SIGNAL 3: Char N-Grams on Name Only
    # Catches misspelled names with missing/garbage addresses
    # -------------------------------------------------------------------------
    print("\n[SIGNAL 3] TF-IDF Char N-Grams on Name Only")
    c3_s2 = build_candidates(df1, df2, text_col='name_norm', analyzer='char_wb', ngram_range=(3,4), top_k=5, threshold=0.40, max_features=80000)
    c3_s3 = build_candidates(df1, df3, text_col='name_norm', analyzer='char_wb', ngram_range=(3,4), top_k=5, threshold=0.40, max_features=80000)
    all_candidates.extend([c3_s2, c3_s3])
    del c3_s2, c3_s3; gc.collect()
    
    print("\nCombining and deduplicating multi-signal candidates...")
    combined_cands = pd.concat(all_candidates, ignore_index=True)
    del all_candidates; gc.collect()
    
    # Deduplicate by picking the highest score per pair across all signals
    combined_cands = combined_cands.sort_values('score', ascending=False).drop_duplicates(subset=['s1_id', 'cand_id'])
    
    print(f"Total unique candidate pairs generated: {len(combined_cands):,}")
    
    print("Formatting output...")
    # Group by S1 ID to list
    grouped = combined_cands.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
    grouped.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'candidate_entity_ids'}, inplace=True)
    
    # Ensure all S1 IDs are present
    missing = set(df1['entity_id']) - set(grouped['source1_entity_id'])
    if missing:
        missing_df = pd.DataFrame({'source1_entity_id': list(missing), 'candidate_entity_ids': ''})
        grouped = pd.concat([grouped, missing_df], ignore_index=True)
        
    grouped.sort_values('source1_entity_id', inplace=True)
    
    out_file = OUTPUT_DIR / ("test_candidate_pairs.tsv" if is_test else "train_candidate_pairs.tsv")
    grouped.to_csv(out_file, sep='\t', index=False)
    
    print(f"\nDone! Saved {len(grouped):,} rows to {out_file}")
    print(f"Total Time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    # Standard execution: Run on train, then test.
    print("=== RUNNING MULTI-SIGNAL BLOCKING ON TRAIN SET ===")
    run_multi_signal_blocking(is_test=False)
    
    print("\n=== RUNNING MULTI-SIGNAL BLOCKING ON TEST SET ===")
    run_multi_signal_blocking(is_test=True)
