"""
Stage 3: Dense Semantic Candidate Generation (FAISS Blocking)
=============================================================
Amazon ML Challenge - Business Entity Resolution

Generates candidate pairs using a multilingual Bi-Encoder transformer.
1. Encodes all entities (Name + Address) into dense embeddings.
2. Builds a FAISS index on Target (Source 2 & 3) embeddings.
3. Queries the FAISS index with Source 1 embeddings to find Top-K matches.

This handles translations (Hindi <-> English, French <-> English) and 
semantic variations natively, far surpassing TF-IDF recall.
"""

import os
import sys
import gc
import time
import torch
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'  # Fast, highly effective multilingual
BATCH_SIZE = 1024

def load_data(is_test=False):
    prefix = "test" if is_test else "train"
    print(f"Loading preprocessed {prefix} data...")
    df1 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source1_cleaned.parquet")
    df2 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source2_cleaned.parquet")
    df3 = pd.read_parquet(PROCESSED_DIR / f"{prefix}_source3_cleaned.parquet")
    return df1, df2, df3

def generate_embeddings(model, texts):
    """Encode a list/series of texts into dense vectors."""
    return model.encode(
        texts.tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,  # Crucial for FAISS Inner Product (Cosine Similarity)
        convert_to_numpy=True
    )

def process_target_chunk(model, df1, target_df, device, top_k=15, prefix="S2"):
    """Encodes a target DF, builds a FAISS index, and queries it with df1."""
    print(f"\n--- Encoding Targets ({prefix}) ---")
    target_ids = target_df['entity_id'].values
    
    # Generate embeddings in chunks to save memory
    target_embeddings = generate_embeddings(model, target_df['full_text'])
    dim = target_embeddings.shape[1]
    
    print(f"--- Building FAISS Index ({prefix}) ---")
    index = faiss.IndexFlatIP(dim)
    if device == "cuda":
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
    index.add(target_embeddings)
    print(f"Successfully indexed {index.ntotal:,} vectors.")
    
    del target_embeddings
    gc.collect()
    
    print(f"--- Encoding Queries (Source 1) & Searching FAISS ({prefix}) ---")
    query_embeddings = generate_embeddings(model, df1['full_text'])
    s1_ids = df1['entity_id'].values
    
    distances, indices = index.search(query_embeddings, top_k)
    
    pairs = []
    for i in range(len(s1_ids)):
        s1 = s1_ids[i]
        for j in range(top_k):
            idx = indices[i][j]
            score = distances[i][j]
            if score > 0.65:
                pairs.append({'s1_id': s1, 'cand_id': target_ids[idx], 'score': score})
                
    del index, query_embeddings
    gc.collect()
    
    return pairs

def run_semantic_blocking(is_test=False, top_k=15):
    t0 = time.time()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Bi-Encoder Model: {MODEL_NAME} on {device.upper()}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    try:
        df1, df2, df3 = load_data(is_test=is_test)
    except FileNotFoundError:
        print("Preprocessed files not found. Run Stage 2 first.")
        sys.exit(1)
        
    all_pairs = []
    
    # Process Source 2
    s2_pairs = process_target_chunk(model, df1, df2, device, top_k=top_k, prefix="Source 2")
    all_pairs.extend(s2_pairs)
    del df2, s2_pairs
    gc.collect()
    
    # Process Source 3
    s3_pairs = process_target_chunk(model, df1, df3, device, top_k=top_k, prefix="Source 3")
    all_pairs.extend(s3_pairs)
    del df3, s3_pairs
    gc.collect()
    
    cands_df = pd.DataFrame(all_pairs)
    print(f"\nGenerated {len(cands_df):,} total candidate pairs after similarity thresholding.")
    
    # Group for output format
    print("Formatting output...")
    grouped = cands_df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
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
    print("=== RUNNING FAISS BLOCKING ON TRAIN SET ===")
    run_semantic_blocking(is_test=False)
    
    print("\n=== RUNNING FAISS BLOCKING ON TEST SET ===")
    run_semantic_blocking(is_test=True)
