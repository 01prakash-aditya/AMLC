"""
Stage 6: Cross-Encoder Inference & Post-Processing
==================================================
Amazon ML Challenge - Business Entity Resolution

Reads the test FAISS candidate pairs.
Scores each pair using the fine-tuned Cross-Encoder.
Applies the Dual-Threshold Singleton Guard to extract the final highly precise matches.
"""

import sys
import gc
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers.cross_encoder import CrossEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

def load_preprocessed_texts():
    print("Loading preprocessed test dictionaries...")
    df1 = pd.read_parquet(PROCESSED_DIR / "test_source1_cleaned.parquet")
    df2 = pd.read_parquet(PROCESSED_DIR / "test_source2_cleaned.parquet")
    df3 = pd.read_parquet(PROCESSED_DIR / "test_source3_cleaned.parquet")
    
    df1['ce_text'] = df1['business_name'] + " " + df1['business_address'] + " " + df1['country']
    df2['ce_text'] = df2['business_name'] + " " + df2['business_address'] + " " + df2['country']
    df3['ce_text'] = df3['business_name'] + " " + df3['business_address'] + " " + df3['country']
    
    s1_dict = df1.set_index('entity_id')['ce_text'].to_dict()
    targets = pd.concat([df2, df3], ignore_index=True)
    tgt_dict = targets.set_index('entity_id')['ce_text'].to_dict()
    
    test_s1_ids = df1['entity_id'].tolist()
    
    del df1, df2, df3, targets
    gc.collect()
    return s1_dict, tgt_dict, test_s1_ids

def run_inference():
    model_path = MODEL_DIR / "cross_encoder_model"
    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}. Run Stage 5 first.")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Fine-Tuned Cross-Encoder on {device.upper()}...")
    model = CrossEncoder(str(model_path), device=device)
    
    s1_dict, tgt_dict, test_s1_ids = load_preprocessed_texts()
    
    cands_path = OUTPUT_DIR / "test_candidate_pairs.tsv"
    if not cands_path.exists():
        print(f"Error: {cands_path} not found. Run Stage 3 first.")
        sys.exit(1)
        
    print("Loading test candidates...")
    cands_df = pd.read_csv(cands_path, sep="\t", dtype=str, keep_default_na=False)
    
    pairs = []
    meta = []
    
    for row in cands_df.itertuples(index=False):
        s1 = row.source1_entity_id
        cands = [c.strip() for c in str(row.candidate_entity_ids).split(',') if c.strip()]
        
        for cand in cands:
            text_a = s1_dict.get(s1, "")
            text_b = tgt_dict.get(cand, "")
            if text_a and text_b:
                pairs.append([text_a, text_b])
                meta.append((s1, cand))
                
    print(f"Scoring {len(pairs):,} candidate pairs using Cross-Encoder (this may take a while)...")
    
    # Predict probabilities (applying sigmoid to logits)
    scores = model.predict(pairs, batch_size=256, show_progress_bar=True, apply_softmax=True)
    
    if len(scores.shape) > 1 and scores.shape[1] > 1:
        probs = scores[:, 1]
    else:
        # Some models output raw logits for 1 class
        probs = 1 / (1 + np.exp(-scores))
        
    results_df = pd.DataFrame(meta, columns=['s1_id', 'cand_id'])
    results_df['prob'] = probs
    
    print("\n--- Applying Dual-Threshold Singleton Guard ---")
    # Base threshold for accepting a match
    BASE_THRESH = 0.50
    # Strict threshold required to confirm an entity is NOT a singleton
    SINGLETON_THRESH = 0.70
    
    results_df.sort_values(by=['s1_id', 'prob'], ascending=[True, False], inplace=True)
    
    max_probs = results_df.groupby('s1_id')['prob'].max()
    active_s1_ids = set(max_probs[max_probs > SINGLETON_THRESH].index)
    
    final_matches = []
    for _, row in results_df.iterrows():
        s1 = row['s1_id']
        cand = row['cand_id']
        prob = row['prob']
        
        if s1 in active_s1_ids and prob > BASE_THRESH:
            final_matches.append({'s1_id': s1, 'cand_id': cand})
            
    matches_df = pd.DataFrame(final_matches)
    
    print("Formatting submission files...")
    if len(matches_df) > 0:
        grouped = matches_df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'matched_entity_ids'}, inplace=True)
    else:
        grouped = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    # Fill missing singletons
    missing = set(test_s1_ids) - set(grouped['source1_entity_id'])
    if missing:
        missing_df = pd.DataFrame({'source1_entity_id': list(missing), 'matched_entity_ids': ''})
        grouped = pd.concat([grouped, missing_df], ignore_index=True)
        
    grouped['matched_entity_ids'] = grouped['matched_entity_ids'].fillna('')
    grouped.sort_values('source1_entity_id', inplace=True)
    
    out_path = OUTPUT_DIR / "matching_results.tsv"
    grouped.to_csv(out_path, sep='\t', index=False)
    
    print(f"Saved final matches to {out_path} ({len(grouped):,} rows)")

if __name__ == "__main__":
    run_inference()
