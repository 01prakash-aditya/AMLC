"""
Stage 6: Inference & Post-Processing
====================================
Amazon ML Challenge - Business Entity Resolution

Reads test_features.parquet generated in Stage 4.
Loads the trained LightGBM model from Stage 5.
Generates final predictions on the test set, applies the dual-threshold
singleton guard strategy, and writes output/matching_results.tsv.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "output" / "models"

def run_inference():
    feature_file = PROCESSED_DIR / "test_features.parquet"
    if not feature_file.exists():
        print(f"Error: {feature_file} not found. Run Stage 4 first.")
        sys.exit(1)
        
    print("Loading test features...")
    test_df = pd.read_parquet(feature_file)
    print(f"Loaded {len(test_df):,} test candidates.")
    
    # Load model configuration
    config_file = MODEL_DIR / "feature_cols.json"
    if not config_file.exists():
        print(f"Error: {config_file} not found. Run Stage 5 first.")
        sys.exit(1)
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    feature_cols = config['feature_cols']
    best_thresh = config['best_thresh']
    
    model_path = MODEL_DIR / "lgbm_model.txt"
    print(f"Loading model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))
    
    print("Predicting probabilities...")
    test_df['prob'] = model.predict(test_df[feature_cols])
    
    # Dual thresholds to aggressively protect singletons from false-positives
    base_threshold = best_thresh - 0.1
    singleton_threshold = best_thresh + 0.1
    
    print(f"Applying dual thresholds (base: {base_threshold:.2f}, singleton: {singleton_threshold:.2f})...")
    
    # Sort by probability
    test_df.sort_values(by=['s1_id', 'prob'], ascending=[True, False], inplace=True)
    
    # Get max prob per S1 entity
    max_probs = test_df.groupby('s1_id')['prob'].max()
    active_s1_ids = set(max_probs[max_probs > singleton_threshold].index)
    
    final_matches = []
    for _, row in test_df.iterrows():
        s1 = row['s1_id']
        cand = row['cand_id']
        prob = row['prob']
        
        if s1 in active_s1_ids and prob > base_threshold:
            final_matches.append({'s1_id': s1, 'cand_id': cand})
            
    matches_df = pd.DataFrame(final_matches)
    
    print("Formatting submission files...")
    if len(matches_df) > 0:
        grouped = matches_df.groupby('s1_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
        grouped.rename(columns={'s1_id': 'source1_entity_id', 'cand_id': 'matched_entity_ids'}, inplace=True)
    else:
        grouped = pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids'])
        
    # Get original test S1 IDs
    test_s1 = pd.read_csv(DATASET_DIR / "test" / "test_source1.tsv", sep="\t", dtype=str)
    test_s1_ids = test_s1['entity_id'].tolist()
    
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
