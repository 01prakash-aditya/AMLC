"""
Stage 5: Pairwise Classification & Training
===========================================
Amazon ML Challenge - Business Entity Resolution

Reads the train_features.parquet generated in Stage 4.
Merges with ground truth labels.
Trains a LightGBM classification model.
Optimizes for the F0.5 precision-heavy metric.
Saves the model to output/models/.
"""

import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def run_training():
    print("Loading training features...")
    feature_file = PROCESSED_DIR / "train_features.parquet"
    if not feature_file.exists():
        print(f"Error: {feature_file} not found. Run Stage 4 first.")
        sys.exit(1)
        
    train_df = pd.read_parquet(feature_file)
    print(f"Loaded {len(train_df):,} feature vectors.")
    
    print("Loading ground truth...")
    gt_df = pd.read_csv(DATASET_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    
    # GT mapping: s1_id -> set of matched target ids
    gt_map = {}
    for _, row in gt_df.iterrows():
        matches = str(row['matched_entity_ids']).split(',')
        gt_map[row['source1_entity_id']] = set(m.strip() for m in matches if m.strip())
        
    # Apply labels
    print("Merging labels into features...")
    def get_label(row):
        s1 = row['s1_id']
        cand = row['cand_id']
        if s1 in gt_map and cand in gt_map[s1]:
            return 1
        return 0
        
    train_df['label'] = train_df.apply(get_label, axis=1)
    
    feature_cols = [c for c in train_df.columns if c not in ('s1_id', 'cand_id', 'label')]
    
    # Simple split by S1 ID
    print("Splitting train/val...")
    s1_ids = train_df['s1_id'].unique()
    np.random.seed(42)
    np.random.shuffle(s1_ids)
    
    split_idx = int(len(s1_ids) * 0.8)
    train_s1 = set(s1_ids[:split_idx])
    
    train_mask = train_df['s1_id'].isin(train_s1)
    
    X_train = train_df[train_mask][feature_cols]
    y_train = train_df[train_mask]['label']
    
    X_val = train_df[~train_mask][feature_cols]
    y_val = train_df[~train_mask]['label']
    
    val_df = train_df[~train_mask].copy()
    
    # Free memory
    del train_df
    gc.collect()
    
    print(f"\nTraining LightGBM on {len(X_train):,} pairs (Val: {len(X_val):,})")
    
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        class_weight='balanced',
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    print("\nFeature Importances:")
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances)
    
    model_path = MODEL_DIR / "lgbm_model.txt"
    model.booster_.save_model(str(model_path))
    print(f"\nModel saved to {model_path}")
    
    # Optimize F0.5
    print("\nOptimizing F0.5 thresholds...")
    val_df['pred_prob'] = model.predict_proba(X_val)[:, 1]
    
    best_f05 = 0
    best_thresh = 0.5
    
    for thresh in np.arange(0.5, 0.95, 0.05):
        val_df['pred'] = (val_df['pred_prob'] > thresh).astype(int)
        
        p = precision_score(val_df['label'], val_df['pred'], zero_division=0)
        r = recall_score(val_df['label'], val_df['pred'], zero_division=0)
        f05 = fbeta_score(val_df['label'], val_df['pred'], beta=0.5, zero_division=0)
        
        print(f"Threshold: {thresh:.2f} | Precision: {p:.3f} | Recall: {r:.3f} | F0.5: {f05:.3f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            
    print(f"\nBest Global Threshold: {best_thresh:.2f} (F0.5 = {best_f05:.3f})")
    
    # Save the feature columns so inference knows exactly what to use
    with open(MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump({"feature_cols": feature_cols, "best_thresh": best_thresh}, f)

if __name__ == "__main__":
    run_training()
