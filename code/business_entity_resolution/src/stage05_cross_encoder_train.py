"""
Stage 5: Cross-Encoder Fine-Tuning
==================================
Amazon ML Challenge - Business Entity Resolution

Fine-tunes a multilingual Transformer Cross-Encoder to predict whether 
two business entities (text_a and text_b) are the same.
Uses the hard-negative mined dataset from Stage 4.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
MODEL_DIR = PROJECT_ROOT / "output" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'cross-encoder/nli-deberta-v3-base' # Exceptional for textual entailment / matching
BATCH_SIZE = 32
EPOCHS = 2

def train_cross_encoder():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base Cross-Encoder model ({MODEL_NAME}) on {device.upper()}...")
    model = CrossEncoder(MODEL_NAME, num_labels=1, device=device)
    
    data_file = PROCESSED_DIR / "cross_encoder_train.parquet"
    if not data_file.exists():
        print(f"Error: {data_file} not found. Run Stage 4 first.")
        sys.exit(1)
        
    print("Loading training dataset...")
    df = pd.read_parquet(data_file)
    
    # Stratified split to keep pos/neg ratio
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    print(f"Training on {len(train_df):,} pairs | Validating on {len(val_df):,} pairs.")
    
    # Convert to InputExamples
    train_samples = []
    for _, row in train_df.iterrows():
        train_samples.append(InputExample(texts=[row['text_a'], row['text_b']], label=float(row['label'])))
        
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
    
    val_samples = []
    for _, row in val_df.iterrows():
        val_samples.append(InputExample(texts=[row['text_a'], row['text_b']], label=float(row['label'])))
        
    # Evaluator to save best model based on F1 / Accuracy
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_samples, name='val')
    
    out_path = MODEL_DIR / "cross_encoder_model"
    
    print(f"Beginning fine-tuning for {EPOCHS} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=EPOCHS,
        evaluation_steps=5000,
        warmup_steps=1000,
        output_path=str(out_path),
        save_best_model=True,
        show_progress_bar=True
    )
    
    print(f"Finished training! Best model saved to {out_path}")

if __name__ == "__main__":
    train_cross_encoder()
