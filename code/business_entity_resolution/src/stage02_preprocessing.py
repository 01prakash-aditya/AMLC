"""
Stage 2: Text Preprocessing & Normalization
===========================================
Amazon ML Challenge - Business Entity Resolution

This module provides optimized functions for normalizing noisy business names,
addresses, and countries. It uses vectorized pandas operations where possible.
"""

import re
import unicodedata
import pandas as pd
from typing import Optional

# Pre-compile regexes for performance on 12M rows
RE_PUNCT_CLEAN = re.compile(r'[^\w\s]')
RE_MULTI_SPACE = re.compile(r'\s+')
RE_PINCODE = re.compile(r'\b\d{5,6}\b')
RE_LANDMARK = re.compile(r'\b(near|opp|opposite|behind|beside|above|below)\b.*', re.IGNORECASE)

# Legal suffix normalization mapping
LEGAL_SUFFIXES = {
    'llc': 'llc', 'l l c': 'llc',
    'inc': 'inc', 'incorporated': 'inc',
    'corp': 'corp', 'corporation': 'corp',
    'co': 'co', 'company': 'co',
    'ltd': 'ltd', 'limited': 'ltd',
    'pvt': 'pvt', 'private': 'pvt',
    'llp': 'llp',
    'sarl': 'sarl', 'sa': 'sa', 'sas': 'sas'
}

# Address abbreviations mapping
ADDR_ABBR = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue',
    'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane',
    'ste': 'suite', 'apt': 'apartment', 'bldg': 'building',
    'opp': 'opposite', 'nr': 'near',
    'dist': 'district', 'distt': 'district'
}

def clean_text_basic(text: pd.Series) -> pd.Series:
    """Basic lowercasing, unicode normalization, and space trimming."""
    s = text.fillna("").astype(str).str.lower()
    # Unicode normalization (NFKD removes accents: é -> e)
    s = s.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8'))
    return s

def normalize_names(names: pd.Series) -> pd.Series:
    """Normalize business names."""
    s = clean_text_basic(names)
    s = s.str.replace(RE_PUNCT_CLEAN, ' ', regex=True)
    for variant, standard in LEGAL_SUFFIXES.items():
        s = s.str.replace(rf'\b{variant}\b', standard, regex=True)
    s = s.str.replace(RE_MULTI_SPACE, ' ', regex=True).str.strip()
    return s

def normalize_addresses(addresses: pd.Series) -> pd.Series:
    """Normalize business addresses, including landmark removal."""
    s = clean_text_basic(addresses)
    
    # Strip out landmark descriptions (e.g., 'near SBI ATM...', 'opp City Mall')
    s = s.str.replace(RE_LANDMARK, ' ', regex=True)
    
    # Remove punctuation
    s = s.str.replace(RE_PUNCT_CLEAN, ' ', regex=True)
    
    # Standardize abbreviations across US, India, France
    for variant, standard in ADDR_ABBR.items():
        s = s.str.replace(rf'\b{variant}\b', standard, regex=True)
    
    s = s.str.replace(RE_MULTI_SPACE, ' ', regex=True).str.strip()
    return s

def extract_pincode(addresses: pd.Series) -> pd.Series:
    """Extract 5 (US/FR) or 6 (IN) digit PIN/ZIP codes from addresses."""
    return addresses.astype(str).apply(
        lambda x: match.group(0) if (match := RE_PINCODE.search(x)) else ""
    )

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing to a dataframe and return it."""
    out_df = df.copy()
    
    print("  Normalizing names...")
    out_df['name_norm'] = normalize_names(df['business_name'])
    
    print("  Normalizing addresses...")
    out_df['address_norm'] = normalize_addresses(df['business_address'])
    
    print("  Extracting PIN/ZIP codes...")
    out_df['pincode'] = extract_pincode(df['business_address'])
    
    print("  Normalizing country...")
    out_df['country_norm'] = clean_text_basic(df['country'])
    
    # Create a concatenated field for full-text search / blocking
    out_df['full_text'] = out_df['name_norm'] + " " + out_df['address_norm']
    
    # Clean up empty strings
    out_df = out_df.replace(r'^\s*$', "", regex=True)
    
    return out_df

def process_all_files():
    """Process all raw TSVs and save them as parquet files for faster downstream loading."""
    from pathlib import Path
    import os
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_DIR = PROJECT_ROOT / "dataset"
    PROCESSED_DIR = PROJECT_ROOT / "output" / "processed"
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
        
        if out_path.exists():
            print(f"Skipping {filename}, already processed: {out_path}")
            continue
            
        print(f"\nProcessing {filename}...")
        df = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False)
        print(f"  Loaded {len(df):,} rows.")
        
        df_clean = preprocess_dataframe(df)
        
        print(f"  Saving to {out_path}...")
        df_clean.to_parquet(out_path, index=False)
        print("  Done.")
        
        # Free memory
        del df
        del df_clean
        import gc
        gc.collect()

if __name__ == "__main__":
    process_all_files()
