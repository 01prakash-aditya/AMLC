"""
Stage 1: Data Ingestion & Exploratory Data Analysis
====================================================
Amazon ML Challenge - Business Entity Resolution

Optimized for LARGE datasets (200-500MB TSVs, millions of rows).
Uses chunked reading, sampling, and memory-efficient operations.

Usage:
    python -X utf8 code/business_entity_resolution/src/01_data_ingestion_eda.py

Output:
    - output/eda_report.txt
    - output/eda_plots/*.png
    - output/processed/eda_summary.json
    - output/processed/ground_truth_parsed.json
    - output/processed/val_split.json
"""

import os
import sys
import io
import json
import re
import warnings
import time
from collections import Counter, defaultdict
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
EDA_PLOTS_DIR = OUTPUT_DIR / "eda_plots"
PROCESSED_DIR = OUTPUT_DIR / "processed"

for d in [OUTPUT_DIR, EDA_PLOTS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

report_lines = []
T0 = time.time()


def elapsed():
    return f"[{time.time() - T0:.1f}s]"


def log(msg: str = ""):
    print(f"{elapsed()} {msg}")
    report_lines.append(msg)


def section(title: str):
    log("")
    log("=" * 80)
    log(f"  {title}")
    log("=" * 80)


# ============================================================================
# 1. DATA LOADING — count rows first, then load
# ============================================================================
section("1. DATA LOADING")


def count_lines(path: Path) -> int:
    """Fast line count for large files."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count - 1  # subtract header


def load_tsv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        log(f"  [ERROR] File not found: {path}")
        sys.exit(1)

    n_lines = count_lines(path)
    size_mb = path.stat().st_size / (1024 * 1024)
    log(f"  Loading {name}: ~{n_lines:,} rows, {size_mb:.1f} MB ...")

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     encoding="utf-8", on_bad_lines="skip")
    log(f"    Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols | Columns: {list(df.columns)}")
    return df


# Training data
train_s1 = load_tsv(TRAIN_DIR / "train_source1.tsv", "train_source1")
train_s2 = load_tsv(TRAIN_DIR / "train_source2.tsv", "train_source2")
train_s3 = load_tsv(TRAIN_DIR / "train_source3.tsv", "train_source3")
train_gt = load_tsv(TRAIN_DIR / "train_ground_truth.tsv", "train_ground_truth")

# Test data
test_s1 = load_tsv(TEST_DIR / "test_source1.tsv", "test_source1")
test_s2 = load_tsv(TEST_DIR / "test_source2.tsv", "test_source2")
test_s3 = load_tsv(TEST_DIR / "test_source3.tsv", "test_source3")


# ============================================================================
# 2. SCHEMA VALIDATION
# ============================================================================
section("2. SCHEMA VALIDATION")

EXPECTED_SOURCE_COLS = {"entity_id", "business_name", "business_address", "country"}
EXPECTED_GT_COLS = {"source1_entity_id", "matched_entity_ids"}

for name, df in [
    ("train_s1", train_s1), ("train_s2", train_s2), ("train_s3", train_s3),
    ("test_s1", test_s1), ("test_s2", test_s2), ("test_s3", test_s3),
]:
    actual = set(df.columns)
    if actual != EXPECTED_SOURCE_COLS:
        log(f"  [WARNING] {name} columns mismatch! Expected: {EXPECTED_SOURCE_COLS}, Got: {actual}")
    else:
        log(f"  [OK] {name} schema valid")

gt_actual = set(train_gt.columns)
if gt_actual != EXPECTED_GT_COLS:
    log(f"  [WARNING] ground_truth columns mismatch! Expected: {EXPECTED_GT_COLS}, Got: {gt_actual}")
else:
    log(f"  [OK] ground_truth schema valid")


# ============================================================================
# 3. BASIC STATISTICS
# ============================================================================
section("3. BASIC STATISTICS")

for name, df in [
    ("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3),
    ("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3),
]:
    n_rows = len(df)
    n_null_name = (df["business_name"] == "").sum()
    n_null_addr = (df["business_address"] == "").sum()
    n_null_country = (df["country"] == "").sum()
    n_unique_ids = df["entity_id"].nunique()
    prefix = df["entity_id"].str[:3].value_counts().to_dict()

    log(f"\n  {name}:")
    log(f"    Rows: {n_rows:,}  |  Unique IDs: {n_unique_ids:,}  |  ID duplicates: {n_rows - n_unique_ids}")
    log(f"    Empty business_name: {n_null_name} ({100*n_null_name/n_rows:.1f}%)")
    log(f"    Empty business_address: {n_null_addr} ({100*n_null_addr/n_rows:.1f}%)")
    log(f"    Empty country: {n_null_country} ({100*n_null_country/n_rows:.1f}%)")
    log(f"    ID prefix distribution: {prefix}")


# ============================================================================
# 4. COUNTRY DISTRIBUTION
# ============================================================================
section("4. COUNTRY DISTRIBUTION")

for name, df in [
    ("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3),
    ("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3),
]:
    country_dist = df["country"].value_counts()
    log(f"\n  {name}:")
    for country, count in country_dist.items():
        log(f"    {country:20s}: {count:>8,} ({100*count/len(df):5.1f}%)")


# ============================================================================
# 5. GROUND TRUTH ANALYSIS
# ============================================================================
section("5. GROUND TRUTH ANALYSIS")

# Parse matched_entity_ids
def parse_matches(val):
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip() for x in str(val).split(",") if x.strip()]

train_gt["match_list"] = train_gt["matched_entity_ids"].apply(parse_matches)
train_gt["match_count"] = train_gt["match_list"].apply(len)
train_gt["has_match"] = train_gt["match_count"] > 0

total_s1 = len(train_gt)
n_with_matches = train_gt["has_match"].sum()
n_singletons = total_s1 - n_with_matches
singleton_rate = n_singletons / total_s1

log(f"\n  Total Source 1 entities in GT: {total_s1:,}")
log(f"  With matches (non-singleton): {n_with_matches:,} ({100*n_with_matches/total_s1:.1f}%)")
log(f"  Singletons (no matches):      {n_singletons:,} ({100*n_singletons/total_s1:.1f}%)")
log(f"  Singleton rate: {singleton_rate:.4f}")

# Match count distribution
log(f"\n  Match count distribution:")
match_dist = train_gt["match_count"].value_counts().sort_index()
for count, freq in match_dist.head(20).items():
    log(f"    {count} matches: {freq:>8,} entities ({100*freq/total_s1:5.1f}%)")
if len(match_dist) > 20:
    log(f"    ... ({len(match_dist)} distinct match counts total)")

# Non-singleton stats
non_single = train_gt[train_gt["has_match"]]["match_count"]
if len(non_single) > 0:
    log(f"\n  Match count stats (non-singleton):")
    log(f"    Mean: {non_single.mean():.2f}  |  Median: {non_single.median():.1f}  |  "
        f"Min: {non_single.min()}  |  Max: {non_single.max()}  |  Std: {non_single.std():.2f}")

# Source breakdown
all_match_ids = [mid for mlist in train_gt["match_list"] for mid in mlist]
s2_matches = [m for m in all_match_ids if m.startswith("S2-")]
s3_matches = [m for m in all_match_ids if m.startswith("S3-")]
log(f"\n  Total matched IDs: {len(all_match_ids):,}")
log(f"    From Source 2: {len(s2_matches):,} ({100*len(s2_matches)/max(1,len(all_match_ids)):.1f}%)")
log(f"    From Source 3: {len(s3_matches):,} ({100*len(s3_matches)/max(1,len(all_match_ids)):.1f}%)")

# S2/S3 overlap
has_s2 = train_gt["match_list"].apply(lambda x: any(m.startswith("S2-") for m in x))
has_s3 = train_gt["match_list"].apply(lambda x: any(m.startswith("S3-") for m in x))
has_both = (has_s2 & has_s3).sum()
has_only_s2 = (has_s2 & ~has_s3).sum()
has_only_s3 = (~has_s2 & has_s3).sum()
log(f"\n  Source coverage of non-singleton entities:")
log(f"    Matches from S2 only:     {has_only_s2:,}")
log(f"    Matches from S3 only:     {has_only_s3:,}")
log(f"    Matches from both S2+S3:  {has_both:,}")

# Duplicate check
match_id_counter = Counter(all_match_ids)
duplicated_matches = {k: v for k, v in match_id_counter.items() if v > 1}
log(f"\n  S2/S3 IDs matched to multiple S1 entities: {len(duplicated_matches)}")
if duplicated_matches:
    top_dups = sorted(duplicated_matches.items(), key=lambda x: -x[1])[:10]
    for mid, cnt in top_dups:
        log(f"    {mid}: appears in {cnt} S1 entity match lists")


# ============================================================================
# 6. NAME & ADDRESS LENGTH ANALYSIS
# ============================================================================
section("6. NAME & ADDRESS LENGTH ANALYSIS")

for name, df in [
    ("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3),
]:
    name_lens = df["business_name"].str.len()
    addr_lens = df["business_address"].str.len()
    log(f"\n  {name} — Business Name:")
    log(f"    Length: Mean={name_lens.mean():.1f}  Median={name_lens.median():.0f}  "
        f"Min={name_lens.min()}  Max={name_lens.max()}  Std={name_lens.std():.1f}")

    name_words = df["business_name"].str.split().apply(len)
    log(f"    Words:  Mean={name_words.mean():.1f}  Max={name_words.max()}")

    log(f"  {name} — Business Address:")
    log(f"    Length: Mean={addr_lens.mean():.1f}  Median={addr_lens.median():.0f}  "
        f"Min={addr_lens.min()}  Max={addr_lens.max()}  Std={addr_lens.std():.1f}")

    addr_nonempty = df[df["business_address"] != ""]
    if len(addr_nonempty) > 0:
        addr_words = addr_nonempty["business_address"].str.split().apply(len)
        log(f"    Words:  Mean={addr_words.mean():.1f}  Max={addr_words.max()}")


# ============================================================================
# 7. NOISE PATTERN ANALYSIS (SAMPLED)
# ============================================================================
section("7. NOISE PATTERN ANALYSIS (100k sample per source)")

sample_size = 100000
s1_sample = train_s1.sample(min(len(train_s1), sample_size), random_state=42)
s2_sample = train_s2.sample(min(len(train_s2), sample_size), random_state=42)
s3_sample = train_s3.sample(min(len(train_s3), sample_size), random_state=42)

# 7a. Most common tokens in names
log("\n  7a. Top 40 Name Tokens (sampled):")
all_names = pd.concat([s1_sample["business_name"], s2_sample["business_name"], s3_sample["business_name"]])
all_name_tokens = Counter()
for name_str in all_names:
    if name_str:
        tokens = str(name_str).lower().split()
        all_name_tokens.update(tokens)

log(f"    Unique tokens in sample: {len(all_name_tokens):,}")
for token, freq in all_name_tokens.most_common(40):
    log(f"      {token:25s}: {freq:>8,}")

# 7b. Legal suffix patterns
log("\n  7b. Legal Suffix Frequency (sampled):")
legal_patterns = [
    (r"\bllc\b", "LLC"), (r"\bllp\b", "LLP"), (r"\binc\b\.?", "Inc"),
    (r"\bcorp\b\.?", "Corp"), (r"\bco\b\.?", "Co"),
    (r"\bltd\b\.?", "Ltd"), (r"\blimited\b", "Limited"),
    (r"\bpvt\b\.?", "Pvt"), (r"\bprivate\b", "Private"),
    (r"\bcorporation\b", "Corporation"), (r"\bincorporated\b", "Incorporated"),
    (r"\bsarl\b", "SARL"), (r"\bsa\b", "SA"), (r"\bsas\b", "SAS"), (r"\beurl\b", "EURL"),
    (r"\benterprises?\b", "Enterprise(s)"), (r"\btraders?\b", "Trader(s)"),
    (r"\bgroup\b", "Group"), (r"\bholdings?\b", "Holding(s)"),
    (r"\bservices?\b", "Service(s)"), (r"\bsolutions?\b", "Solution(s)"),
    (r"\btechnolog(y|ies)\b", "Technology"), (r"\bindustries\b", "Industries"),
]

for pattern, label in legal_patterns:
    counts = {}
    for sname, df in [("S1", s1_sample), ("S2", s2_sample), ("S3", s3_sample)]:
        n = df["business_name"].str.lower().str.contains(pattern, regex=True, na=False).sum()
        counts[sname] = n
    total = sum(counts.values())
    if total > 0:
        log(f"    {label:20s}: S1={counts['S1']:>7,}  S2={counts['S2']:>7,}  S3={counts['S3']:>7,}  Total={total:>8,}")

# 7c. Special characters
log("\n  7c. Special Characters in Names (sampled):")
for ch in ["&", ".", ",", "-", "'", '"', "(", ")", "/", "#"]:
    counts = {}
    for sname, df in [("S1", s1_sample), ("S2", s2_sample), ("S3", s3_sample)]:
        n = df["business_name"].str.contains(re.escape(ch), na=False).sum()
        counts[sname] = n
    total = sum(counts.values())
    if total > 0:
        log(f"    '{ch}': S1={counts['S1']:>7,}  S2={counts['S2']:>7,}  S3={counts['S3']:>7,}  Total={total:>8,}")

# 7d. Address patterns
log("\n  7d. Address Noise Patterns (sampled):")
all_addrs = pd.concat([s1_sample["business_address"], s2_sample["business_address"], s3_sample["business_address"]])
addr_patterns = [
    (r"\bnear\b", "Near (landmark)"),
    (r"\bopp(osite)?\b", "Opposite"),
    (r"\bbehind\b", "Behind"),
    (r"\brd\b\.?", "Rd (Road)"),
    (r"\bst\b\.?", "St (Street)"),
    (r"\bave\b\.?", "Ave (Avenue)"),
    (r"\bblvd\b\.?", "Blvd"),
    (r"\bste\b\.?", "Suite"),
    (r"\brue\b", "Rue (French)"),
    (r"\bcedex\b", "Cedex (French)"),
    (r"\bnagar\b", "Nagar (Indian)"),
    (r"\bmarg\b", "Marg (Indian)"),
    (r"\bsector\b", "Sector"),
    (r"\b\d{6}\b", "6-digit PIN"),
    (r"\b\d{5}\b", "5-digit ZIP"),
]
for pattern, desc in addr_patterns:
    n = all_addrs.str.lower().str.contains(pattern, regex=True, na=False).sum()
    if n > 0:
        log(f"    {desc:25s}: {n:>8,}")

del all_addrs  # free memory


# ============================================================================
# 8. CROSS-SOURCE SIMILARITY (SAMPLED)
# ============================================================================
section("8. CROSS-SOURCE SIMILARITY (SAMPLED)")

log("  Loading rapidfuzz for similarity computation...")
from rapidfuzz import fuzz as rfuzz, distance as rf_distance

# Build lookups
s1_lookup = train_s1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
s2_lookup = train_s2.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
s3_lookup = train_s3.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
s2s3_lookup = {**s2_lookup, **s3_lookup}

log(f"  Built lookups: S1={len(s1_lookup):,}, S2={len(s2_lookup):,}, S3={len(s3_lookup):,}")

# Sample matched pairs
matched_gt = train_gt[train_gt["has_match"]]
SAMPLE_SIZE = min(3000, len(matched_gt))
sample_gt = matched_gt.sample(n=SAMPLE_SIZE, random_state=42) if len(matched_gt) > SAMPLE_SIZE else matched_gt

name_lev_sims = []
name_ts_sims = []
name_jw_sims = []
addr_lev_sims = []

n_analyzed = 0
for _, row in sample_gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in s1_lookup:
        continue
    s1_rec = s1_lookup[s1_id]

    for match_id in row["match_list"]:
        if match_id not in s2s3_lookup:
            continue
        match_rec = s2s3_lookup[match_id]

        n1 = str(s1_rec["business_name"]).lower().strip()
        n2 = str(match_rec["business_name"]).lower().strip()
        if n1 and n2:
            name_lev_sims.append(rfuzz.ratio(n1, n2) / 100.0)
            name_ts_sims.append(rfuzz.token_sort_ratio(n1, n2) / 100.0)
            name_jw_sims.append(rf_distance.JaroWinkler.similarity(n1, n2))

        a1 = str(s1_rec["business_address"]).lower().strip()
        a2 = str(match_rec["business_address"]).lower().strip()
        if a1 and a2:
            addr_lev_sims.append(rfuzz.ratio(a1, a2) / 100.0)

        n_analyzed += 1

log(f"  Analyzed {n_analyzed:,} matched pairs")

if name_lev_sims:
    arr = np.array(name_lev_sims)
    log(f"\n  Name Levenshtein Ratio (matched pairs):")
    log(f"    Mean: {arr.mean():.4f}  Median: {np.median(arr):.4f}  Std: {arr.std():.4f}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        pct = (arr < thresh).mean() * 100
        log(f"    <{thresh}: {pct:.1f}%")

    arr2 = np.array(name_ts_sims)
    log(f"\n  Name Token Sort Ratio (matched pairs):")
    log(f"    Mean: {arr2.mean():.4f}  Median: {np.median(arr2):.4f}  Std: {arr2.std():.4f}")

    arr3 = np.array(name_jw_sims)
    log(f"\n  Name Jaro-Winkler (matched pairs):")
    log(f"    Mean: {arr3.mean():.4f}  Median: {np.median(arr3):.4f}  Std: {arr3.std():.4f}")

if addr_lev_sims:
    arr4 = np.array(addr_lev_sims)
    log(f"\n  Address Levenshtein Ratio (matched pairs):")
    log(f"    Mean: {arr4.mean():.4f}  Median: {np.median(arr4):.4f}  Std: {arr4.std():.4f}")
    for thresh in [0.3, 0.5, 0.7, 0.9]:
        pct = (arr4 < thresh).mean() * 100
        log(f"    <{thresh}: {pct:.1f}%")


# ============================================================================
# 9. HARD CASES
# ============================================================================
section("9. HARD CASES (Low-Similarity Matched Pairs)")

hard_cases = []
for _, row in sample_gt.iterrows():
    s1_id = row["source1_entity_id"]
    if s1_id not in s1_lookup:
        continue
    s1_rec = s1_lookup[s1_id]
    for match_id in row["match_list"]:
        if match_id not in s2s3_lookup:
            continue
        mr = s2s3_lookup[match_id]
        n1 = str(s1_rec["business_name"]).lower().strip()
        n2 = str(mr["business_name"]).lower().strip()
        if n1 and n2:
            sim = rfuzz.ratio(n1, n2) / 100.0
            if sim < 0.5:
                hard_cases.append({
                    "s1_id": s1_id, "match_id": match_id,
                    "s1_name": s1_rec["business_name"],
                    "match_name": mr["business_name"],
                    "sim": sim,
                    "s1_addr": str(s1_rec["business_address"])[:80],
                    "match_addr": str(mr["business_address"])[:80],
                    "s1_country": s1_rec["country"],
                    "match_country": mr["country"],
                })

hard_cases.sort(key=lambda x: x["sim"])
log(f"  Found {len(hard_cases):,} hard cases (name Levenshtein < 0.5)")
log(f"\n  Top 15 hardest matched pairs:")
for i, c in enumerate(hard_cases[:15]):
    log(f"\n    [{i+1}] Sim={c['sim']:.3f}  Country: {c['s1_country']}")
    log(f"        S1:    {c['s1_name']}")
    log(f"        Match: {c['match_name']}")
    log(f"        S1 addr:    {c['s1_addr']}")
    log(f"        Match addr: {c['match_addr']}")

# Free lookup memory
del s2s3_lookup, s2_lookup, s3_lookup


# ============================================================================
# 10. COUNTRY-WISE ANALYSIS
# ============================================================================
section("10. COUNTRY-WISE GROUND TRUTH ANALYSIS")

s1_country_map = dict(zip(train_s1["entity_id"], train_s1["country"]))
train_gt["s1_country"] = train_gt["source1_entity_id"].map(s1_country_map)

for country in sorted(train_gt["s1_country"].dropna().unique()):
    if not country:
        continue
    cgt = train_gt[train_gt["s1_country"] == country]
    n_total = len(cgt)
    n_single = (cgt["match_count"] == 0).sum()
    n_matched = n_total - n_single
    avg = cgt[cgt["has_match"]]["match_count"].mean() if n_matched > 0 else 0

    log(f"\n  Country: {country}")
    log(f"    Total S1: {n_total:,}  |  Singletons: {n_single:,} ({100*n_single/n_total:.1f}%)  "
        f"|  With matches: {n_matched:,}  |  Avg matches: {avg:.2f}")
    cd = cgt["match_count"].value_counts().sort_index()
    for mc, freq in cd.head(10).items():
        log(f"    {mc} matches: {freq:>7,}")


# ============================================================================
# 11. TEST SET ANALYSIS
# ============================================================================
section("11. TEST SET ANALYSIS")

log(f"\n  Test set sizes:")
log(f"    Source 1: {len(test_s1):,}")
log(f"    Source 2: {len(test_s2):,}")
log(f"    Source 3: {len(test_s3):,}")
log(f"    S2+S3 candidates: {len(test_s2) + len(test_s3):,}")
log(f"    Brute force comparisons: {len(test_s1) * (len(test_s2) + len(test_s3)):,}")

log(f"\n  Test country distribution:")
for name, df in [("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3)]:
    log(f"    {name}:")
    for country, count in df["country"].value_counts().items():
        log(f"      {country:15s}: {count:>8,}")

all_test_countries = set(test_s1["country"].unique()) | set(test_s2["country"].unique()) | set(test_s3["country"].unique())
all_train_countries = set(train_s1["country"].unique()) | set(train_s2["country"].unique()) | set(train_s3["country"].unique())
unseen = all_test_countries - all_train_countries
if unseen:
    log(f"\n  [IMPORTANT] Unseen countries in test: {unseen}")
    for uc in unseen:
        n1 = (test_s1["country"] == uc).sum()
        n2 = (test_s2["country"] == uc).sum()
        n3 = (test_s3["country"] == uc).sum()
        log(f"    {uc}: S1={n1:,}, S2={n2:,}, S3={n3:,}")
else:
    log(f"\n  No unseen countries in test set.")


# ============================================================================
# 12. ENTITY ID INTEGRITY
# ============================================================================
section("12. ENTITY ID INTEGRITY CHECKS")

gt_s1_ids = set(train_gt["source1_entity_id"])
train_s1_ids = set(train_s1["entity_id"])
log(f"\n  GT S1 IDs not in train_s1: {len(gt_s1_ids - train_s1_ids)}")
log(f"  Train S1 IDs not in GT:   {len(train_s1_ids - gt_s1_ids)}")

train_s2_ids = set(train_s2["entity_id"])
train_s3_ids = set(train_s3["entity_id"])
all_match_set = set(all_match_ids)
missing = all_match_set - train_s2_ids - train_s3_ids
log(f"  Match IDs not in S2/S3:   {len(missing)}")
if missing:
    log(f"    Examples: {list(missing)[:5]}")

matched_s2 = {m for m in all_match_ids if m.startswith("S2-")}
matched_s3 = {m for m in all_match_ids if m.startswith("S3-")}
log(f"\n  S2 records matched: {len(matched_s2):,} / {len(train_s2_ids):,} ({100*len(matched_s2)/max(1,len(train_s2_ids)):.1f}%)")
log(f"  S3 records matched: {len(matched_s3):,} / {len(train_s3_ids):,} ({100*len(matched_s3)/max(1,len(train_s3_ids)):.1f}%)")
log(f"  Unmatched S2: {len(train_s2_ids) - len(matched_s2):,}")
log(f"  Unmatched S3: {len(train_s3_ids) - len(matched_s3):,}")


# ============================================================================
# 13. SCALE ANALYSIS — key numbers for pipeline design
# ============================================================================
section("13. SCALE ANALYSIS (Pipeline Design)")

log(f"\n  === TRAINING ===")
log(f"  S1 entities to match:       {len(train_s1):,}")
log(f"  S2+S3 candidate pool:       {len(train_s2) + len(train_s3):,}")
log(f"  Brute force pairs:          {len(train_s1) * (len(train_s2) + len(train_s3)):,}")
log(f"  Ground truth total matches: {len(all_match_ids):,}")
log(f"  Avg matches per S1 entity:  {len(all_match_ids)/total_s1:.2f}")

log(f"\n  === TEST ===")
log(f"  S1 entities to match:       {len(test_s1):,}")
log(f"  S2+S3 candidate pool:       {len(test_s2) + len(test_s3):,}")
log(f"  Brute force pairs:          {len(test_s1) * (len(test_s2) + len(test_s3)):,}")

log(f"\n  === MEMORY ESTIMATES ===")
total_records = len(train_s1) + len(train_s2) + len(train_s3) + len(test_s1) + len(test_s2) + len(test_s3)
log(f"  Total records across all files: {total_records:,}")
log(f"  Estimated TF-IDF vectors (200k features): ~{total_records * 200000 * 4 / 1e9:.1f} GB (dense) — use sparse!")
log(f"  Estimated embeddings (384-d float32): ~{total_records * 384 * 4 / 1e6:.0f} MB")


# ============================================================================
# 14. VALIDATION SPLIT
# ============================================================================
section("14. CREATING VALIDATION SPLIT")

from sklearn.model_selection import StratifiedKFold

train_gt["match_bucket"] = pd.cut(
    train_gt["match_count"],
    bins=[-1, 0, 1, 2, 5, 100],
    labels=["0", "1", "2", "3-5", "6+"]
).astype(str)

train_gt["strata"] = train_gt["s1_country"].fillna("unknown") + "_" + train_gt["match_bucket"]

# Merge rare strata
strata_counts = train_gt["strata"].value_counts()
rare = strata_counts[strata_counts < 5].index
train_gt.loc[train_gt["strata"].isin(rare), "strata"] = "rare"

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = {}
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_gt, train_gt["strata"])):
    val_ids = train_gt.iloc[val_idx]["source1_entity_id"].tolist()
    folds[fold_idx] = val_ids
    n_val_single = train_gt.iloc[val_idx]["match_count"].eq(0).sum()
    n_val_total = len(val_idx)
    log(f"  Fold {fold_idx}: {len(train_idx):,} train / {n_val_total:,} val  "
        f"(val singletons: {n_val_single:,} = {100*n_val_single/n_val_total:.1f}%)")

primary_val_ids = folds[0]

val_split = {
    "primary_val_s1_ids": primary_val_ids,
    "n_folds": 5,
    "all_folds": {str(k): v for k, v in folds.items()},
}
val_path = PROCESSED_DIR / "val_split.json"
with open(val_path, "w") as f:
    json.dump(val_split, f)
log(f"\n  Saved: {val_path}")
log(f"  Primary validation (fold 0): {len(primary_val_ids):,} S1 entities")


# ============================================================================
# 15. SAVE PROCESSED DATA
# ============================================================================
section("15. SAVING PROCESSED DATA")

# Ground truth parsed
gt_structured = {}
for _, row in train_gt.iterrows():
    gt_structured[row["source1_entity_id"]] = row["match_list"]

gt_path = PROCESSED_DIR / "ground_truth_parsed.json"
with open(gt_path, "w") as f:
    json.dump(gt_structured, f)
log(f"  Saved: {gt_path}")

# Summary JSON
summary = {
    "train_s1_count": int(len(train_s1)),
    "train_s2_count": int(len(train_s2)),
    "train_s3_count": int(len(train_s3)),
    "test_s1_count": int(len(test_s1)),
    "test_s2_count": int(len(test_s2)),
    "test_s3_count": int(len(test_s3)),
    "total_s1_in_gt": int(total_s1),
    "n_singletons": int(n_singletons),
    "singleton_rate": float(singleton_rate),
    "n_with_matches": int(n_with_matches),
    "avg_match_count": float(non_single.mean()) if len(non_single) > 0 else 0,
    "max_match_count": int(non_single.max()) if len(non_single) > 0 else 0,
    "total_matched_ids": len(all_match_ids),
    "train_countries": sorted(all_train_countries),
    "test_countries": sorted(all_test_countries),
    "unseen_countries": sorted(unseen) if unseen else [],
}
summary_path = PROCESSED_DIR / "eda_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
log(f"  Saved: {summary_path}")


# ============================================================================
# 16. PLOTS
# ============================================================================
section("16. GENERATING PLOTS")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    # Plot 1: Match count distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    mc = train_gt["match_count"]
    max_mc = min(int(mc.max()), 15)
    ax.hist(mc.clip(upper=max_mc), bins=range(0, max_mc + 2),
            edgecolor="black", alpha=0.7, color="#2196F3")
    ax.set_xlabel("Number of Matches per S1 Entity", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Match Count Distribution", fontsize=14)
    ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5,
               label=f"Singletons: {n_singletons:,} ({singleton_rate:.1%})")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "01_match_count_distribution.png", dpi=150)
    plt.close(fig)
    log("  Saved: 01_match_count_distribution.png")

    # Plot 2: Country distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    datasets = [
        ("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3),
        ("Test S1", test_s1), ("Test S2", test_s2), ("Test S3", test_s3),
    ]
    for ax, (name, df) in zip(axes.flat, datasets):
        cc = df["country"].value_counts()
        bars = ax.barh(cc.index, cc.values, color="#4CAF50", edgecolor="black", alpha=0.8)
        ax.set_title(f"{name} ({len(df):,} total)", fontsize=11)
        for bar, v in zip(bars, cc.values):
            ax.text(v + max(cc.values)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:,}", va="center", fontsize=9)
    plt.suptitle("Country Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "02_country_distribution.png", dpi=150)
    plt.close(fig)
    log("  Saved: 02_country_distribution.png")

    # Plot 3: Name similarity distributions
    if name_lev_sims:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (data, title, color) in zip(axes, [
            (name_lev_sims, "Levenshtein Ratio", "#FF9800"),
            (name_ts_sims, "Token Sort Ratio", "#9C27B0"),
            (name_jw_sims, "Jaro-Winkler", "#009688"),
        ]):
            d = np.array(data)
            ax.hist(d, bins=50, edgecolor="black", alpha=0.7, color=color)
            ax.set_title(f"Name {title}\n(Matched Pairs)", fontsize=11)
            ax.set_xlabel("Similarity")
            ax.axvline(x=d.mean(), color="red", linestyle="--",
                       label=f"Mean: {d.mean():.3f}")
            ax.legend()
        plt.tight_layout()
        fig.savefig(EDA_PLOTS_DIR / "03_name_similarity.png", dpi=150)
        plt.close(fig)
        log("  Saved: 03_name_similarity.png")

    # Plot 4: Address similarity
    if addr_lev_sims:
        fig, ax = plt.subplots(figsize=(10, 5))
        d = np.array(addr_lev_sims)
        ax.hist(d, bins=50, edgecolor="black", alpha=0.7, color="#E91E63")
        ax.set_title("Address Levenshtein Ratio (Matched Pairs)", fontsize=14)
        ax.set_xlabel("Similarity")
        ax.axvline(x=d.mean(), color="red", linestyle="--",
                   label=f"Mean: {d.mean():.3f}")
        ax.legend(fontsize=11)
        plt.tight_layout()
        fig.savefig(EDA_PLOTS_DIR / "04_address_similarity.png", dpi=150)
        plt.close(fig)
        log("  Saved: 04_address_similarity.png")

    # Plot 5: Country-wise singleton rate
    cs = train_gt.groupby("s1_country").agg(
        total=("has_match", "count"),
        singletons=("has_match", lambda x: (~x).sum()),
    )
    cs["rate"] = cs["singletons"] / cs["total"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(cs.index, cs["rate"], color="#FF5722", edgecolor="black", alpha=0.8)
    ax.set_title("Singleton Rate by Country", fontsize=14)
    ax.set_ylabel("Singleton Rate")
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, cs["rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.2%}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "05_singleton_rate.png", dpi=150)
    plt.close(fig)
    log("  Saved: 05_singleton_rate.png")

    # Plot 6: Name length distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (sn, df) in zip(axes, [("S1", train_s1), ("S2", train_s2), ("S3", train_s3)]):
        lens = df["business_name"].str.len()
        ax.hist(lens, bins=60, edgecolor="black", alpha=0.7, color="#3F51B5")
        ax.set_title(f"Train {sn} Name Length", fontsize=12)
        ax.set_xlabel("Characters")
        ax.axvline(x=lens.mean(), color="red", linestyle="--",
                   label=f"Mean: {lens.mean():.0f}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "06_name_length.png", dpi=150)
    plt.close(fig)
    log("  Saved: 06_name_length.png")

    log("  All plots saved to output/eda_plots/")

except Exception as e:
    log(f"  [WARNING] Plotting error: {e}")


# ============================================================================
# DONE
# ============================================================================
section("EDA COMPLETE")

total_time = time.time() - T0
log(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
log(f"  Report: output/eda_report.txt")
log(f"  Plots:  output/eda_plots/")
log(f"  Data:   output/processed/")

report_path = OUTPUT_DIR / "eda_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"\n{'='*80}")
print(f"  DONE — Full report saved to {report_path}")
print(f"{'='*80}")
