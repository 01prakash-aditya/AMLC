# Business Entity Resolution Pipeline

This repository contains the end-to-end Machine Learning pipeline for the Amazon ML Challenge (Business Entity Resolution). 

The pipeline scales efficiently to tens of millions of records by utilizing sparse matrix blocking (TF-IDF + `sparse_dot_topn`) and vectorized string similarity extraction (`rapidfuzz`), culminating in a finely-tuned LightGBM classifier.

## Directory Structure & Setup

Before running the pipeline, ensure the raw `dataset/` directory is placed correctly relative to this code, and the `output/` directory exists.

The required workspace structure is:
```text
<project_root>/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── output/
│   ├── processed/          # Intermediary parquet and feature files are saved here
│   ├── models/             # Trained LightGBM model is saved here
│   ├── matching_results.tsv  # Final Output
│   └── candidate_pairs.tsv   # Blocking Output
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            ├── stage01_eda.py
            ├── stage02_preprocessing.py
            ├── stage03_blocking.py
            ├── stage04_feature_engineering.py
            ├── stage05_model_training.py
            └── stage06_inference.py
```

## System Requirements
- **Memory**: The blocking phase (`stage03`) and feature engineering phase (`stage04`) process ~12.5 million rows and require **16GB to 32GB of RAM** depending on the concurrency. 
- **Python**: 3.10+ recommended.

## 1. Install Dependencies
Install the required packages in your environment:
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

## 2. Running the Pipeline (End-to-End)

The pipeline is split into independent stages to maximize memory efficiency. **Run the stages sequentially from the project root directory.**

### Stage 2: Text Preprocessing
Reads the raw TSVs, performs multi-lingual text normalization (stripping accents, standardizing abbreviations, removing punctuation), and saves them as compressed `.parquet` files for fast loading.
```bash
python code/business_entity_resolution/src/stage02_preprocessing.py
```

### Stage 3: Dense Semantic Blocking
Uses `sentence-transformers` (paraphrase-multilingual-MiniLM) to encode names and addresses into dense semantic vectors, then uses GPU-accelerated **FAISS** to instantly retrieve the Top-15 most semantically similar matches. Handles multi-lingual abbreviations natively.
```bash
python code/business_entity_resolution/src/stage03_blocking.py
```
*Note: This generates `train_candidate_pairs.tsv` and `test_candidate_pairs.tsv`.*

### Stage 4: Cross-Encoder Data Prep
Mines the FAISS output against the ground truth to build a high-quality dataset of positive matches and "hard negatives".
```bash
python code/business_entity_resolution/src/stage04_cross_encoder_prep.py
```

### Stage 5: Cross-Encoder Fine-Tuning
Fine-tunes a Deep Learning `xlm-roberta-base` Cross-Encoder model on the hard-negative dataset using Binary Cross-Entropy loss. This replaces manually engineered string-features with deep contextual attention.
```bash
python code/business_entity_resolution/src/stage05_cross_encoder_train.py
```

### Stage 6: Cross-Encoder Inference & Post-Processing
Scores the test FAISS candidates through the fine-tuned Transformer. Applies a Dual-Threshold Singleton Guard to maximize the F0.5 precision score before outputting final matches.
```bash
python code/business_entity_resolution/src/stage06_graph_inference.py
```

*Note: The official challenge `matching_results.tsv` output is generated and placed directly in the `output/` folder after this step.*

## Optional: Exploratory Data Analysis
If you wish to view data distributions, dataset health, missing values, and ground-truth patterns, you can run the EDA script:
```bash
python code/business_entity_resolution/src/stage01_eda.py
```
*(Outputs reports and matplotlib plots to `output/eda_plots/`)*
