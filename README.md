# 💳 Credit Card Default Prediction (UCI) — End-to-End ML + Streamlit + Docker

ML project for predicting **credit card default** using the **UCI “Default of Credit Card Clients” dataset** with a fully reproducible pipeline:

EDA → Feature Engineering → statistical validation → feature selection → Optuna tuning (**LightGBM**) → **full custom end-to-end pipeline (raw → processed → model)** → Streamlit app → Docker.

---

## ✨ Highlights
- ✅ Dataset understanding: identify which feature groups drive default risk (**repayment status**, bills, payments, credit limit, demographics)
- ✅ Feature engineering built for real-world inference:
  - `LIMIT_BAL_LOG = log1p(LIMIT_BAL)`
  - `PAY_AMT* = log1p(PAY_AMT*)`
  - `BILL_AMT* = PowerTransformer(Yeo–Johnson)` (supports negative values)
  - `AGE_BIN` (age binning)
  - `PAY_*` stabilization via **clipping** (`upper=3`) to reduce outlier noise while keeping ordinal meaning
- ✅ Statistical validation to confirm class-separating signals (p-values)
- ✅ Feature selection:
  - greedy experiments + L1 regularization + permutation importance
- ✅ Optuna tuning + saving best params and **thresholds optimized for Recall and F1**
- ✅ Pipeline parity check:
  - `custom_full_pipeline(raw)` ≈ `final_pipe(processed)` (same behavior from raw inputs)
- ✅ Deployment:
  - Streamlit app + Docker + Docker Compose

---

## 📌 Contents
- [Project overview](#-project-overview)
- [Repository structure](#-repository-structure)
- [Results](#-results)
- [How it works (step-by-step)](#-how-it-works-step-by-step)
- [Run locally](#️-run-locally)
- [Run with Docker](#-run-with-docker)
- [Artifacts](#-artifacts)
- [Roadmap / Future work](#-roadmap--future-work)
- [License](#-license)

---

## 📖 Project overview

**Goal:** build a reproducible end-to-end ML pipeline to predict `default (0/1)`:
raw data → preprocessing/feature engineering → training → artifact saving → inference → UI.

**Dataset:** UCI “Default of Credit Card Clients” (Taiwan, 2005)

**What to look for in this dataset**
- The features naturally split into:
  - **Repayment status (`PAY_*`)** — typically the strongest predictive signal
  - **Bill statements (`BILL_AMT*`)** — may contain negative values (refunds/adjustments)
  - **Payments (`PAY_AMT*`)** — heavy right-skew (many small, few huge)
  - **Credit limit (`LIMIT_BAL`)** + demographics (`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`)
- The main practical challenge is balancing:
  - catching defaulters (**Recall**) vs. avoiding too many false alarms (**Precision/F1**)
- That’s why this project stores **multiple thresholds** (Recall-optimized and F1-optimized) instead of using a fixed 0.5 cutoff.

**Modeling approach**
- Baseline model(s) to validate feature signal and stability
- Final model: **LightGBM (LGBMClassifier)** — strong on tabular data and non-linear interactions

**Core metric during development:** ROC-AUC (stable for imbalanced data)  
**Final decision metrics:** Accuracy / Precision / Recall / F1  
**Operating point:** thresholds chosen from OOF predictions (Recall-optimized + F1-optimized)

---

## 🧱 Repository structure
```text
.
├── artifacts/
│   ├── model_data/
│   │   ├── models/
│   │   │   ├── full_custom_final_model.joblib
│   │   │   └── full_precustom_final_model.joblib
│   │   ├── best_params.json
│   │   ├── threshold_performance.json
│   │   └── thresholds.json
│   └── final_features.json
├── dataset/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv
│   └── split/
│       ├── raw/
│       │   ├── train_set.csv
│       │   └── test_set.csv
│       └── preprocessed/
│           ├── train_set.csv
│           └── test_set.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_optuna.ipynb
├── src/
│   ├── feat_engineering.py
│   └── predict.py
├── app.py
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements_dev.txt
└── README.md
```

---

## 📊 Results

> Results are computed using a stratified train/test split and validated via cross-validation with out-of-fold predictions for threshold selection.

### Models saved
- **`full_precustom_final_model.joblib`**  
  Trained on already *preprocessed* data (expects processed features at inference)

- **`full_custom_final_model.joblib`**  
  Full end-to-end pipeline: *raw → preprocessing → model* (recommended for deployment)

### Thresholding
Two operating thresholds are stored:
- **Recall-optimized threshold** (default in inference)
- **F1-optimized threshold**

Why two?
- **Recall threshold** is useful when missing a defaulter is costly
- **F1 threshold** is a balanced option when you want fewer false positives

See:
- `artifacts/model_data/thresholds.json`
- `artifacts/model_data/threshold_performance.json`

---

## 🧠 How it works (step-by-step)

### 1) EDA + Feature Engineering + Feature Selection (`notebooks/01_eda.ipynb`)

First, EDA to understand:
- class balance (`default=1` share)
- distribution shifts between default vs non-default
- which groups (`PAY_*`, `BILL_AMT*`, `PAY_AMT*`) carry the most signal
- whether separation is statistically meaningful

Then feature engineering:

**Core transforms**
- `AGE_BIN` — binned age
- `LIMIT_BAL_LOG = log1p(LIMIT_BAL)` — reduce skew & stabilize scale
- `PAY_*` — clipped to `upper=3`
  - keeps ordinal information but reduces extreme/noisy values
- `BILL_AMT*` — `PowerTransformer(method="yeo-johnson")`
  - handles negative values and improves distribution shape
- `PAY_AMT*` — `log1p(PAY_AMT*)`
  - addresses heavy right-skew

**Target relationship checks**
- Numeric: correlation matrix (quick signal map)
- Categorical: Cramér’s V (association strength)

**Feature usefulness**
Combined three approaches:
- greedy selection (experiment-driven)
- L1 regularization (sparse selection)
- permutation importance (final sanity check of contribution)

**Outputs**
- raw split → `dataset/split/raw/`
- preprocessed split → `dataset/split/preprocessed/`
- feature metadata → `artifacts/final_features.json`

---

### 2) Modeling + Optuna (LightGBM) (`notebooks/02_optuna.ipynb`)

- Tested multiple models with default params
- Selected **LGBMClassifier**
- Ran sanity checks for suspicious overlearning/leakage
- Optuna tuning → best score around **0.786**
- Saved:
  - best hyperparameters → `artifacts/model_data/best_params.json`

**Threshold selection (OOF-based)**
- Generated out-of-fold probabilities
- Picked:
  - recall-optimized threshold
  - f1-optimized threshold
- Saved:
  - `artifacts/model_data/thresholds.json`
  - `artifacts/model_data/threshold_performance.json`

---

### 3) Full Custom Pipeline (raw → processed → model)

A key difference from many “notebook-only” projects:
- The repo contains a pipeline that can accept **raw user input** and reproduce the same transformations used for training.

Implemented in:
- `src/feat_engineering.py` — preprocessing logic + pipeline builder

Validated by comparing:
- `custom_full_pipeline(raw)` vs `final_pipe(processed)`  
so deployment behavior matches training behavior.

---

### 4) Inference code (`src/predict.py`)

`predict.py`:
- loads the model and thresholds from `artifacts/`
- takes a raw input dict
- returns probability + prediction using the selected threshold

By default it uses the **recall threshold** (more conservative for catching defaulters).

---

### 5) Streamlit app (`app.py`)

Streamlit UI:
- user inputs raw values
- app calls `predict()`
- app displays probability + decision based on threshold

---

## ▶️ Run locally

### Install dependencies
```bash
pip install -r requirements.txt
# dev dependencies (optional)
pip install -r requirements_dev.txt
```

### Run Streamlit
```bash
streamlit run app.py
```

Open:
- http://localhost:8501

---

## 🐳 Run with Docker

### Build
```bash
docker build -t credit-default-streamlit .
```

### Run
```bash
docker run --rm -p 8501:8501 credit-default-streamlit
```

### Or Docker Compose
```bash
docker compose up --build
```

Open:
- http://localhost:8501

---

## 📦 Artifacts

Stored in `artifacts/model_data/`:
- `best_params.json` — best Optuna hyperparameters for LightGBM
- `thresholds.json` — thresholds optimized for Recall and F1
- `threshold_performance.json` — metrics summary for each threshold
- `models/full_custom_final_model.joblib` — full pipeline (raw → processed → model)
- `models/full_precustom_final_model.joblib` — model expecting preprocessed inputs

Also:
- `artifacts/final_features.json` — final selected feature list + metadata

---

## 🛣 Roadmap / Future work

This project currently **does not** apply explicit class balancing techniques (e.g., **SMOTE**, random oversampling/undersampling).  
The model is trained on the original class distribution, and performance control is handled mainly through:
- feature engineering,
- cross-validation + OOF predictions,
- and **threshold tuning** (Recall-optimized / F1-optimized).

### Planned improvements
- **Class imbalance handling**
  - Try **SMOTE** and compare against simpler baselines (random oversampling / undersampling).
  - Evaluate whether balancing improves **Recall/F1** without increasing false positives too much.
  - Apply balancing **inside CV folds only** to avoid leakage.
- **Cost-sensitive learning**
  - Try `class_weight` / `scale_pos_weight` (LightGBM) as a lightweight alternative to SMOTE.
  - Tune these weights with Optuna and compare trade-offs.
- **Better thresholding strategies**
  - Optimize thresholds for business constraints (e.g., minimum Recall, maximum false positive rate).
  - Consider probability calibration (Platt scaling / isotonic) before thresholding.

---

## 📄 License
MIT — see `LICENSE`
