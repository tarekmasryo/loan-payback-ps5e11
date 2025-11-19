# 💰 Loan Payback — EDA & Stacked Ensemble (Kaggle PS S5E11)

Compact, production-style notebook for the **Kaggle Playground Series S5E11** loan payback task.

The notebook covers:

- Robust **EDA for binary classification** (data health, drift, signal)
- **Unified feature ranking** (AUC, KS, correlation, MI, IV)
- **Leak-safe encodings** (K-fold target encoding, numeric binning, frequency)
- **Baseline model** (5-fold Logistic Regression)
- **Advanced models** (XGBoost + LightGBM + stacked meta layer)
- Ready-to-use **submissions & JSON/CSV artifacts**

---

## 🔍 What this notebook does

- Finds the dataset in `/kaggle/input/playground-series-s5e11`
- Auto-detects:
  - Target column (loan paid back / not)
  - ID column
  - Numeric vs categorical features
- Runs a **practical EDA**:
  - Missingness, outliers, train–test drift
  - Univariate signal and feature stability
- Builds a **feature matrix** for tree models:
  - Winsorization, log/ratio transforms
  - Target encoding (single & pairwise)
  - Numeric bin TE + frequency features
- Trains:
  - Several tuned **XGBoost** configs (with optional GPU)
  - One **LightGBM** baseline
  - **Meta models** (Logistic Regression + shallow XGBoost)
- Saves multiple **submission files** and a diagnostics JSON.

---

## 📦 Key Outputs

Under `artifacts/` and the working directory you get:

- `artifacts/top_features.csv` — unified feature ranking  
- `artifacts/drift_numeric.csv`, `artifacts/drift_categorical.csv`  
- `artifacts/univariate_scores.csv`  
- `artifacts/schema_diff.csv`, `artifacts/unseen_categories.csv`  
- `artifacts/eda_summary.json` — EDA + baseline summary  

Ensemble & submissions:

- `diag_ensemble.json` — model/ensemble diagnostics  
- `submission_single_best.csv`  
- `submission_stacked_lr.csv`  
- `submission_stacked_xgb.csv`  
- `submission_stacked_blend.csv`  
- `submission.csv` (final chosen submission)

These files are meant to be reused in other scripts, dashboards, or monitoring jobs.

---

## ⚙️ Tech Stack

- Python 3.10–3.12  
- NumPy, pandas, matplotlib, seaborn, SciPy  
- scikit-learn  
- XGBoost (GPU support if available)  
- LightGBM (GPU optional)  
- Jupyter / Kaggle Notebook

---

## 🚀 How to run

Locally:

```bash
git clone https://github.com/tarekmasryo/loan-payback-ps5e11.git
cd <loan-payback-ps5e11>

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
