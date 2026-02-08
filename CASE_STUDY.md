# 🧠 Case Study — Loan Payback (Kaggle PS S5E11)

## Problem
Predict whether a loan will be **paid back** (binary classification) from tabular borrower and loan attributes.

- **Goal:** train a robust model that generalizes well to the hidden test set.
- **Output:** a probability score per row, exported as Kaggle submission files.

---

## Data
- **Source:** Kaggle Playground Series **Season 5, Episode 11**
- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`
- **Grain:** one row per loan applicant / loan record
- **Target:** auto-detected in the notebook (binary)

Practical checks included:
- Missingness and type consistency
- Train ↔ test drift (numeric + categorical)
- Unseen categories and schema differences

---

## Approach
- **EDA (decision-oriented):**
  - data health, outliers, stability, drift
  - univariate signal and feature ranking
- **Feature engineering (leak-safe):**
  - K-fold target encoding (single and selected pairwise)
  - numeric bin encodings and frequency features
  - winsorization + simple transforms for tree models
- **Models:**
  - baseline Logistic Regression (stratified CV)
  - XGBoost + LightGBM
  - stacked meta layer (LogReg / shallow XGB) + blend

---

## Outputs
Artifacts are saved under `artifacts/`:
- feature ranking tables
- drift reports
- EDA summary JSON
- multiple submission variants + ensemble diagnostics JSON

---

## Decisions & Takeaways
- Drift and unseen categories can dominate leaderboard surprises — monitor them early.
- Leak-safe encodings (K-fold TE) provide strong uplift without contaminating validation.
- Stacking/blending is most useful after stabilizing single-model performance and calibration.

---

## Next Steps
- Add monotonic constraints where domain rules apply (if available).
- Expand drift analysis with time-based splits if temporal leakage is plausible.
- Convert the artifacts into a small monitoring dashboard (feature drift + score stability).
