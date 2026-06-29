# Single-Bidder Prediction in Public Procurement

Machine learning project that predicts whether a public tender will attract
only a **single bidder** — a recognized indicator of low competition and
potential procurement risk.

Developed during an internship at **TOO "АБМ компании"** as part of the
Big Data Analysis / Mathematical and Computational Science program.

---

## Problem

In public procurement, tenders that receive only one offer often signal
weak competition or restrictive conditions. Being able to predict such
tenders in advance helps suppliers prioritize opportunities and helps
analysts flag procurement risks.

**Task type:** binary classification
**Target:** `single_bidder = 1` if `Offers_Number == 1`, else `0`

---

## Dataset

[Public Tenders Romania 2007–2016](https://www.kaggle.com/datasets/gpreda/public-tenders-romania-20072016)
(`contracts.csv`, ~887k rows, 39 columns).

A stratified sample of 150,000 rows was used for efficient experimentation.

---

## Approach

1. **Data cleaning** — removed columns with >60% missing values.
2. **Leakage prevention** — removed all features known only *after* the
   award (winner info, final value, award dates). This is critical: using
   them would leak the answer into the model.
3. **Feature encoding**
   - Frequency encoding for high-cardinality columns
     (`Contracting_Authority`, `CPV_Code`)
   - Label encoding for the remaining categorical features
4. **Modeling** — Logistic Regression (baseline) vs XGBoost (main model),
   with class-imbalance handling (`scale_pos_weight` ≈ 4).
5. **Evaluation** — ROC-AUC, PR-AUC, F1 (suited for imbalanced data).
6. **Interpretation** — feature importance + SHAP values.

---

## Results

| Model               | ROC-AUC | PR-AUC | F1 (single-bidder) |
|---------------------|:-------:|:------:|:------------------:|
| Logistic Regression |  0.747  | 0.500  |       0.467        |
| **XGBoost**         | **0.850** | **0.638** |     **0.557**      |

**Key finding:** `Procedure_Type` is by far the strongest predictor
(~48% of model importance). Non-competitive procedures such as
*"Negociere accelerata"* show a single-bidder rate of ~79%, while open
tenders stay around 15%. The model independently confirmed the pattern
found during exploratory analysis.

---

## Repository structure

```
.
├── data/
│   └── tenders_clean.csv          # cleaned sample (or link to Kaggle)
├── figures/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   ├── feature_importance.png
│   ├── shap_bar.png
│   └── shap_summary.png
├── notebooks/
│   ├── week1-2.py                # loading, target, cleaning, EDA
│   ├── week2-3.py              # encoding, models, metrics
│   └── SHAP_analysis.py               # SHAP interpretation
├── single_bidder_model.pkl        # trained model
└── README.md
```

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib

python week1-2.py      # prepares tenders_clean.csv
python week2-3.py      # trains models, saves figures
python SHAP_analysis.py    # SHAP analysis, saves model
```

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib · seaborn

---

## Possible improvements

- Convert `Participation_Estimated_Value` to a single currency and add it
  as a feature.
- Extract temporal features (year, month) from announcement dates.
- Hyperparameter tuning via `RandomizedSearchCV`.
- Deploy as a small REST API for real-time scoring.