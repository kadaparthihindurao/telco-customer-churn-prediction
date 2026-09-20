# Telco Customer Churn Prediction

An end-to-end machine-learning project that predicts customer churn using the public [Telco Customer Churn dataset by blastchar on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Headline results

- Analyzed 7,043 customer records; converted `TotalCharges` to numeric and removed 11 blank records, leaving 7,032 rows.
- Overall churn was 26.58%.
- Month-to-month customers churned at 42.71%, versus 11.28% for one-year and 2.85% for two-year contracts.
- Churn decreased from 47.68% for customers in months 0–12 to 9.51% for customers in months 49–72.
- Logistic Regression achieved the best holdout performance: **80.38% accuracy, 64.85% churn precision, 57.22% churn recall, 60.80% churn F1, and 0.836 ROC-AUC**.
- Random Forest achieved 78.89% accuracy and 0.826 ROC-AUC.

## Model comparison

| Model | Accuracy | Churn precision | Churn recall | Churn F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **80.38%** | **64.85%** | **57.22%** | **60.80%** | **0.836** |
| Random Forest | 78.89% | 63.14% | 49.47% | 55.47% | 0.826 |

The comparison uses a fixed random seed, stratified 80/20 train/test split, and preprocessing inside scikit-learn pipelines to prevent leakage. Accuracy is reported with class-specific metrics because the churn class is imbalanced.

## Main drivers

Random Forest feature importance identified `TotalCharges`, `tenure`, `MonthlyCharges`, and month-to-month contract status as the leading signals. Online security, tech support, fiber-optic service, and electronic-check payment also ranked highly. These are predictive associations, not causal effects.

## Repository contents

- `Telco_Customer_Churn_Analysis.ipynb` — Colab-ready notebook with cleaning, EDA, preprocessing, modeling, evaluation, charts, and conclusions.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — source CSV downloaded from the Kaggle dataset archive.
- `analyze_churn.py` — reproducible local validation script.
- `results.json` — exact metrics produced by the validation run.
- `linkedin_post.md` — ready-to-edit LinkedIn project post.

## Run in Google Colab

1. Download the notebook and CSV from this repository.
2. Open [Google Colab](https://colab.research.google.com/), upload the notebook, and choose **Runtime → Run all**.
3. Upload the CSV when prompted. Colab already includes pandas, seaborn, matplotlib, and scikit-learn.

## Responsible interpretation

This analysis uses observational historical data. It does not establish that a feature causes churn, and model quality should be revalidated on current customer data before any real deployment. A production retention workflow should tune the probability threshold to campaign capacity and the relative costs of false positives and missed churners.

## Data source

`blastchar`, “Telco Customer Churn,” Kaggle. Accessed September 20, 2026. Review the dataset page for its current license and usage terms before redistributing the CSV.
