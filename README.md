# Explainable and Accountable Credit Risk Prediction

COSC 4V88 (Ethics in AI), Brock University, Project 5.
Shuo Gao (Part A), Kane Zhu (Part B), Ayaan Alam Biabani Feroz (Part C).

We train two credit-default models on the UCI Default of Credit Card Clients dataset, explain their predictions with SHAP and LIME, measure how faithful and how stable those explanations are, and test a reliability flag that sends questionable explanations to a human reviewer.

All notebooks are saved **with their outputs**, so every result can be read without re-running anything.

## Repository layout

```
01_Data/processed/      cleaned train / validation / test split (seed 42)
02_Code/                notebooks and scripts, in run order below
03_Results/
  config.json           split, features, model settings, decision thresholds
  cleaning_log.json     every change made to the raw data
  models/               trained Logistic Regression and XGBoost (joblib)
  tables/               result tables for each part
  figures/              figures used in the report and slides
```

## Run order

| Step | File | Owner | Produces |
|---|---|---|---|
| 1 | `Part_A_data_models.ipynb` | Part A | Cleaned split, two models, test metrics, proxy check (`PartA_Step1_tables`) |
| 2 | `Part_A_step2_fidelity_fairness.ipynb` | Part A | Explanation fidelity per client, fairness audit by group (`PartA_Step2_tables`) |
| 3 | `Part_B_explanations.ipynb` | Part B | LIME stability, sensitivity, SHAP–LIME agreement, demo cases (`PartB_tables`) |
| 4 | `Part_C_reliability_flag.ipynb` (uses `part_c_analysis.py`) | Part C | Reliability flag, evaluation, ablation, baselines (`PartC_tables`) |

`Part_C_quick_check.py --output-dir 03_Results/tables/PartC_tables` re-checks the saved Part C results.

Steps 2 and 3 use the same 500 explained clients (200 validation + 300 test, listed in `explain_sample.csv`) and the same 150-client SHAP background.

**Part C note:** the saved Part C results come from a pilot on an earlier set of 100 explained test clients (split 50/50 into calibration and evaluation), so `part_c_analysis.py` expects those earlier input files and will not run against the 500-client tables in this repository. An update of Part C to the 500-client sample is in progress.

## How to run

- **Step 1 (Colab or local):** reads the dataset zip if present, otherwise downloads it from UCI, and writes `part_a_outputs.zip`.
- **Step 2 (Colab or local):** upload `part_a_outputs.zip` from step 1 next to the notebook, then run all; writes `part_a_step2_outputs.zip`.
- **Steps 3 and 4:** run from inside `02_Code` with this folder structure intact; they read `../01_Data` and `../03_Results`.

```
pip install -r requirements.txt
```

The models were saved with scikit-learn 1.6.1. Loading them with another version gives a warning and may change results.

## Fixed settings

- Random seed 42 for the split, models, explanation samples and bootstrap (Part C uses seed 88 only for its record-keeping)
- Stratified 60/20/20 split: 18,000 / 6,000 / 6,000 clients
- 19 model inputs; sex, education, marital status and age are kept for auditing only, never as inputs
- Decision thresholds chosen on validation: 0.25 (Logistic Regression), 0.26 (XGBoost)

## Demo

The demo uses saved outputs, not a live run:

- `03_Results/figures/shap_demo_success_xgb.png`: client 17775, a defaulter correctly flagged by XGBoost, with a stable explanation that SHAP and LIME agree on
- `03_Results/figures/shap_demo_failure_xgb.png`: client 6391, whose LIME explanation changes from run to run
- `03_Results/tables/PartB_tables/demo_cases.json` lists the demo clients and their scores

## Data

Yeh, I.-C. (2016). *Default of credit card clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H. Licensed under CC BY 4.0.

This is a course research prototype. It is not suitable for making decisions about real people.
