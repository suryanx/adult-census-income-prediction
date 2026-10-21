# Adult Income Classification — ML Hackathon Mock

## Overview

This project predicts whether an individual's annual income exceeds $50,000 using demographic and employment-related features from the UCI Adult dataset.

**Task:** Binary classification  
**Selected model:** LightGBM  
**Positive class:** `>50K`  
**Decision threshold:** `0.44`

## Dataset

The dataset contains features such as age, education, occupation, marital status, hours worked per week, and capital gains or losses.

- `data/raw/adult_train.csv` — labeled training dataset
- `data/raw/adult_test.csv` — unlabeled dataset used to generate predictions

The model learns patterns from labeled examples and predicts income categories for unseen records. These predictions are statistical estimates, not guarantees about an individual's actual income.

## Workflow

1. Audit the data and inspect missing values and class balance.
2. Establish a baseline model.
3. Compare linear models, decision trees, random forests, and boosting models.
4. Tune model hyperparameters and classification thresholds.
5. Compare models using cross-validation.
6. Evaluate the selected model on an untouched holdout partition.
7. Train the final pipeline on all labeled training rows.
8. Generate and verify predictions for the unlabeled test dataset.

## Model selection

LightGBM was selected based on the experiments conducted in this project. An equal-weight ensemble was also tested, but did not outperform LightGBM on the validation comparison.

## Holdout results

The recorded final holdout metrics at a threshold of `0.44` were:

| Metric | Score |
|---|---:|
| Accuracy | 0.8710 |
| Precision | 0.7438 |
| Recall | 0.7083 |
| F1 | 0.7256 |

These results describe the holdout evaluation, not the unlabeled test dataset.

## Project structure

```text
ml-hackathon-mock/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── final_lightgbm_pipeline.joblib
│   └── model_config.json
├── src/
│   ├── data_audit.py
│   ├── train_models.py
│   ├── first_prediction.py
│   ├── tree_models.py
│   ├── boosting_models.py
│   ├── tune_models.py
│   ├── threshold_tuning.py
│   ├── cross_validation.py
│   ├── compare_model_cv.py
│   ├── ensemble.py
│   ├── final_evaluation.py
│   ├── finalize_model.py
│   ├── generate_submission.py
│   └── verify_submission.py
└── submissions/
    └── adult_test_predictions.csv
```

## Setup

Create and activate a virtual environment, then install the dependencies used by the scripts:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn lightgbm xgboost catboost joblib
```

## Reproduce the final pipeline

Run these commands from the project root:

```bash
python src/final_evaluation.py
python src/finalize_model.py
python src/generate_submission.py
python src/verify_submission.py
```

**Note:** The holdout evaluation should be treated as a final check, not a repeated tuning target. For strict reproducibility, record the dependency versions and use the same dataset and split settings.

## Limitations

- The dataset reflects historical US census data and may not represent current populations or other countries.
- Predictions can reflect biases present in the source data.
- The model estimates statistical associations and should not be used as the sole basis for consequential decisions about individuals.
- The generated CSV is a practice prediction artifact. An actual competition may require a different submission schema.

## Purpose

This project is a practice environment for learning data auditing, preprocessing, model selection, validation, threshold tuning, final evaluation, and prediction generation in a timed ML competition.
