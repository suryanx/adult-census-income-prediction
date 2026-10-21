
import time
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. Load and clean data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# 2. Use only the original training partition.
# Validation and final holdout remain untouched.
X_train, _, y_train, _ = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 3. Preprocessing for numeric/categorical features
numeric_cols = X_train.select_dtypes(include="number").columns
categorical_cols = X_train.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ]), categorical_cols),
])

# 4. Define candidate models
models = {
    "LightGBM": Pipeline([
        ("preprocessing", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=10,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )),
    ]),
    "XGBoost": Pipeline([
        ("preprocessing", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )),
    ]),
    "Random Forest": Pipeline([
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=4,
        )),
    ]),
    "CatBoost": CatBoostClassifier(
        iterations=500,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=42,
        thread_count=4,
        allow_writing_files=False,
        cat_features=list(categorical_cols),
    ),
}

# CatBoost receives raw categorical features; its missing values
# are filled explicitly. Other models use the shared pipeline.
X_cat = X_train.copy()
for col in categorical_cols:
    X_cat[col] = X_cat[col].fillna("__MISSING__").astype(str)


# 5. Identical folds and metrics for every model
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}

rows = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    start = time.time()

    fold_scores = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train), start=1
    ):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        if name == "CatBoost":
            # CatBoost uses raw categorical columns.
            X_fold_train_cat = X_cat.iloc[train_idx]
            X_fold_val_cat = X_cat.iloc[val_idx]

            fold_model = CatBoostClassifier(
                iterations=500,
                depth=7,
                learning_rate=0.05,
                loss_function="Logloss",
                verbose=False,
                random_seed=42,
                thread_count=4,
                allow_writing_files=False,
                cat_features=list(categorical_cols),
            )

            fold_model.fit(
                X_fold_train_cat,
                y_fold_train,
            )
            pred = fold_model.predict(
                X_fold_val_cat
            ).reshape(-1).astype(int)

        else:
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            pred = fold_model.predict(X_fold_val).astype(int)

        fold_scores["accuracy"].append(
            accuracy_score(y_fold_val, pred)
        )
        fold_scores["precision"].append(
            precision_score(y_fold_val, pred, zero_division=0)
        )
        fold_scores["recall"].append(
            recall_score(y_fold_val, pred, zero_division=0)
        )
        fold_scores["f1"].append(
            f1_score(y_fold_val, pred, zero_division=0)
        )

        print(
            f"  Fold {fold}: "
            f"F1={fold_scores['f1'][-1]:.4f}"
        )

    elapsed = time.time() - start

    rows.append({
        "Model": name,
        "Accuracy_mean": sum(fold_scores["accuracy"]) / 5,
        "Precision_mean": sum(fold_scores["precision"]) / 5,
        "Recall_mean": sum(fold_scores["recall"]) / 5,
        "F1_mean": sum(fold_scores["f1"]) / 5,
        "F1_std": pd.Series(fold_scores["f1"]).std(ddof=0),
        "Seconds": elapsed,
    })

results_df = pd.DataFrame(rows).sort_values(
    "F1_mean", ascending=False
)

print("\nCROSS-VALIDATION MODEL COMPARISON")
print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

results_df.to_csv(
    "data/processed/model_cv_comparison.csv",
    index=False,
)
print("\nSaved: data/processed/model_cv_comparison.csv")
