
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from lightgbm import LGBMClassifier

# 1. Load and clean data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# 2. Reproduce the existing training/validation/holdout split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# 3. Preprocessing: fit imputers and encoder on training data only
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

X_train_encoded = preprocessor.fit_transform(X_train)
X_val_encoded = preprocessor.transform(X_val)

# 4. Controlled LightGBM experiments
configs = [
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "reg_lambda": 0.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "reg_lambda": 0.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 20, "max_depth": -1, "reg_lambda": 1.0},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 40, "max_depth": -1, "reg_lambda": 1.0},
    {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 20, "max_depth": 8, "reg_lambda": 2.0},
    {"n_estimators": 400, "learning_rate": 0.04, "num_leaves": 31, "max_depth": 10, "reg_lambda": 2.0},
    {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 31, "max_depth": 8, "reg_lambda": 1.0},
]

results = []

for i, config in enumerate(configs, start=1):
    print(f"\nExperiment {i}/{len(configs)}: {config}")
    model = LGBMClassifier(
        **config,
        random_state=42,
        n_jobs=4,
        verbosity=-1,
    )

    start = time.time()
    model.fit(X_train_encoded, y_train)
    val_probabilities = model.predict_proba(X_val_encoded)[:, 1]
    predictions = (val_probabilities >= 0.44).astype(int)
    elapsed = time.time() - start

    row = {
        **config,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
        "seconds": round(elapsed, 2),
    }
    results.append(row)
    print(
        f"Accuracy={row['accuracy']:.4f} | "
        f"Precision={row['precision']:.4f} | "
        f"Recall={row['recall']:.4f} | "
        f"F1={row['f1']:.4f} | "
        f"Time={elapsed:.1f}s"
    )

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

print("\nLIGHTGBM TUNING RESULTS — RANKED BY VALIDATION F1")
print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

results_df.to_csv(
    "data/processed/lightgbm_tuning_results.csv",
    index=False,
)
print("\nSaved to data/processed/lightgbm_tuning_results.csv")
