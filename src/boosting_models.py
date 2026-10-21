
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. Load data and clean labels
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = df["income"]

# 2. Reproduce our fixed split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# Encode target for XGBoost, LightGBM and HistGradientBoosting
y_train_bin = (y_train == ">50K").astype(int)
y_val_bin = (y_val == ">50K").astype(int)

# 3. Shared preprocessing for models that need numeric input
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns


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


# 4. Define models
models = {
    "XGBoost": Pipeline([
        ("preprocessing", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=4,
        )),
    ]),
    "LightGBM": Pipeline([
        ("preprocessing", preprocessor),
        ("model", LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, random_state=42, n_jobs=4,
            verbosity=-1,
        )),
    ]),
    "HistGradientBoosting": Pipeline([
        ("preprocessing", preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08,
            max_leaf_nodes=31, random_state=42,
        )),
    ]),
}

# CatBoost handles categorical features natively.
# Convert missing values to strings and identify categorical columns.
X_cat = X.copy()
for col in categorical_cols:
    X_cat[col] = X_cat[col].fillna("__MISSING__").astype(str)

X_cat_train, X_cat_temp, y_cat_train, y_cat_temp = train_test_split(
    X_cat, y, test_size=0.30, random_state=42, stratify=y
)
X_cat_val = X_cat_temp.iloc[
    X_temp.index.get_indexer(X_val.index)
] if False else X_cat.loc[X_val.index]

cat_model = CatBoostClassifier(
    iterations=500, depth=7, learning_rate=0.05,
    loss_function="Logloss", verbose=False,
    random_seed=42, thread_count=4,
)
models["CatBoost"] = cat_model

# 5. Train and evaluate
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()

    if name == "CatBoost":
        model.fit(
            X_cat_train, (y_cat_train == ">50K").astype(int),
            cat_features=list(categorical_cols),
        )
        pred = model.predict(X_cat_val).reshape(-1).astype(int)
        y_true = y_val_bin
    else:
        model.fit(X_train, y_train_bin)
        pred = model.predict(X_val).astype(int)
        y_true = y_val_bin

    elapsed = time.time() - start

    accuracy = accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision_>50K": precision,
        "Recall_>50K": recall,
        "F1_>50K": f1,
        "Seconds": elapsed,
    })

    print(
        f"Accuracy={accuracy:.4f} | Precision={precision:.4f} | "
        f"Recall={recall:.4f} | F1={f1:.4f} | Time={elapsed:.1f}s"
    )

results_df = pd.DataFrame(results).sort_values(
    "F1_>50K", ascending=False
)
print("\nBOOSTING MODEL COMPARISON")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results_df.to_csv("data/processed/boosting_model_results.csv", index=False)
print("\nSaved to data/processed/boosting_model_results.csv")
