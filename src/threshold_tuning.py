
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from lightgbm import LGBMClassifier

# Load and clean
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# Same fixed split as our tuning experiments
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

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

# Best configuration from our current search
model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=10,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

model.fit(X_train_encoded, y_train)
probabilities = model.predict_proba(X_val_encoded)[:, 1]

# Compare thresholds using validation predictions only
results = []

for threshold in np.arange(0.30, 0.701, 0.02):
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
    })

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

print("\nTHRESHOLD SEARCH — VALIDATION ONLY")
print(results_df.head(10).to_string(
    index=False, float_format=lambda value: f"{value:.4f}"
))

results_df.to_csv(
    "data/processed/threshold_results.csv", index=False
)
print("\nSaved to data/processed/threshold_results.csv")
