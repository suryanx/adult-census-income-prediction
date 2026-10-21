
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from lightgbm import LGBMClassifier

# Load and clean data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# Reproduce the exact split used in earlier experiments
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# Keep validation separate from the final holdout
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )),
    ]), categorical_cols),
])

model = Pipeline([
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
])

# Train on the training partition only
model.fit(X_train, y_train)

# FINAL HOLDOUT EVALUATION — do not tune using these results
probabilities = model.predict_proba(X_holdout)[:, 1]
predictions = (probabilities >= 0.44).astype(int)

print("FINAL HOLDOUT RESULTS")
print(f"Accuracy:  {accuracy_score(y_holdout, predictions):.4f}")
print(f"Precision: {precision_score(y_holdout, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_holdout, predictions, zero_division=0):.4f}")
print(f"F1:        {f1_score(y_holdout, predictions, zero_division=0):.4f}")


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# Get probabilities and predictions at our chosen threshold
probabilities = model.predict_proba(X_holdout)[:, 1]
predictions = (probabilities >= 0.44).astype(int)

# Convert actual labels to binary values
actual = y_holdout.astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(
    actual, predictions, labels=[0, 1]
).ravel()

print("\n--- ERROR ANALYSIS ---")
print(f"Correct lower-income predictions: {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"Correct higher-income predictions: {tp:,}")

print("\nClassification report:")
print(classification_report(
    actual, predictions,
    labels=[0, 1],
    target_names=["<=50K", ">50K"],
    zero_division=0
))

# Inspect individual mistakes
errors = X_holdout.copy()
errors["actual_income"] = np.where(actual == 1, ">50K", "<=50K")
errors["predicted_income"] = np.where(
    predictions == 1, ">50K", "<=50K"
)
errors["probability_over_50k"] = probabilities
errors["error_type"] = np.where(
    (actual == 0) & (predictions == 1), "False positive",
    np.where(
        (actual == 1) & (predictions == 0), "False negative",
        "Correct"
    )
)

mistakes = errors[errors["error_type"] != "Correct"]
mistakes.to_csv("data/processed/holdout_errors.csv", index=False)
# Save ALL holdout rows, including correct predictions
errors.to_csv("data/processed/holdout_all_predictions.csv", index=False)

print("Saved all holdout predictions.")

print(f"\nTotal mistakes: {len(mistakes):,}")
print("Saved individual errors to data/processed/holdout_errors.csv")
print("\nSample false negatives:")
print(
    mistakes[mistakes["error_type"] == "False negative"]
    .head(10).to_string(index=False)
)