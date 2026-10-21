
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Paths
Path("models").mkdir(exist_ok=True)
Path("submissions").mkdir(exist_ok=True)

# 1. Load all labeled training data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# 2. Define preprocessing
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ]), categorical_cols),
])

# 3. Build and train the selected model on all labeled data
final_model = Pipeline([
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

print(f"Training on {len(X):,} labeled rows...")
final_model.fit(X, y)

# 4. Save the complete pipeline and prediction threshold
joblib.dump(final_model, "models/final_lightgbm_pipeline.joblib")

with open("models/model_config.json", "w") as f:
    json.dump({
        "model": "LightGBM",
        "positive_class": ">50K",
        "threshold": 0.44,
        "features": list(X.columns),
    }, f, indent=2)

print("Saved model and configuration.")

# 5. Verify that the saved model loads and predicts
loaded_model = joblib.load("models/final_lightgbm_pipeline.joblib")
probabilities = loaded_model.predict_proba(X.head(5))[:, 1]
predictions = (probabilities >= 0.44).astype(int)

print("\nSaved-model smoke test:")
print(pd.DataFrame({
    "probability_over_50k": probabilities.round(4),
    "predicted_income": [
        ">50K" if p == 1 else "<=50K" for p in predictions
    ],
}))
print("\nModel packaging successful.")
