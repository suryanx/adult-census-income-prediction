
import json
from pathlib import Path

import joblib
import pandas as pd

model_path = Path("models/final_lightgbm_pipeline.joblib")
config_path = Path("models/model_config.json")
test_path = Path("data/raw/adult_test.csv")

if not model_path.exists() or not config_path.exists():
    raise FileNotFoundError("Run src/finalize_model.py first.")

if not test_path.exists():
    raise FileNotFoundError(f"Test dataset not found: {test_path}")

model = joblib.load(model_path)

with open(config_path) as f:
    config = json.load(f)

test = pd.read_csv(test_path)

# The UCI Adult test file may contain a target column.
# Remove it so predictions use features only.
test.columns = test.columns.str.strip()
if "income" in test.columns:
    test = test.drop(columns="income")

# Match the feature columns used during training.
expected_features = config["features"]
missing = set(expected_features) - set(test.columns)

if missing:
    raise ValueError(f"Missing required test features: {sorted(missing)}")

test = test[expected_features].copy()

# Generate probabilities and classify at the selected threshold.
probabilities = model.predict_proba(test)[:, 1]
predictions = probabilities >= config["threshold"]

output = pd.DataFrame({
    "probability_over_50k": probabilities,
    "predicted_income": [
        ">50K" if pred else "<=50K" for pred in predictions
    ],
})

output_path = Path("submissions/adult_test_predictions.csv")
output.to_csv(output_path, index=False)

print(f"Test rows: {len(test):,}")
print(f"Predicted >50K: {predictions.sum():,}")
print(f"Predicted <=50K: {(~predictions).sum():,}")
print(f"Saved predictions to: {output_path}")
print("\nFirst 5 predictions:")
print(output.head())
