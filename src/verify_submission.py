
import pandas as pd
from pathlib import Path

test_path = Path("data/raw/adult_test.csv")
pred_path = Path("submissions/adult_test_predictions.csv")

test = pd.read_csv(test_path)
pred = pd.read_csv(pred_path)

assert len(test) == len(pred), (
    f"Row mismatch: test={len(test)}, predictions={len(pred)}"
)
assert list(pred.columns) == [
    "probability_over_50k", "predicted_income"
]
assert pred["probability_over_50k"].between(0, 1).all()
assert pred["predicted_income"].isin([">50K", "<=50K"]).all()
assert pred.notna().all().all()

print("PASS: row counts match")
print("PASS: output columns are correct")
print("PASS: probabilities are within [0, 1]")
print("PASS: predictions and missing-value checks passed")
print(f"Verified {len(pred):,} predictions.")
