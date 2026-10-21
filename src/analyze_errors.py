
import pandas as pd

errors = pd.read_csv("data/processed/holdout_errors.csv")

# Keep only people the model incorrectly classified as <=50K
fn = errors[errors["error_type"] == "False negative"].copy()

print("FALSE NEGATIVES:", len(fn))

print("\n--- By occupation ---")
print(
    fn["occupation"]
    .value_counts(dropna=False)
    .head(10)
)

print("\n--- By education ---")
print(
    fn["education"]
    .value_counts(dropna=False)
    .head(10)
)

print("\n--- By age group ---")
fn["age_group"] = pd.cut(
    fn["age"],
    bins=[0, 25, 35, 45, 55, 65, 100],
    labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
)
print(fn["age_group"].value_counts(sort=False))

print("\n--- By weekly working hours ---")
print(
    fn["hours-per-week"]
    .describe()
    .round(1)
)

print("\n--- Most confident mistakes ---")
print(
    fn.sort_values("probability_over_50k")[
        ["age", "education", "occupation",
         "hours-per-week", "probability_over_50k"]
    ].head(10).to_string(index=False)
)



# Analyze false-negative rates across different groups
all_rows = pd.read_csv(
    "data/processed/holdout_all_predictions.csv"
)

# Only people who actually earn >50K
high_income = all_rows[
    all_rows["actual_income"] == ">50K"
].copy()

high_income["missed"] = (
    high_income["predicted_income"] == "<=50K"
).astype(int)

for column in ["education", "occupation"]:
    print(f"\n--- FALSE NEGATIVE RATE BY {column.upper()} ---")

    summary = high_income.groupby(
        column, dropna=False
    )["missed"].agg(
        total="count",
        missed="sum",
        miss_rate="mean"
    )

    summary["miss_rate"] *= 100

    print(
        summary.sort_values(
            "miss_rate", ascending=False
        ).round(1).to_string()
    )
