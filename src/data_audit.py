
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")

# Column names from the UCI Adult dataset
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

# Load the original training data
train = pd.read_csv(
    DATA_DIR / "adult.data",
    names=columns,
    sep=",",
    skipinitialspace=True,
    na_values="?",
)

# Load the official test data; it has a comment line at the end
test = pd.read_csv(
    DATA_DIR / "adult.test",
    names=columns,
    sep=",",
    skipinitialspace=True,
    na_values="?",
    comment="|",
    skiprows=1,
)

# Standardise target labels: remove whitespace and trailing periods
for df in (train, test):
    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

print("TRAIN SHAPE:", train.shape)
print("TEST SHAPE:", test.shape)

print("\nTRAIN PREVIEW:")
print(train.head())

print("\nMISSING VALUES:")
print(train.isna().sum().sort_values(ascending=False))

print("\nTRAIN TARGET DISTRIBUTION:")
print(train["income"].value_counts(dropna=False))

# Save reproducible local CSV copies
train.to_csv(DATA_DIR / "adult_train.csv", index=False)
test.to_csv(DATA_DIR / "adult_test.csv", index=False)

print("\nSaved:")
print(DATA_DIR / "adult_train.csv")
print(DATA_DIR / "adult_test.csv")
