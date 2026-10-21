
import time
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

# 1. Load and clean the training data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# 2. Keep the existing 70/15/15 split in mind.
# Cross-validation uses ONLY the 70% training portion.
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 3. Preprocess numeric and categorical columns
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

# 4. Use our best tuned LightGBM configuration
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

# 5. Evaluate across five stratified folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Running 5-fold cross-validation...")
print("Only the training partition is being used.")

start = time.time()

results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    },
    n_jobs=1,
    return_train_score=False,
)

elapsed = time.time() - start

# 6. Display the results
for metric in ["accuracy", "precision", "recall", "f1"]:
    scores = results[f"test_{metric}"]
    print(f"\n{metric.upper()}")
    print("Fold scores:", [round(s, 4) for s in scores])
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std deviation: {scores.std():.4f}")

print(f"\nTotal runtime: {elapsed:.1f} seconds")
