
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.dummy import DummyClassifier

# Load data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = df["income"]

# Recreate the same reproducible split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# Identify column types
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

# Numeric preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols),
])

# Train Logistic Regression
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_val)

# Compare against the dummy baseline on identical rows
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_val)

for name, pred in [
    ("Dummy baseline", baseline_predictions),
    ("Logistic Regression", predictions),
]:
    print(f"\n{name}")
    print("Accuracy:", round(accuracy_score(y_val, pred), 4))
    print("Precision (>50K):", round(
        precision_score(y_val, pred, pos_label=">50K", zero_division=0), 4
    ))
    print("Recall (>50K):", round(
        recall_score(y_val, pred, pos_label=">50K", zero_division=0), 4
    ))
    print("F1 (>50K):", round(
        f1_score(y_val, pred, pos_label=">50K", zero_division=0), 4
    ))

print("\nLOGISTIC REGRESSION CLASSIFICATION REPORT")
print(classification_report(y_val, predictions, zero_division=0))

# Show predictions for 10 unseen validation rows
comparison = pd.DataFrame({
    "actual": y_val.iloc[:10].to_numpy(),
    "predicted": predictions[:10],
})
comparison["correct"] = comparison["actual"] == comparison["predicted"]

print("\nFIRST 10 VALIDATION PREDICTIONS")
print(comparison.to_string(index=False))
