
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = df["income"]

# 2. Reproduce our existing split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# 3. Preprocessing
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_cols),
])

# 4. Candidate models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    ),
}

# 5. Train and evaluate
results = []

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", classifier),
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, pos_label=">50K", zero_division=0
    )
    recall = recall_score(
        y_val, predictions, pos_label=">50K", zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, pos_label=">50K", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision_>50K": precision,
        "Recall_>50K": recall,
        "F1_>50K": f1,
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

# 6. Compare candidates
results_df = pd.DataFrame(results).sort_values(
    "F1_>50K", ascending=False
)

print("\nMODEL COMPARISON — SORTED BY F1")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Save experiment results
results_df.to_csv("data/processed/tree_model_results.csv", index=False)
print("\nSaved results to data/processed/tree_model_results.csv")
