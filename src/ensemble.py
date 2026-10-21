
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 1. Load data
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = (df["income"] == ">50K").astype(int)

# 2. Reproduce our existing split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# 3. Preprocessing for LightGBM and XGBoost
numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

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

# 4. Prepare CatBoost's categorical data
X_cat = X.copy()
for col in categorical_cols:
    X_cat[col] = X_cat[col].fillna("__MISSING__").astype(str)

X_cat_train = X_cat.loc[X_train.index]
X_cat_val = X_cat.loc[X_val.index]

# 5. Define models using our previous settings
models = {
    "LightGBM": Pipeline([
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
    ]),
    "XGBoost": Pipeline([
        ("preprocessing", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )),
    ]),
}

# 6. Train models and collect probabilities
probabilities = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    probabilities[name] = model.predict_proba(X_val)[:, 1]
    print(f"Completed in {time.time() - start:.1f}s")

cat_model = CatBoostClassifier(
    iterations=500,
    depth=7,
    learning_rate=0.05,
    loss_function="Logloss",
    verbose=False,
    random_seed=42,
    thread_count=4,
)

print("\nTraining CatBoost...")
start = time.time()
cat_model.fit(
    X_cat_train,
    y_train.loc[X_cat_train.index],
    cat_features=list(categorical_cols),
)
probabilities["CatBoost"] = cat_model.predict_proba(X_cat_val)[:, 1]
print(f"Completed in {time.time() - start:.1f}s")

# 7. Compare individual models with equal-weight ensemble
candidates = {
    "LightGBM": probabilities["LightGBM"],
    "XGBoost": probabilities["XGBoost"],
    "CatBoost": probabilities["CatBoost"],
    "Ensemble (equal average)": (
        probabilities["LightGBM"]
        + probabilities["XGBoost"]
        + probabilities["CatBoost"]
    ) / 3,
}

results = []

for name, prob in candidates.items():
    pred = (prob >= 0.5).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred, zero_division=0),
        "Recall": recall_score(y_val, pred, zero_division=0),
        "F1": f1_score(y_val, pred, zero_division=0),
    })

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)

print("\nVALIDATION RESULTS — threshold = 0.50")
print(results_df.to_string(
    index=False,
    float_format=lambda value: f"{value:.4f}"
))

results_df.to_csv(
    "data/processed/ensemble_results.csv",
    index=False
)
print("\nSaved: data/processed/ensemble_results.csv")
