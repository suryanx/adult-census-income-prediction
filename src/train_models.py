
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

# 1. Load the labelled dataset
df = pd.read_csv("data/raw/adult_train.csv")
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")

X = df.drop(columns="income")
y = df["income"]

# 2. Split: 70% training, 15% validation, 15% final holdout
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_holdout, y_val, y_holdout = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Final holdout rows:", len(X_holdout))

# 3. Train a simple majority-class baseline
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# 4. Evaluate on validation data
val_predictions = baseline.predict(X_val)

print("\nBASELINE RESULTS")
print("Accuracy:", round(accuracy_score(y_val, val_predictions), 4))
print("F1 (>50K):", round(
    f1_score(y_val, val_predictions, pos_label=">50K"),
    4,
))
