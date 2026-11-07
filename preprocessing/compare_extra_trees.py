
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("../dataset/motor_features.csv")

features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

# Same validation folds
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_acc = []
rf_f1 = []

et_acc = []
et_f1 = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Extra Trees
    et = ExtraTreesClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    et.fit(X_train, y_train)
    et_pred = et.predict(X_test)

    # Metrics
    rf_acc.append(accuracy_score(y_test, rf_pred))
    rf_f1.append(
        f1_score(y_test, rf_pred, average="macro")
    )

    et_acc.append(accuracy_score(y_test, et_pred))
    et_f1.append(
        f1_score(y_test, et_pred, average="macro")
    )

    print(f"\n========== FOLD {fold} ==========")

    print("RF Accuracy:", round(rf_acc[-1], 4))
    print("RF Macro F1:", round(rf_f1[-1], 4))

    print("Extra Trees Accuracy:", round(et_acc[-1], 4))
    print("Extra Trees Macro F1:", round(et_f1[-1], 4))

# Final results
print("\n========== FINAL COMPARISON ==========")

print("\nRANDOM FOREST")
print("Accuracy:", np.mean(rf_acc))
print("Macro F1:", np.mean(rf_f1))

print("\nEXTRA TREES")
print("Accuracy:", np.mean(et_acc))
print("Macro F1:", np.mean(et_f1))
