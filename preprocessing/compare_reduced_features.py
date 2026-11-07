
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../dataset/motor_features.csv")

# Original 33 features
excluded = [
    "condition",
    "source_file",
    "speed",
    "load"
]

base_features = [
    col for col in df.columns
    if col not in excluded
]

# Candidate reduced feature set
remove_features = [
    "X_STD",
    "Y_STD",
    "Z_STD",
    "Y_PeakToPeak",
    "X_DominantAmplitude",
    "Z_DominantAmplitude",
    "X_HighBandRatio",
    "Y_HighBandRatio",
    "Z_HighBandRatio"
]

reduced_features = [
    col for col in base_features
    if col not in remove_features
]

X = df[base_features]
X_reduced = df[reduced_features]

y = df["condition"]
groups = df["source_file"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

base_acc = []
base_f1 = []

reduced_acc = []
reduced_f1 = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    # Baseline model
    model_base = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_base.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    pred_base = model_base.predict(
        X.iloc[test_idx]
    )

    # Reduced feature model
    model_reduced = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_reduced.fit(
        X_reduced.iloc[train_idx],
        y.iloc[train_idx]
    )

    pred_reduced = model_reduced.predict(
        X_reduced.iloc[test_idx]
    )

    y_test = y.iloc[test_idx]

    a1 = accuracy_score(y_test, pred_base)
    f1 = f1_score(
        y_test, pred_base,
        average="macro",
        zero_division=0
    )

    a2 = accuracy_score(y_test, pred_reduced)
    f2 = f1_score(
        y_test, pred_reduced,
        average="macro",
        zero_division=0
    )

    base_acc.append(a1)
    base_f1.append(f1)

    reduced_acc.append(a2)
    reduced_f1.append(f2)

    print(f"\nFOLD {fold}")
    print(f"Baseline accuracy: {a1:.4f}")
    print(f"Reduced accuracy: {a2:.4f}")
    print(f"Baseline macro F1: {f1:.4f}")
    print(f"Reduced macro F1: {f2:.4f}")

print("\n========== FINAL COMPARISON ==========")

print("Baseline features:", len(base_features))
print("Reduced features:", len(reduced_features))

print("Baseline accuracy:", round(np.mean(base_acc), 4))
print("Reduced accuracy:", round(np.mean(reduced_acc), 4))

print("Baseline F1:", round(np.mean(base_f1), 4))
print("Reduced F1:", round(np.mean(reduced_f1), 4))
