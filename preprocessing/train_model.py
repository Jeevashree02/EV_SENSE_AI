
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

print("Total samples:", len(df))
print("Total features:", len(features))
print("Total recordings:", groups.nunique())
print("Classes:", sorted(y.unique()))

# ==========================================
# 2. STRATIFIED GROUP CROSS VALIDATION
# ==========================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store predictions for every test window
oof_predictions = np.empty(len(df), dtype=object)

# Track how many times each row is tested
test_counts = np.zeros(len(df), dtype=int)

# ==========================================
# 3. TRAIN AND TEST EACH FOLD
# ==========================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    # Verify no recording appears in both sets
    assert train_groups.isdisjoint(test_groups), (
        "DATA LEAKAGE DETECTED!"
    )

    print(f"\n========== FOLD {fold} ==========")

    print("Training samples:", len(train_idx))
    print("Testing samples:", len(test_idx))
    print("Training recordings:", len(train_groups))
    print("Testing recordings:", len(test_groups))

    print("Training classes:")
    print(y.iloc[train_idx].value_counts().sort_index())

    print("Testing classes:")
    print(y.iloc[test_idx].value_counts().sort_index())

    print("Recording separation: PASSED")

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    # Predict unseen recordings
    predictions = model.predict(X.iloc[test_idx])

    # Store predictions
    oof_predictions[test_idx] = predictions

    # Track test coverage
    test_counts[test_idx] += 1

    fold_accuracy = accuracy_score(
        y.iloc[test_idx],
        predictions
    )

    fold_f1 = f1_score(
        y.iloc[test_idx],
        predictions,
        average="macro",
        zero_division=0
    )

    print(f"Fold accuracy: {fold_accuracy * 100:.2f}%")
    print(f"Fold macro F1: {fold_f1:.4f}")

# ==========================================
# 4. VERIFY PREDICTION COMPLETENESS
# ==========================================

print("\n========== VALIDATION CHECK ==========")

assert np.all(test_counts == 1), (
    "ERROR: Some rows were not tested exactly once!"
)

assert not pd.isna(oof_predictions).any(), (
    "ERROR: Missing predictions detected!"
)

print("Recording separation: PASSED")
print("Every sample tested exactly once: PASSED")
print("Missing predictions: NONE")

# ==========================================
# 5. OVERALL PERFORMANCE
# ==========================================

df["predicted"] = oof_predictions

print("\n========== OVERALL PERFORMANCE ==========")

overall_accuracy = accuracy_score(
    y,
    df["predicted"]
)

overall_f1 = f1_score(
    y,
    df["predicted"],
    average="macro",
    zero_division=0
)

print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
print(f"Macro F1-score: {overall_f1:.4f}")

# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y,
        df["predicted"],
        zero_division=0
    )
)

# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

print("\n========== CONFUSION MATRIX ==========")

labels = sorted(y.unique())

cm = confusion_matrix(
    y,
    df["predicted"],
    labels=labels
)

print("Class order:", labels)
print(cm)

# ==========================================
# 8. ACCURACY BY SPEED
# ==========================================

print("\n========== ACCURACY BY SPEED ==========")

speed_accuracy = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("speed")["correct"]
    .mean() * 100
)

print(speed_accuracy.round(2))

# ==========================================
# 9. ACCURACY BY LOAD
# ==========================================

print("\n========== ACCURACY BY LOAD ==========")

load_accuracy = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("load")["correct"]
    .mean() * 100
)

print(load_accuracy.round(2))

# ==========================================
# 10. ACCURACY BY CONDITION
# ==========================================

print("\n========== ACCURACY BY CONDITION ==========")

condition_accuracy = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("condition")["correct"]
    .mean() * 100
)

print(condition_accuracy.round(2))

# ==========================================
# 11. ACCURACY BY RECORDING
# ==========================================

print("\n========== ACCURACY BY RECORDING ==========")

recording_accuracy = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("source_file")["correct"]
    .mean() * 100
)

print(recording_accuracy.round(2))

# ==========================================
# 12. SAVE PREDICTIONS
# ==========================================

df.to_csv(
    "../dataset/cv_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
print("File: dataset/cv_predictions.csv")
print("\nCROSS-VALIDATION COMPLETED!")
