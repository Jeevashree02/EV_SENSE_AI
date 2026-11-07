
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================
# 1. LOAD DATASET
# =====================================

print("\nLoading motor feature dataset...")

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    col for col in df.columns
    if col not in [
        "condition",
        "source_file",
        "load",
        "speed"
    ]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

print("Total samples:", len(df))
print("Total features:", len(features))
print("Fault classes:", sorted(y.unique()))
print("Original recordings:", groups.nunique())

# =====================================
# 2. GROUPED CROSS VALIDATION
# =====================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_predictions = np.empty(len(df), dtype=object)

fold_accuracies = []
fold_f1_scores = []

print("\nStarting 5-fold grouped validation...")

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    oof_predictions[test_idx] = predictions

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    fold_accuracies.append(accuracy)
    fold_f1_scores.append(macro_f1)

    print(f"\n========== FOLD {fold} ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Training recordings:", groups.iloc[train_idx].nunique())
    print("Testing recordings:", groups.iloc[test_idx].nunique())

# =====================================
# SAVE FRESH OOF PREDICTIONS
# =====================================

df["predicted"] = oof_predictions

df[
    ["condition", "predicted", "source_file", "speed", "load"]
].to_csv(
    "../dataset/cv_predictions.csv",
    index=False
)

print("\nFresh CV predictions saved!")
print("Location: dataset/cv_predictions.csv")

# =====================================
# 3. OVERALL PERFORMANCE
# =====================================

print("\n========== FINAL CV RESULTS ==========")

print(
    "Mean Accuracy:",
    round(np.mean(fold_accuracies), 4)
)

print(
    "Mean Macro F1:",
    round(np.mean(fold_f1_scores), 4)
)

print(
    "Accuracy Standard Deviation:",
    round(np.std(fold_accuracies), 4)
)

# =====================================
# 4. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y,
        oof_predictions,
        zero_division=0
    )
)

# =====================================
# 5. CONFUSION MATRIX
# =====================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y,
    oof_predictions,
    labels=labels
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

display.plot(cmap="Blues", xticks_rotation=45)

plt.title("Random Forest - Cross Validation")
plt.tight_layout()

plt.savefig("../dataset/final_confusion_matrix.png")
plt.show()

# =====================================
# 6. FEATURE IMPORTANCE
# =====================================

# Fit one final model on all data for feature inspection
final_model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": final_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n========== TOP 15 FEATURES ==========")
print(importance.head(15).to_string(index=False))

importance.to_csv(
    "../dataset/final_feature_importance.csv",
    index=False
)

# =====================================
# 7. SAVE FINAL MODEL
# =====================================

joblib.dump(
    {
        "model": final_model,
        "features": features,
        "classes": list(final_model.classes_)
    },
    "../model/final_random_forest.pkl"
)

print("\nFinal model saved successfully!")
print("Location: model/final_random_forest.pkl")

print("\n========== BACKEND MODEL TRAINING COMPLETE ==========")
