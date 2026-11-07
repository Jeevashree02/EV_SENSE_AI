
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

# --------------------------------
# CROSS VALIDATION
# --------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rf_accuracies = []
rf_f1_scores = []

svm_accuracies = []
svm_f1_scores = []

# --------------------------------
# TRAIN AND EVALUATE
# --------------------------------

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

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(
        y_test, rf_pred,
        average="macro",
        zero_division=0
    )

    rf_accuracies.append(rf_acc)
    rf_f1_scores.append(rf_f1)

    # SVM with feature scaling
    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )

    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)
    svm_f1 = f1_score(
        y_test, svm_pred,
        average="macro",
        zero_division=0
    )

    svm_accuracies.append(svm_acc)
    svm_f1_scores.append(svm_f1)

    print(f"\n========== FOLD {fold} ==========")
    print(f"RF Accuracy:  {rf_acc:.4f}")
    print(f"RF Macro F1:  {rf_f1:.4f}")
    print(f"SVM Accuracy: {svm_acc:.4f}")
    print(f"SVM Macro F1: {svm_f1:.4f}")

# --------------------------------
# FINAL COMPARISON
# --------------------------------

print("\n========== FINAL RESULTS ==========")

print("\nRANDOM FOREST")
print("Mean Accuracy:", np.mean(rf_accuracies))
print("Mean Macro F1:", np.mean(rf_f1_scores))
print("Accuracy Std:", np.std(rf_accuracies))

print("\nSVM")
print("Mean Accuracy:", np.mean(svm_accuracies))
print("Mean Macro F1:", np.mean(svm_f1_scores))
print("Accuracy Std:", np.std(svm_accuracies))
