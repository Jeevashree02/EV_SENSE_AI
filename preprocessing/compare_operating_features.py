
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("../dataset/motor_features.csv")

# Model A: Vibration features only
base_features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

# Model B: Vibration + operating conditions
extended_features = base_features + ["speed", "load"]

y = df["condition"]
groups = df["source_file"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = {
    "Base Accuracy": [],
    "Base F1": [],
    "Extended Accuracy": [],
    "Extended F1": []
}

for fold, (train_idx, test_idx) in enumerate(
    cv.split(df[base_features], y, groups), start=1
):

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Model A
    model_a = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_a.fit(
        df.iloc[train_idx][base_features],
        y_train
    )

    pred_a = model_a.predict(
        df.iloc[test_idx][base_features]
    )

    # Model B
    model_b = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model_b.fit(
        df.iloc[train_idx][extended_features],
        y_train
    )

    pred_b = model_b.predict(
        df.iloc[test_idx][extended_features]
    )

    results["Base Accuracy"].append(
        accuracy_score(y_test, pred_a)
    )

    results["Base F1"].append(
        f1_score(y_test, pred_a, average="macro")
    )

    results["Extended Accuracy"].append(
        accuracy_score(y_test, pred_b)
    )

    results["Extended F1"].append(
        f1_score(y_test, pred_b, average="macro")
    )

    print(f"\nFOLD {fold}")
    print("Base accuracy:", results["Base Accuracy"][-1])
    print("Extended accuracy:", results["Extended Accuracy"][-1])
    print("Base macro F1:", results["Base F1"][-1])
    print("Extended macro F1:", results["Extended F1"][-1])

print("\n========== FINAL COMPARISON ==========")

for name, scores in results.items():
    print(name, ":", round(np.mean(scores), 4))
