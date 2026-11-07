
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# =====================================
# 1. LOAD DATASET
# =====================================

print("\nLoading motor feature dataset...")

df = pd.read_csv("../dataset/motor_features.csv")

excluded = [
    "condition",
    "source_file",
    "speed",
    "load"
]

features = [
    col for col in df.columns
    if col not in excluded
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

print("Total features:", len(features))
print("Total recordings:", groups.nunique())

# =====================================
# 2. CROSS VALIDATION
# =====================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

feature_counts = [25, 20, 15]

results = {
    33: {"accuracy": [], "f1": []},
    25: {"accuracy": [], "f1": []},
    20: {"accuracy": [], "f1": []},
    15: {"accuracy": [], "f1": []}
}

selected_feature_frequency = {
    count: {} for count in feature_counts
}

# =====================================
# 3. TRAIN AND TEST EACH FOLD
# =====================================

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Feature ranking uses training data only
    ranking_model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    ranking_model.fit(X_train, y_train)

    importance = pd.Series(
        ranking_model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    # ---------------------------------
    # Baseline: all 33 features
    # ---------------------------------

    baseline_predictions = ranking_model.predict(X_test)

    results[33]["accuracy"].append(
        accuracy_score(y_test, baseline_predictions)
    )

    results[33]["f1"].append(
        f1_score(
            y_test,
            baseline_predictions,
            average="macro",
            zero_division=0
        )
    )

    # ---------------------------------
    # Reduced feature experiments
    # ---------------------------------

    for count in feature_counts:

        selected_features = importance.head(count).index.tolist()

        for feature in selected_features:
            selected_feature_frequency[count][feature] = (
                selected_feature_frequency[count].get(feature, 0) + 1
            )

        reduced_model = RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        reduced_model.fit(
            X_train[selected_features],
            y_train
        )

        predictions = reduced_model.predict(
            X_test[selected_features]
        )

        accuracy = accuracy_score(y_test, predictions)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )

        results[count]["accuracy"].append(accuracy)
        results[count]["f1"].append(macro_f1)

        print(
            f"Top {count} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    print(
        "Baseline | "
        f"Accuracy: {results[33]['accuracy'][-1]:.4f} | "
        f"Macro F1: {results[33]['f1'][-1]:.4f}"
    )

# =====================================
# 4. FINAL COMPARISON
# =====================================

print("\n========== FINAL FEATURE COMPARISON ==========")

summary = []

for count in [33, 25, 20, 15]:

    mean_accuracy = np.mean(results[count]["accuracy"])
    mean_f1 = np.mean(results[count]["f1"])

    std_accuracy = np.std(results[count]["accuracy"])

    summary.append({
        "Features": count,
        "Mean Accuracy": mean_accuracy,
        "Mean Macro F1": mean_f1,
        "Accuracy Std": std_accuracy
    })

    print(f"\nFeatures: {count}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Macro F1: {mean_f1:.4f}")
    print(f"Accuracy Std: {std_accuracy:.4f}")

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "../dataset/feature_selection_comparison.csv",
    index=False
)

# =====================================
# 5. FEATURE SELECTION FREQUENCY
# =====================================

frequency_rows = []

for count, frequency in selected_feature_frequency.items():

    for feature, times in frequency.items():

        frequency_rows.append({
            "Feature_Count": count,
            "Feature": feature,
            "Selected_Folds": times
        })

frequency_df = pd.DataFrame(frequency_rows)

frequency_df.to_csv(
    "../dataset/selected_feature_frequency.csv",
    index=False
)

print("\nComparison saved successfully!")
print("Feature frequency saved successfully!")
