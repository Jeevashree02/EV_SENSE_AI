
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load the existing feature dataset
df = pd.read_csv("../dataset/motor_features.csv")

features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

# Four requested train-test ratios
ratios = {
    "70:30": 0.30,
    "80:20": 0.20,
    "90:10": 0.10,
    "95:5": 0.05
}

results = []
REPEATS = 30

for ratio_name, test_size in ratios.items():

    accuracies = []
    f1_scores = []
    actual_test_percentages = []

    splitter = GroupShuffleSplit(
        n_splits=REPEATS,
        test_size=test_size,
        random_state=42
    )

    for train_idx, test_idx in splitter.split(X, y, groups):

        # Verify no recording appears in both sets
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        assert train_groups.isdisjoint(test_groups)

        model = RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        predictions = model.predict(X.iloc[test_idx])

        accuracies.append(
            accuracy_score(y.iloc[test_idx], predictions)
        )

        f1_scores.append(
            f1_score(
                y.iloc[test_idx],
                predictions,
                average="macro",
                zero_division=0
            )
        )

        actual_test_percentages.append(
            len(test_idx) / len(df) * 100
        )

    results.append({
        "Split": ratio_name,
        "Mean Accuracy (%)": np.mean(accuracies) * 100,
        "Accuracy Std (%)": np.std(accuracies) * 100,
        "Mean Macro F1 (%)": np.mean(f1_scores) * 100,
        "Actual Test Data (%)": np.mean(actual_test_percentages)
    })

    print(f"{ratio_name} experiment completed")

# Display comparison
results_df = pd.DataFrame(results)

print("\n===== TRAIN-TEST ACCURACY COMPARISON =====")
print(results_df.round(2).to_string(index=False))

# Save results
results_df.to_csv(
    "../dataset/train_test_comparison.csv",
    index=False
)

# Draw accuracy comparison graph
plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Split"],
    results_df["Mean Accuracy (%)"],
    yerr=results_df["Accuracy Std (%)"],
    capsize=5
)

plt.xlabel("Training : Testing Ratio")
plt.ylabel("Mean Accuracy (%)")
plt.title("Random Forest Accuracy Comparison")
plt.ylim(0, 100)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "../dataset/accuracy_comparison.png",
    dpi=300
)

plt.show()

print("\nResults and graph saved successfully!")
