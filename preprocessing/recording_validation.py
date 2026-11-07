
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =====================================
# 1. LOAD OOF PREDICTIONS
# =====================================

print("Loading OOF predictions...")

data = pd.read_csv("../dataset/cv_predictions.csv")

print("Data loaded successfully!")
print("Total windows:", len(data))
print("Total recordings:", data["source_file"].nunique())


# =====================================
# 2. RECORDING-LEVEL MAJORITY VOTING
# =====================================

print("\nPerforming recording-level majority voting...")

def majority_vote(series):

    return series.value_counts().idxmax()


recording_results = (
    data.groupby("source_file")
    .agg(
        actual_condition=("condition", "first"),
        predicted_condition=("predicted", majority_vote),
        speed=("speed", "first"),
        load=("load", "first"),
        total_windows=("predicted", "count"),
        winning_votes=("predicted", lambda x: x.value_counts().max())
    )
    .reset_index()
)

recording_results["vote_percentage"] = (
    recording_results["winning_votes"] /
    recording_results["total_windows"]
) * 100

recording_results["correct"] = (
    recording_results["actual_condition"] ==
    recording_results["predicted_condition"]
)


# =====================================
# 3. RECORDING-LEVEL ACCURACY
# =====================================

actual = recording_results["actual_condition"]

predicted = recording_results["predicted_condition"]

accuracy = accuracy_score(actual, predicted)

print("\n========== RECORDING-LEVEL RESULTS ==========")

print("Total recordings:", len(recording_results))

print("Correct predictions:", recording_results["correct"].sum())

print("Incorrect predictions:", (~recording_results["correct"]).sum())

print(f"Recording-level accuracy: {accuracy * 100:.2f}%")


# =====================================
# 4. CLASSIFICATION REPORT
# =====================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        actual,
        predicted,
        labels=["B1", "B2", "B3", "H", "R", "V"],
        zero_division=0
    )
)


# =====================================
# 5. CONFUSION MATRIX
# =====================================

labels = ["B1", "B2", "B3", "H", "R", "V"]

cm = confusion_matrix(
    actual,
    predicted,
    labels=labels
)

print("\n========== CONFUSION MATRIX ==========")

print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))


plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predicted Condition")
plt.ylabel("Actual Condition")
plt.title("Recording-Level Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "../dataset/recording_confusion_matrix.png",
    dpi=300
)

plt.show()


# =====================================
# 6. DISPLAY ALL RECORDING PREDICTIONS
# =====================================

print("\n========== INDIVIDUAL RECORDING RESULTS ==========")

print(
    recording_results[
        [
            "source_file",
            "actual_condition",
            "predicted_condition",
            "speed",
            "load",
            "vote_percentage",
            "correct"
        ]
    ].to_string(index=False)
)


# =====================================
# 7. SAVE RESULTS
# =====================================

recording_results.to_csv(
    "../dataset/recording_validation_results.csv",
    index=False
)

print("\nSaved:")
print("dataset/recording_validation_results.csv")
print("dataset/recording_confusion_matrix.png")

print("\nRECORDING VALIDATION COMPLETED!")
