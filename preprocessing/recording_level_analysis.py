
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load fresh out-of-fold predictions
df = pd.read_csv("../dataset/cv_predictions.csv")

# Find the most frequent prediction for each recording
recording_results = (
    df.groupby("source_file")
    .agg(
        actual=("condition", "first"),
        predicted=("predicted", lambda x: x.mode().iloc[0]),
        total_windows=("predicted", "count")
    )
    .reset_index()
)

recording_results["correct"] = (
    recording_results["actual"] ==
    recording_results["predicted"]
)

recording_results["accuracy"] = (
    recording_results["correct"].astype(float) * 100
)

print("\n========== RECORDING LEVEL RESULTS ==========")

print(
    recording_results[
        ["source_file", "actual", "predicted",
         "total_windows", "accuracy"]
    ].to_string(index=False)
)

print("\n========== OVERALL RECORDING ACCURACY ==========")

print(
    round(recording_results["correct"].mean() * 100, 2),
    "%"
)

print("\n========== RECORDING LEVEL CLASSIFICATION REPORT ==========")

print(
    classification_report(
        recording_results["actual"],
        recording_results["predicted"],
        zero_division=0
    )
)

recording_results.to_csv(
    "../dataset/recording_level_results.csv",
    index=False
)

print("\nRecording-level results saved!")
