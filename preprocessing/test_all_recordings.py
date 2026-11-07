
import numpy as np
import pandas as pd

from motor_inference import extract_features, model

# =====================================
# 1. LOAD DATA
# =====================================

print("Loading dataset...")

raw = pd.read_csv("../dataset/combined_motor_data.csv")
metadata = pd.read_csv("../dataset/motor_features.csv")

sensor_columns = ["gUserX", "gUserY", "gUserZ"]

results = []

# =====================================
# 2. PROCESS RECORDINGS
# =====================================

recordings = metadata["source_file"].drop_duplicates()

print("Total recordings:", len(recordings))

for index, source_file in enumerate(recordings, start=1):

    recording = raw[
        raw["source_file"] == source_file
    ].reset_index(drop=True)

    actual = metadata.loc[
        metadata["source_file"] == source_file,
        "condition"
    ].iloc[0]

    windows = []

    # Create non-overlapping windows
    for start in range(0, len(recording) - 199, 200):

        window = recording.loc[
            start:start + 199,
            sensor_columns
        ].to_numpy(dtype=float)

        windows.append(window)

    # Extract features
    feature_frames = [
        extract_features(window)
        for window in windows
    ]

    X_test = pd.concat(
        feature_frames,
        ignore_index=True
    )

    # Batch prediction: one model call per recording
    predictions = model.predict(X_test)

    prediction = pd.Series(
        predictions
    ).value_counts().idxmax()

    results.append({
        "source_file": source_file,
        "actual_condition": actual,
        "predicted_condition": prediction,
        "total_windows": len(windows),
        "correct": actual == prediction
    })

    print(
        f"Processed {index}/{len(recordings)}: {source_file}"
    )

# =====================================
# 3. DISPLAY RESULTS
# =====================================

results_df = pd.DataFrame(results)

accuracy = results_df["correct"].mean() * 100

print("\n========== FINAL RESULTS ==========")

print(results_df.to_string(index=False))

print("\nTotal recordings:", len(results_df))
print("Correct:", results_df["correct"].sum())
print("Incorrect:", (~results_df["correct"]).sum())

print(f"Accuracy: {accuracy:.2f}%")

# =====================================
# 4. SAVE RESULTS
# =====================================

results_df.to_csv(
    "../dataset/inference_all_recordings.csv",
    index=False
)

print("\nSaved: dataset/inference_all_recordings.csv")

print("\nBATCH INFERENCE TEST COMPLETED!")
