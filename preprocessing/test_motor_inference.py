
import numpy as np
import pandas as pd

from motor_inference import (
    extract_features,
    predict_motor_condition,
    features
)

# =====================================
# 1. LOAD REAL DATASET
# =====================================

print("Loading real vibration dataset...")

raw = pd.read_csv(
    "../dataset/combined_motor_data.csv"
)

original_features = pd.read_csv(
    "../dataset/motor_features.csv"
)

sensor_columns = ["gUserX", "gUserY", "gUserZ"]

# =====================================
# 2. SELECT FIRST RECORDING
# =====================================

original_row = original_features.iloc[0]

source_file = original_row["source_file"]

recording = raw[
    raw["source_file"] == source_file
].reset_index(drop=True)

test_window = recording.loc[
    :199, sensor_columns
].to_numpy(dtype=float)

actual_condition = original_row["condition"]

print("\nRecording:", source_file)
print("Actual condition:", actual_condition)
print("Window shape:", test_window.shape)

# =====================================
# 3. VERIFY FEATURE EXTRACTION
# =====================================

backend_features = extract_features(test_window)

original_values = original_row[features].astype(float)

backend_values = backend_features.iloc[0]

difference = (
    original_values - backend_values
).abs()

tolerance = 1e-5

print("\n========== FEATURE VALIDATION ==========")

print("Maximum difference:", difference.max())

print("Tolerance:", tolerance)

print(
    "Mismatched features:",
    (difference > tolerance).sum()
)

if (difference > tolerance).sum() == 0:
    print("SUCCESS: Features match!")
else:
    print("WARNING: Feature mismatch detected!")

# =====================================
# 4. PREDICT USING REAL DATA
# =====================================

print("\n========== REAL DATA PREDICTION ==========")

result = predict_motor_condition(test_window)

print("Actual condition:", actual_condition)

print(
    "Predicted condition:",
    result["predicted_condition"]
)

print("\nClass probabilities:")

for label, probability in result["class_probabilities"].items():

    print(f"{label}: {probability * 100:.2f}%")

print("\nReal data inference test completed!")
