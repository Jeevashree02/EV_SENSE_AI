
import pandas as pd

df = pd.read_csv("../dataset/motor_features.csv")

selected_features = [
    "X_RMS",
    "Y_RMS",
    "Z_RMS",
    "X_PeakToPeak",
    "Y_PeakToPeak",
    "Z_PeakToPeak",
    "X_Kurtosis",
    "Y_Kurtosis",
    "Z_Kurtosis"
]

# Compare mean feature values by fault
# and operating condition
summary = (
    df.groupby(["condition", "speed", "load"])[selected_features]
    .mean()
    .round(4)
)

print("\n========== FAULT FEATURE COMPARISON ==========")

print(summary.to_string())

summary.to_csv(
    "../dataset/fault_condition_feature_summary.csv"
)

print("\nFeature comparison saved!")
