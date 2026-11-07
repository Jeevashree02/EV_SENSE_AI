
import pandas as pd

df = pd.read_csv("../dataset/motor_features.csv")

important_features = [
    "Y_PeakToPeak",
    "Z_PeakToPeak",
    "X_MidBandRatio",
    "X_PeakToPeak",
    "Y_STD",
    "Z_RMS",
    "Y_RMS",
    "X_HighBandRatio"
]

print("\n===== FAULT-WISE FEATURE MEDIANS =====")

result = df.groupby("condition")[important_features].median()

print(result.round(4))

print("\n===== FAULT-WISE FEATURE MEANS =====")

result_mean = df.groupby("condition")[important_features].mean()

print(result_mean.round(4))

print("\n===== FAULT DISTRIBUTION BY SPEED =====")

print(
    df.groupby(["condition", "speed"]).size()
)

print("\n===== FAULT DISTRIBUTION BY LOAD =====")

print(
    df.groupby(["condition", "load"]).size()
)

print("\nAnalysis completed!")
