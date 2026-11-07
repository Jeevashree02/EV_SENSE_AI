
import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset/cv_predictions.csv")

# Calculate prediction accuracy for each recording
results = []

for filename, group in df.groupby("source_file"):
    accuracy = accuracy_score(
        group["condition"],
        group["predicted"]
    )

    results.append({
        "Recording": filename,
        "Actual_Class": group["condition"].iloc[0],
        "Speed": group["speed"].iloc[0],
        "Load": group["load"].iloc[0],
        "Samples": len(group),
        "Accuracy": round(accuracy * 100, 2)
    })

result_df = pd.DataFrame(results)

print("\nACCURACY BY RECORDING\n")
print(
    result_df.sort_values("Accuracy").to_string(index=False)
)

print("\nAVERAGE ACCURACY BY CLASS\n")
print(
    result_df.groupby("Actual_Class")["Accuracy"]
    .mean()
    .round(2)
)

result_df.to_csv(
    "dataset/recording_accuracy_analysis.csv",
    index=False
)

print("\nSaved recording_accuracy_analysis.csv")


print("\nACCURACY BY SPEED\n")

speed_results = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("speed")["correct"]
    .mean() * 100
)

print(speed_results.round(2))

print("\nACCURACY BY LOAD\n")

load_results = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby("load")["correct"]
    .mean() * 100
)

print(load_results.round(2))

print("\nACCURACY BY CONDITION AND SPEED\n")

condition_speed = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby(["condition", "speed"])["correct"]
    .mean() * 100
)

print(condition_speed.round(2))

print("\nACCURACY BY CONDITION AND LOAD\n")

condition_load = (
    df.assign(correct=df["condition"] == df["predicted"])
    .groupby(["condition", "load"])["correct"]
    .mean() * 100
)

print(condition_load.round(2))


import pandas as pd

df = pd.read_csv("dataset/cv_predictions.csv")

# Count predictions made for each recording
for filename in [
    "B1_0_30Hz.csv",
    "B1_1_50Hz.csv",
    "B2_0_50Hz.csv",
    "B3_0_30Hz.csv",
    "H_0_30Hz.csv",
    "V_0_50Hz.csv"
]:
    group = df[df["source_file"] == filename]

    print("\n------------------------------")
    print("RECORDING:", filename)
    print("ACTUAL CLASS:", group["condition"].iloc[0])
    print("PREDICTION COUNTS:")
    print(group["predicted"].value_counts())
