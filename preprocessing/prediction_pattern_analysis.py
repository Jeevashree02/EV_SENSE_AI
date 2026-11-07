
import pandas as pd

df = pd.read_csv("../dataset/cv_predictions.csv")

# Extract speed and load from recording filename
df["speed"] = df["source_file"].str.extract(
    r"_(\d+)Hz"
).astype(int)

df["load"] = df["source_file"].str.extract(
    r"_(\d+)_\d+Hz"
).astype(int)

# Prediction distribution per recording
print("\n===== PREDICTED CLASS PER RECORDING =====")

for filename, group in df.groupby("source_file"):

    actual = group["condition"].iloc[0]

    predictions = group["predicted"].value_counts(
        normalize=True
    ) * 100

    print(f"\nRecording: {filename}")
    print("Actual condition:", actual)
    print("Predicted distribution:")

    print(predictions.round(2).to_string())

# Accuracy grouped by speed
print("\n===== ACCURACY BY SPEED =====")

df["correct"] = (
    df["condition"] == df["predicted"]
)

print(
    df.groupby("speed")["correct"]
    .mean()
    .mul(100)
    .round(2)
)

# Accuracy grouped by load
print("\n===== ACCURACY BY LOAD =====")

print(
    df.groupby("load")["correct"]
    .mean()
    .mul(100)
    .round(2)
)

# True condition versus predicted condition
print("\n===== CONDITION PREDICTION TABLE =====")

print(
    pd.crosstab(
        df["condition"],
        df["predicted"],
        normalize="index"
    ).round(3)
)

print("\nAnalysis completed!")
