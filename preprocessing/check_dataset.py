import pandas as pd

df = pd.read_csv("../dataset/motor_features.csv")

print("\n===== RECORDINGS PER CONDITION =====")
print(
    df.groupby("condition")["source_file"]
    .nunique()
)

print("\n===== RECORDINGS PER CONDITION AND SPEED =====")
recordings = df[
    ["condition", "source_file", "speed", "load"]
].drop_duplicates()

print(
    recordings.groupby(
        ["condition", "speed", "load"]
    ).size()
)

print("\n===== WINDOWS PER RECORDING =====")
print(
    df.groupby("source_file").size().describe()
)

print("\n===== SPEED AND LOAD VALUES =====")
print("Speeds:", sorted(df["speed"].unique()))
print("Loads:", sorted(df["load"].unique()))