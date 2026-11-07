
import pandas as pd

df = pd.read_csv("../dataset/cv_predictions.csv")

# Extract speed and load from filenames
df["speed"] = df["source_file"].str.extract(
    r"_(\d+)Hz"
).astype(int)

df["load"] = df["source_file"].str.extract(
    r"_(\d+)_\d+Hz"
).astype(int)

# Correct prediction indicator
df["correct"] = (
    df["condition"] == df["predicted"]
)

# --------------------------------
# 1. ACCURACY BY SPEED AND LOAD
# --------------------------------

print("\n===== ACCURACY BY SPEED AND LOAD =====")

result = (
    df.groupby(["speed", "load"])["correct"]
    .mean()
    .mul(100)
    .unstack()
)

print(result.round(2))

# --------------------------------
# 2. ACCURACY BY FAULT AND SPEED
# --------------------------------

print("\n===== ACCURACY BY FAULT AND SPEED =====")

result = (
    df.groupby(["condition", "speed"])["correct"]
    .mean()
    .mul(100)
    .unstack()
)

print(result.round(2))

# --------------------------------
# 3. ACCURACY BY FAULT AND LOAD
# --------------------------------

print("\n===== ACCURACY BY FAULT AND LOAD =====")

result = (
    df.groupby(["condition", "load"])["correct"]
    .mean()
    .mul(100)
    .unstack()
)

print(result.round(2))

# --------------------------------
# 4. SAVE RESULTS
# --------------------------------

print("\nAnalysis completed!")
