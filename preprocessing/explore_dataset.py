
import pandas as pd

# Load the combined dataset
df = pd.read_csv("../dataset/combined_motor_data.csv")

# 1. Dataset dimensions
print("\n========== DATASET INFORMATION ==========")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Columns:", list(df.columns))

# 2. Number of readings in each condition
print("\n========== CONDITION DISTRIBUTION ==========")
print(df["condition"].value_counts())

# 3. Number of recordings in each condition
print("\n========== FILE DISTRIBUTION ==========")
print(
    df.groupby("condition")["source_file"]
    .nunique()
)

# 4. Statistical summary of sensor readings
sensor_columns = [
    "gX", "gY", "gZ",
    "gUserX", "gUserY", "gUserZ"
]

print("\n========== SENSOR STATISTICS ==========")
print(df[sensor_columns].describe())

# 5. Check duplicate rows
print("\n========== DUPLICATE CHECK ==========")
print("Duplicate rows:", df.duplicated().sum())

print("\nDATASET ANALYSIS COMPLETED!")
