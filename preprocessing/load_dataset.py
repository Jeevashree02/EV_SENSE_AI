
import pandas as pd
from pathlib import Path

# Change this path if your dataset is stored elsewhere
DATASET_PATdirH = Path("../dataset/raw")

# Sensor column names
COLUMNS = ["gX", "gY", "gZ", "gUserX", "gUserY", "gUserZ"]

all_data = []

# Read every CSV inside all six folders
for folder in DATASET_PATH.iterdir():
    if folder.is_dir():
        for file in folder.glob("*.csv"):
            try:
                df = pd.read_csv(file, sep=";")
                df.columns = df.columns.str.strip()

                if not set(COLUMNS).issubset(df.columns):
                    print(f"Skipped: {file.name} - unexpected columns")
                    continue

                df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
                df["condition"] = folder.name
                df["source_file"] = file.name

                all_data.append(df)
                print(f"Loaded: {file.name}")

            except Exception as error:
                print(f"Error in {file.name}: {error}")

if all_data:
    combined = pd.concat(all_data, ignore_index=True)

    output_path = Path("../dataset/combined_motor_data.csv")
    combined.to_csv(output_path, index=False)

    print("\nDataset loading complete!")
    print("Total rows:", len(combined))
    print("Missing values:\n", combined.isna().sum())
    print("Saved to:", output_path)
else:
    print("No valid CSV files found. Check the dataset path.")
