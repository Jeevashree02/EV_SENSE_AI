
import pandas as pd

# Load saved out-of-fold predictions
data = pd.read_csv("../dataset/cv_predictions.csv")

print("\n========== OOF DATASET INSPECTION ==========")

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst 10 rows:")
print(data.head(10).to_string(index=False))

print("\nMissing values:")
print(data.isnull().sum())

print("\nNumber of unique recordings:")

if "source_file" in data.columns:
    print(data["source_file"].nunique())
else:
    print("source_file column not found!")
