
import pandas as pd
import numpy as np

df = pd.read_csv("../dataset/motor_features.csv")

exclude = [
    "condition",
    "source_file",
    "speed",
    "load"
]

features = [
    col for col in df.columns
    if col not in exclude
]

corr = df[features].corr().abs()

# Keep only the upper triangle to avoid duplicate pairs
upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

# Find pairs with correlation greater than 0.90
high_corr = (
    upper.stack()
    .reset_index()
)

high_corr.columns = [
    "Feature_1",
    "Feature_2",
    "Correlation"
]

high_corr = high_corr[
    high_corr["Correlation"] > 0.90
].sort_values(
    by="Correlation",
    ascending=False
)

print("\n========== HIGHLY CORRELATED FEATURES ==========")

print(high_corr.to_string(index=False))

high_corr.to_csv(
    "../dataset/high_correlation_pairs.csv",
    index=False
)

print("\nCorrelation analysis completed!")
