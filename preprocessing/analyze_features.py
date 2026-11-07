
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load extracted features
df = pd.read_csv("../dataset/motor_features.csv")

# Select the 15 numerical features
feature_columns = [
    col for col in df.columns
    if col not in ["condition", "source_file", "load", "speed"]
]

# 1. Compare feature medians across conditions
print("\n========== FEATURE MEDIANS BY CONDITION ==========")

medians = df.groupby("condition")[feature_columns].median()

print(medians.round(4))

# Save statistics
medians.to_csv("../dataset/feature_medians.csv")

# 2. Plot feature distributions
fig, axes = plt.subplots(3, 5, figsize=(19, 12))
axes = axes.flatten()

for i, feature in enumerate(feature_columns):

    groups = [
        df.loc[df["condition"] == condition, feature]
        for condition in sorted(df["condition"].unique())
    ]

    axes[i].boxplot(groups, showfliers=False)

    axes[i].set_title(feature)
    axes[i].set_xticks(range(1, 7))
    axes[i].set_xticklabels(
        sorted(df["condition"].unique()),
        rotation=45
    )
    axes[i].grid(alpha=0.3)

plt.suptitle("Feature Distribution Across Motor Conditions")
plt.tight_layout()

plt.savefig("../dataset/feature_boxplots.png", dpi=150)
plt.show()

# 3. Correlation analysis
correlation = df[feature_columns].corr()

print("\n========== FEATURE CORRELATION ==========")
print(correlation.round(2))

plt.figure(figsize=(12, 9))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(feature_columns)),
    feature_columns,
    rotation=90
)

plt.yticks(range(len(feature_columns)), feature_columns)

plt.title("Feature Correlation Matrix")
plt.tight_layout()

plt.savefig("../dataset/feature_correlation.png", dpi=150)
plt.show()

print("\nFEATURE ANALYSIS COMPLETED!")
print("Saved feature medians and graphs in dataset folder.")
