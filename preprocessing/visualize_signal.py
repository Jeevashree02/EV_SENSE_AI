
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Dataset location
data_path = Path("../dataset/raw")

# Compare two recordings at the same speed/load setting
healthy_file = data_path / "H" / "H_0_30Hz.csv"
comparison_file = data_path / "B1" / "B1_0_30Hz.csv"

healthy = pd.read_csv(healthy_file, sep=";")
comparison = pd.read_csv(comparison_file, sep=";")

# Plot user acceleration signals
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

for column in ["gUserX", "gUserY", "gUserZ"]:
    axes[0].plot(healthy[column].iloc[:1000], label=column)

axes[0].set_title("H - First 1000 Samples")
axes[0].set_xlabel("Sample Number")
axes[0].set_ylabel("Acceleration")
axes[0].legend()
axes[0].grid(True)

for column in ["gUserX", "gUserY", "gUserZ"]:
    axes[1].plot(comparison[column].iloc[:1000], label=column)

axes[1].set_title("B1 - First 1000 Samples")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Acceleration")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
