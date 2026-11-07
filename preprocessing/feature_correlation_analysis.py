
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

correlation = df[features].corr()

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Vibration Feature Correlation")
plt.tight_layout()

plt.savefig(
    "../dataset/feature_correlation.png",
    dpi=200
)

plt.show()

print("\nCorrelation analysis completed!")
