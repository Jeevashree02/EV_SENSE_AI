
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    "X_RMS",
    "Y_PeakToPeak",
    "Z_PeakToPeak"
]

# Visualize feature distributions for all fault classes
for feature in features:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="condition",
        y=feature,
        hue="condition",
        legend=False
    )

    plt.title(f"{feature} - Fault Class Comparison")
    plt.xlabel("Fault Condition")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        f"../dataset/{feature}_separation.png"
    )

    plt.show()

print("\nFeature separation plots saved!")
