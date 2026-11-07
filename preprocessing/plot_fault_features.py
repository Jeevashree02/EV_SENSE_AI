
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    "Y_PeakToPeak",
    "Z_PeakToPeak",
    "X_MidBandRatio",
    "X_PeakToPeak",
    "Y_STD",
    "Z_RMS",
    "Y_RMS",
    "X_HighBandRatio"
]

for feature in features:

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        data=df,
        x="condition",
        y=feature
    )

    plt.title(f"{feature} Across Motor Conditions")
    plt.xlabel("Motor Condition")
    plt.ylabel(feature)

    plt.tight_layout()

    plt.savefig(
        f"../dataset/{feature}_boxplot.png"
    )

    plt.show()

print("\nAll feature distribution plots generated!")
