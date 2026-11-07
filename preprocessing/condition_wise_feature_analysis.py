
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    "X_RMS",
    "Y_PeakToPeak",
    "Z_PeakToPeak"
]

conditions = [
    (30, 0), (30, 1),
    (40, 0), (40, 1),
    (50, 0), (50, 1)
]

for speed, load in conditions:

    subset = df[
        (df["speed"] == speed) &
        (df["load"] == load)
    ]

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=subset,
            x="condition",
            y=feature,
            hue="condition",
            legend=False
        )

        plt.title(
            f"{feature} | Speed: {speed} Hz | Load: {load}"
        )

        plt.xlabel("Fault Class")
        plt.ylabel(feature)

        plt.tight_layout()

        filename = (
            f"../dataset/{feature}_{speed}Hz_load{load}.png"
        )

        plt.savefig(filename)
        plt.show()

print("\nAll condition-wise feature plots saved!")
