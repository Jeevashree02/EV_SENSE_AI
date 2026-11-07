
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("../dataset/motor_features.csv")

features = [
    c for c in df.columns
    if c not in ["condition", "source_file", "load", "speed"]
]

X = df[features]
y = df["condition"]
groups = df["source_file"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(cv.split(X, y, groups))

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X.iloc[train_idx], y.iloc[train_idx])

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")
print(importance.head(15).to_string(index=False))

plt.figure(figsize=(10, 7))

top_features = importance.head(15).sort_values(
    "Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.title("Top 15 Random Forest Features")
plt.tight_layout()

plt.savefig("../dataset/feature_importance.png")
plt.show()

print("\nFeature importance saved!")
