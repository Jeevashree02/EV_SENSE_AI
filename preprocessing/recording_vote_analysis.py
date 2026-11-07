
import pandas as pd

df = pd.read_csv("../dataset/cv_predictions.csv")

# Calculate percentage of windows assigned to each class
vote_table = pd.crosstab(
    df["source_file"],
    df["predicted"],
    normalize="index"
) * 100

# Add actual condition
actual = df.groupby("source_file")["condition"].first()

vote_table["actual"] = actual

# Find winning prediction and its vote percentage
class_columns = [
    col for col in vote_table.columns if col != "actual"
]

vote_table["predicted"] = (
    vote_table[class_columns].idxmax(axis=1)
)

vote_table["winning_vote_percent"] = (
    vote_table[class_columns].max(axis=1)
)

vote_table["correct"] = (
    vote_table["actual"] == vote_table["predicted"]
)

print("\n========== RECORDING VOTE ANALYSIS ==========")

print(
    vote_table[
        ["actual", "predicted", "winning_vote_percent", "correct"]
    ].round(2).to_string()
)

vote_table.to_csv(
    "../dataset/recording_vote_analysis.csv"
)

print("\nVote analysis saved!")
