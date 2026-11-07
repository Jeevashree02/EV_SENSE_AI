
import pandas as pd
import numpy as np
from pathlib import Path

# Dataset location
DATA_PATH = Path("../dataset/raw")

# Sampling and window settings
SAMPLE_RATE = 100
WINDOW_SIZE = 2 * SAMPLE_RATE  # 200 samples
HOP_SIZE = WINDOW_SIZE // 2    # 50% overlap

# Use gravity-compensated acceleration
CHANNELS = ["gUserX", "gUserY", "gUserZ"]

all_windows = []
all_labels = []
all_metadata = []

# Read every recording
for folder in sorted(DATA_PATH.iterdir()):

    if not folder.is_dir():
        continue

    condition = folder.name

    for file in sorted(folder.glob("*.csv")):

        df = pd.read_csv(file, sep=";")
        df.columns = df.columns.str.strip()

        # Check required columns
        if not set(CHANNELS).issubset(df.columns):
            print("Skipped:", file.name)
            continue

        signal = df[CHANNELS].apply(
            pd.to_numeric, errors="coerce"
        ).dropna().to_numpy(dtype=np.float32)

        # Extract load and speed from filename
        parts = file.stem.split("_")
        load = int(parts[1])
        speed = int(parts[2].replace("Hz", ""))

        # Divide signal into overlapping windows
        count = 0

        for start in range(0, len(signal) - WINDOW_SIZE + 1, HOP_SIZE):

            window = signal[start:start + WINDOW_SIZE]

            all_windows.append(window)
            all_labels.append(condition)

            all_metadata.append([
                file.name, load, speed, start
            ])

            count += 1

        print(f"{file.name}: {count} windows")

# Convert to NumPy arrays
X = np.stack(all_windows)
y = np.array(all_labels)
metadata = np.array(all_metadata, dtype=str)

# Save segmented dataset
output_path = Path("../dataset/segmented_windows.npz")

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    metadata=metadata,
    channels=np.array(CHANNELS)
)

print("\n========== SEGMENTATION RESULTS ==========")
print("Window data shape:", X.shape)
print("Total windows:", len(X))
print("Window size:", WINDOW_SIZE)
print("Hop size:", HOP_SIZE)
print("Classes:", np.unique(y))
print("Saved to:", output_path)
