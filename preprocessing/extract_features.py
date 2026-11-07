
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

# ==========================================
# 1. LOAD SEGMENTED DATA
# ==========================================

data = np.load("../dataset/segmented_windows.npz")

X = data["X"]
y = data["y"]
metadata = data["metadata"]

SAMPLE_RATE = 100

features = []

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================

for window in X:

    row = []

    # Process all three acceleration axes
    for axis in range(3):

        signal = window[:, axis]

        # ----------------------------------
        # TIME-DOMAIN FEATURES
        # ----------------------------------

        rms = np.sqrt(np.mean(signal ** 2))

        std = np.std(signal)

        peak_to_peak = np.ptp(signal)

        kurt = kurtosis(signal)

        # Crest factor
        peak_amplitude = np.max(np.abs(signal))

        crest_factor = (
            peak_amplitude / rms if rms > 0 else 0
        )

        # ----------------------------------
        # FREQUENCY-DOMAIN FEATURES
        # ----------------------------------

        spectrum = np.abs(np.fft.rfft(signal))

        frequencies = np.fft.rfftfreq(
            len(signal),
            d=1 / SAMPLE_RATE
        )

        # Exclude zero-frequency component
        spectrum_no_dc = spectrum[1:]

        frequencies_no_dc = frequencies[1:]

        # Dominant frequency
        dominant_index = np.argmax(spectrum_no_dc)

        dominant_frequency = frequencies_no_dc[
            dominant_index
        ]

        dominant_amplitude = spectrum_no_dc[
            dominant_index
        ]

        # ----------------------------------
        # SPECTRAL CENTROID
        # ----------------------------------

        spectral_sum = np.sum(spectrum_no_dc)

        if spectral_sum > 0:

            spectral_centroid = np.sum(
                frequencies_no_dc * spectrum_no_dc
            ) / spectral_sum

        else:
            spectral_centroid = 0

        # ----------------------------------
        # CORRECTED BAND-ENERGY RATIOS
        # ----------------------------------

        # Power spectrum excluding DC
        valid = frequencies > 0

        freq_valid = frequencies[valid]

        power_valid = spectrum[valid] ** 2

        # Total energy excluding DC
        total_energy = np.sum(power_valid)

        # Low frequency: 0-10 Hz
        low_energy = np.sum(
            power_valid[freq_valid < 10]
        )

        # Mid frequency: 10-25 Hz
        mid_energy = np.sum(
            power_valid[
                (freq_valid >= 10) &
                (freq_valid < 25)
            ]
        )

        # High frequency: 25-50 Hz
        high_energy = np.sum(
            power_valid[
                (freq_valid >= 25) &
                (freq_valid <= 50)
            ]
        )

        # Normalize energy
        if total_energy > 0:

            low_ratio = low_energy / total_energy

            mid_ratio = mid_energy / total_energy

            high_ratio = high_energy / total_energy

        else:

            low_ratio = 0
            mid_ratio = 0
            high_ratio = 0

        # ----------------------------------
        # STORE FEATURES
        # ----------------------------------

        row.extend([
            rms,
            std,
            peak_to_peak,
            kurt,
            dominant_frequency,
            spectral_centroid,
            dominant_amplitude,
            crest_factor,
            low_ratio,
            mid_ratio,
            high_ratio
        ])

    features.append(row)

# ==========================================
# 3. FEATURE NAMES
# ==========================================

feature_names = []

for axis in ["X", "Y", "Z"]:

    feature_names.extend([
        f"{axis}_RMS",
        f"{axis}_STD",
        f"{axis}_PeakToPeak",
        f"{axis}_Kurtosis",
        f"{axis}_DominantFrequency",
        f"{axis}_SpectralCentroid",
        f"{axis}_DominantAmplitude",
        f"{axis}_CrestFactor",
        f"{axis}_LowBandRatio",
        f"{axis}_MidBandRatio",
        f"{axis}_HighBandRatio"
    ])

# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

feature_df = pd.DataFrame(
    features,
    columns=feature_names
)

# Add metadata
feature_df["condition"] = y

feature_df["source_file"] = metadata[:, 0]

feature_df["load"] = metadata[:, 1].astype(int)

feature_df["speed"] = metadata[:, 2].astype(int)

# ==========================================
# 5. VERIFY BAND-ENERGY RATIOS
# ==========================================

print("\n========== BAND RATIO VALIDATION ==========")

for axis in ["X", "Y", "Z"]:

    ratio_sum = (
        feature_df[f"{axis}_LowBandRatio"] +
        feature_df[f"{axis}_MidBandRatio"] +
        feature_df[f"{axis}_HighBandRatio"]
    )

    print(f"\n{axis} axis:")
    print("Minimum ratio sum:", ratio_sum.min())
    print("Maximum ratio sum:", ratio_sum.max())

    print(
        "All ratios sum to approximately 1:",
        np.allclose(ratio_sum, 1.0, atol=1e-6)
    )

# ==========================================
# 6. SAVE FEATURE DATASET
# ==========================================

feature_df.to_csv(
    "../dataset/motor_features.csv",
    index=False
)

# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("\n========== FEATURE EXTRACTION RESULTS ==========")

print("Total windows:", len(feature_df))

print("Total features:", len(feature_names))

print("Feature shape:", feature_df.shape)

print("\nFirst five rows:")

print(feature_df.head())

print(
    "\nMissing values:",
    feature_df.isnull().sum().sum()
)

print("\nSaved: dataset/motor_features.csv")

print("\nFEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
