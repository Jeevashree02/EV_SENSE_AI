
import numpy as np
import pandas as pd
import joblib

from scipy.stats import kurtosis
from scipy.fft import rfft, rfftfreq


# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

print("Loading trained Random Forest model...")

saved = joblib.load("../model/final_random_forest.pkl")

model = saved["model"]
features = saved["features"]
classes = saved["classes"]

print("Model loaded successfully!")
print("Expected features:", len(features))


# =====================================
# 2. FEATURE EXTRACTION FOR ONE AXIS
# =====================================

def extract_axis_features(signal, axis, fs=100):

    signal = np.asarray(signal, dtype=float)

    # TIME-DOMAIN FEATURES

    rms = np.sqrt(np.mean(signal ** 2))

    std = np.std(signal)

    peak_to_peak = np.ptp(signal)

    # Match original training calculation
    kurt = kurtosis(signal)

    if not np.isfinite(kurt):
        kurt = 0.0

    peak_amplitude = np.max(np.abs(signal))

    if rms > 0:
        crest_factor = peak_amplitude / rms
    else:
        crest_factor = 0.0

    # FREQUENCY-DOMAIN FEATURES

    spectrum = np.abs(rfft(signal))

    frequencies = rfftfreq(
        len(signal),
        d=1 / fs
    )

    # Exclude DC component
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

    # SPECTRAL CENTROID

    spectral_sum = np.sum(spectrum_no_dc)

    if spectral_sum > 0:

        spectral_centroid = np.sum(
            frequencies_no_dc * spectrum_no_dc
        ) / spectral_sum

    else:
        spectral_centroid = 0.0

    # BAND-ENERGY RATIOS

    valid = frequencies > 0

    freq_valid = frequencies[valid]

    power_valid = spectrum[valid] ** 2

    total_energy = np.sum(power_valid)

    low_energy = np.sum(
        power_valid[freq_valid < 10]
    )

    mid_energy = np.sum(
        power_valid[
            (freq_valid >= 10) &
            (freq_valid < 25)
        ]
    )

    high_energy = np.sum(
        power_valid[
            (freq_valid >= 25) &
            (freq_valid <= 50)
        ]
    )

    if total_energy > 0:

        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy

    else:

        low_ratio = 0.0
        mid_ratio = 0.0
        high_ratio = 0.0

    return {

        f"{axis}_RMS": rms,
        f"{axis}_STD": std,
        f"{axis}_PeakToPeak": peak_to_peak,
        f"{axis}_Kurtosis": kurt,
        f"{axis}_DominantFrequency": dominant_frequency,
        f"{axis}_SpectralCentroid": spectral_centroid,
        f"{axis}_DominantAmplitude": dominant_amplitude,
        f"{axis}_CrestFactor": crest_factor,
        f"{axis}_LowBandRatio": low_ratio,
        f"{axis}_MidBandRatio": mid_ratio,
        f"{axis}_HighBandRatio": high_ratio
    }


# =====================================
# 3. EXTRACT ALL 33 FEATURES
# =====================================

def extract_features(window):

    window = np.asarray(window, dtype=float)

    if window.shape != (200, 3):

        raise ValueError(
            "Expected input shape (200, 3), "
            f"received {window.shape}"
        )

    if not np.isfinite(window).all():

        raise ValueError(
            "Input contains NaN or infinite values."
        )

    result = {}

    axes = ["X", "Y", "Z"]

    for i, axis in enumerate(axes):

        axis_features = extract_axis_features(
            window[:, i],
            axis
        )

        result.update(axis_features)

    feature_df = pd.DataFrame([result])

    # Preserve training feature order
    feature_df = feature_df[features]

    return feature_df


# =====================================
# 4. PREDICT MOTOR CONDITION
# =====================================

def predict_motor_condition(window):

    feature_df = extract_features(window)

    prediction = model.predict(feature_df)[0]

    probabilities = model.predict_proba(feature_df)[0]

    probability_dict = {

        str(label): float(probability)

        for label, probability in zip(
            model.classes_,
            probabilities
        )
    }

    return {

        "predicted_condition": str(prediction),

        "class_probabilities": probability_dict
    }


# =====================================
# 5. FEATURE CONSISTENCY TEST
# =====================================

def check_feature_consistency(window, original_row):

    backend_features = extract_features(window)

    original_values = original_row[features].astype(float)

    backend_values = backend_features.iloc[0]

    difference = (
        backend_values - original_values
    ).abs()

    comparison = pd.DataFrame({

        "Feature": features,

        "Original": original_values.values,

        "Backend": backend_values.values,

        "Absolute_Difference": difference.values
    })

    # Allow tiny floating-point differences
    tolerance = 1e-5

    max_difference = difference.max()

    mismatch_count = (
        difference > tolerance
    ).sum()

    print("\n========== FEATURE CONSISTENCY TEST ==========")

    print(comparison.to_string(index=False))

    print("\nMaximum feature difference:", max_difference)

    print("Tolerance:", tolerance)

    print(
        "Features with difference above tolerance:",
        mismatch_count
    )

    if mismatch_count == 0:

        print(
            "\nSUCCESS: All 33 features match within tolerance!"
        )

    else:

        print("\nWARNING: Some features do not match.")

    return comparison


# =====================================
# 6. TEST BACKEND
# =====================================

if __name__ == "__main__":

    print("\nTesting prediction backend...")

    # Load raw dataset
    raw = pd.read_csv(
        "../dataset/combined_motor_data.csv"
    )

    # Load original extracted features
    original_features = pd.read_csv(
        "../dataset/motor_features.csv"
    )

    sensor_columns = [
        "gUserX",
        "gUserY",
        "gUserZ"
    ]

    # Validate required columns
    for column in sensor_columns:

        if column not in raw.columns:

            raise ValueError(
                f"Missing sensor column: {column}"
            )

    if "source_file" not in raw.columns:

        raise ValueError(
            "source_file column missing from raw dataset."
        )

    if "source_file" not in original_features.columns:

        raise ValueError(
            "source_file column missing from feature dataset."
        )

    # Select first feature row
    original_row = original_features.iloc[0]

    # Find matching raw recording
    source_file = original_row["source_file"]

    recording = raw[
        raw["source_file"] == source_file
    ].reset_index(drop=True)

    if len(recording) < 200:

        raise ValueError(
            "Not enough samples in selected recording."
        )

    # Select first 200 samples
    test_window = recording.loc[
        :199,
        sensor_columns
    ].to_numpy(dtype=float)

    # Run feature consistency test
    check_feature_consistency(
        test_window,
        original_row
    )

    # Predict motor condition
    result = predict_motor_condition(test_window)

    print("\n========== PREDICTION RESULT ==========")

    print(
        "Predicted condition:",
        result["predicted_condition"]
    )

    print("\nClass probabilities:")

    for label, probability in (
        result["class_probabilities"].items()
    ):

        print(
            f"{label}: {probability * 100:.2f}%"
        )

    print("\nBackend test completed!")
