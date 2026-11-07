
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis


# -----------------------------
# Load the trained model
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "final_random_forest.pkl"

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
FEATURES = saved_data["features"]
CLASSES = saved_data["classes"]

EXPECTED_FS = 100
EXPECTED_WINDOW_SIZE = 200
EXPECTED_AXES = 3


# -----------------------------
# Extract features from one axis
# -----------------------------

def extract_axis_features(signal, fs=EXPECTED_FS):
    signal = np.asarray(signal, dtype=float)

    rms = np.sqrt(np.mean(signal ** 2))
    std = np.std(signal)
    peak_to_peak = np.ptp(signal)
    kurt = kurtosis(signal)

    frequencies = rfftfreq(len(signal), d=1 / fs)
    amplitude = np.abs(rfft(signal))

    valid = frequencies > 0

    if np.any(valid):
        valid_amplitude = amplitude[valid]
        valid_frequencies = frequencies[valid]

        dominant_index = np.argmax(valid_amplitude)

        dominant_frequency = valid_frequencies[dominant_index]
        dominant_amplitude = valid_amplitude[dominant_index]

        total_amplitude = np.sum(valid_amplitude)

        if total_amplitude > 0:
            spectral_centroid = (
                np.sum(valid_frequencies * valid_amplitude)
                / total_amplitude
            )
        else:
            spectral_centroid = 0.0
    else:
        dominant_frequency = 0.0
        dominant_amplitude = 0.0
        spectral_centroid = 0.0

    crest_factor = (
        np.max(np.abs(signal)) / rms if rms > 0 else 0.0
    )

    power = amplitude ** 2
    total_energy = np.sum(power[valid])

    low_energy = np.sum(
        power[(frequencies > 0) & (frequencies < 10)]
    )
    mid_energy = np.sum(
        power[(frequencies >= 10) & (frequencies < 25)]
    )
    high_energy = np.sum(
        power[(frequencies >= 25) & (frequencies <= 50)]
    )

    if total_energy > 0:
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
    else:
        low_ratio = mid_ratio = high_ratio = 0.0

    return [
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
        high_ratio,
    ]


# -----------------------------
# Extract all 33 features
# -----------------------------

def extract_features(window, fs=EXPECTED_FS):

    window = np.asarray(window, dtype=float)

    if window.shape != (EXPECTED_WINDOW_SIZE, EXPECTED_AXES):
        raise ValueError(
            f"Expected window shape (200, 3), got {window.shape}"
        )

    if not np.all(np.isfinite(window)):
        raise ValueError("Sensor window contains NaN or infinite values.")

    if fs != EXPECTED_FS:
        raise ValueError(
            f"Expected sampling frequency {EXPECTED_FS} Hz, got {fs} Hz."
        )

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
            f"{axis}_HighBandRatio",
        ])

    values = []

    for axis_index in range(EXPECTED_AXES):
        values.extend(
            extract_axis_features(window[:, axis_index], fs)
        )

    feature_df = pd.DataFrame([values], columns=feature_names)

    return feature_df[FEATURES]


# -----------------------------
# Predict motor condition
# -----------------------------

def predict_motor_condition(window, fs=EXPECTED_FS):

    feature_df = extract_features(window, fs)

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]

    probability_dict = {
        str(label): float(probability)
        for label, probability in zip(CLASSES, probabilities)
    }

    return prediction, probability_dict


# -----------------------------
# Basic test
# -----------------------------

if __name__ == "__main__":

    test_window = np.random.normal(0, 0.1, (200, 3))

    prediction, probabilities = predict_motor_condition(test_window)

    print("Predicted condition:", prediction)
    print("Model probability scores:")

    for label, score in probabilities.items():
        print(f"{label}: {score * 100:.2f}%")
