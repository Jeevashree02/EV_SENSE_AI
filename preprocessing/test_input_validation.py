
import numpy as np

from motor_inference import extract_features


valid_window = np.random.normal(0, 0.1, (200, 3))

# Test 1: Valid input
extract_features(valid_window)
print("PASS: Valid input accepted")

# Test 2: Incorrect shape
try:
    extract_features(np.random.rand(100, 3))
except ValueError:
    print("PASS: Incorrect shape rejected")

# Test 3: NaN values
bad_window = valid_window.copy()
bad_window[0, 0] = np.nan

try:
    extract_features(bad_window)
except ValueError:
    print("PASS: NaN input rejected")

# Test 4: Infinite values
bad_window = valid_window.copy()
bad_window[0, 0] = np.inf

try:
    extract_features(bad_window)
except ValueError:
    print("PASS: Infinite input rejected")

# Test 5: Incorrect sampling frequency
try:
    extract_features(valid_window, fs=50)
except ValueError:
    print("PASS: Incorrect sampling frequency rejected")
