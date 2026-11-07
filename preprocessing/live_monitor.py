
import time

import numpy as np

from motor_inference import predict_motor_condition


FS = 100
WINDOW_SIZE = 200


def simulate_sensor_data():
    """
    Simulate 3-axis vibration readings.
    Replace this function with ESP32 serial input later.
    """
    t = np.arange(WINDOW_SIZE) / FS

    x = 0.10 * np.sin(2 * np.pi * 10 * t)
    y = 0.08 * np.sin(2 * np.pi * 15 * t)
    z = 0.06 * np.sin(2 * np.pi * 20 * t)

    noise = np.random.normal(0, 0.01, (WINDOW_SIZE, 3))

    return np.column_stack((x, y, z)) + noise


def main():

    print("EV-Sense AI - Live Motor Monitor")
    print("--------------------------------")

    for _ in range(3):

        print("\nCollecting vibration window...")

        window = simulate_sensor_data()

        prediction, probabilities = predict_motor_condition(
            window,
            fs=FS
        )

        print("\nPredicted motor condition:", prediction)
        print("\nModel vote scores:")

        for label, score in probabilities.items():
            print(f"{label}: {score * 100:.2f}%")

        print("\nWaiting for next window...")
        time.sleep(1)


if __name__ == "__main__":
    main()
