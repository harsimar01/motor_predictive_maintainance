import pandas as pd
import numpy as np
from scipy.stats import kurtosis
from scipy.fft import fft

df = pd.read_csv("data/raw_motor_sensor_data.csv")

features = []

grouped = df.groupby("sample_id")

for sample_id, group in grouped:

    group = group.sort_index()

    vibration = group["vibration"].values
    current = group["current"].values
    rpm = group["rpm"].values
    temperature = group["temperature"].mean()

    fault_type = group["fault_type"].iloc[0]

    # -------- Vibration Features --------

    vibration_rms = np.sqrt(np.mean(vibration**2))

    vibration_kurtosis = kurtosis(vibration)

    vibration_fft = np.abs(fft(vibration))[:len(vibration)//2]

    vibration_spectral_energy = np.sum(vibration_fft**2)

    freqs = np.fft.fftfreq(len(vibration))[:len(vibration)//2]

    dominant_frequency = freqs[np.argmax(vibration_fft)]

    # -------- Current Features --------

    current_rms = np.sqrt(np.mean(current**2))

    current_kurtosis = kurtosis(current)

    current_fft = np.abs(fft(current))[:len(current)//2]

    current_spectral_energy = np.sum(current_fft**2)

    # -------- RPM Feature --------

    rpm_std = np.std(rpm)

    features.append([
        sample_id,
        vibration_rms,
        vibration_kurtosis,
        vibration_spectral_energy,
        dominant_frequency,
        current_rms,
        current_kurtosis,
        current_spectral_energy,
        rpm_std,
        temperature,
        fault_type
    ])

features_df = pd.DataFrame(features, columns=[
    "sample_id",
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_spectral_energy",
    "dominant_frequency",
    "current_rms",
    "current_kurtosis",
    "current_spectral_energy",
    "rpm_std",
    "temperature",
    "fault_type"
])

features_df.to_csv("data/motor_multisensor_features.csv", index=False)

print("Feature dataset created")
print(features_df.head())