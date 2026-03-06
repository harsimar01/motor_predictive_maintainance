import numpy as np
import pandas as pd

np.random.seed(42)

samples = 2000
signal_length = 256

rows = []

for sample_id in range(samples):

    t = np.linspace(0,1,signal_length)

    fault_type = "normal"

    r = np.random.rand()

    # Base signals
    vibration = 0.5*np.sin(2*np.pi*50*t) + 0.05*np.random.randn(signal_length)
    current = 5 + 0.2*np.random.randn(signal_length)
    rpm = 1500 + 5*np.random.randn(signal_length)

    # Temperature time series
    temperature = 60 + np.random.normal(0,2,signal_length)

    #  Fault Injection 

    if r < 0.08:
        fault_type = "imbalance"

        vibration += 1.0*np.sin(2*np.pi*50*t)
        current += 1.2*np.sin(2*np.pi*50*t)

        temperature += np.random.uniform(5,10)

    elif r < 0.14:
        fault_type = "misalignment"

        vibration += 0.8*np.sin(2*np.pi*100*t)
        vibration += 0.5*np.sin(2*np.pi*150*t)

        current += 0.5*np.sin(2*np.pi*100*t)

        rpm += np.random.uniform(-20,20)

        temperature += np.random.uniform(8,15)

    elif r < 0.20:
        fault_type = "bearing_fault"

        impulses = np.zeros(signal_length)

        impulse_positions = np.random.choice(signal_length,6)

        impulses[impulse_positions] = np.random.uniform(2,4,6)

        vibration += impulses

        current += 0.3*np.random.randn(signal_length)

        temperature += np.random.uniform(10,20)

    # Save Each Time Step 

    for i in range(signal_length):

        rows.append([
            sample_id,
            vibration[i],
            current[i],
            rpm[i],
            temperature[i],
            fault_type
        ])

df = pd.DataFrame(rows, columns=[
    "sample_id",
    "vibration",
    "current",
    "rpm",
    "temperature",
    "fault_type"
])

df.to_csv("data/raw_motor_sensor_data.csv", index=False)

print("Raw sensor dataset generated")
print(df.head())
