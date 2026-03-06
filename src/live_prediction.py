import joblib
import pandas as pd

# Load model
model = joblib.load("models/predictive_maintenance_model.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Feature names used during training
columns = [
"vibration_rms",
"vibration_kurtosis",
"vibration_spectral_energy",
"dominant_frequency",
"current_rms",
"current_kurtosis",
"current_spectral_energy",
"rpm_std",
"temperature"
]

def predict_fault(features):

    X = pd.DataFrame([features], columns=columns)

    prediction = model.predict(X)

    fault = encoder.inverse_transform(prediction)

    return fault[0]


sample_features = [
0.35,
2.5,
4200,
50,
5.1,
3.0,
5000,
4.2,
61
]

result = predict_fault(sample_features)

print("Predicted Fault:", result)