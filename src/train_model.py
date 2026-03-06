import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load feature dataset
df = pd.read_csv("data/motor_multisensor_features.csv")

# Features and target
X = df.drop(["sample_id", "fault_type"], axis=1)
y = df["fault_type"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ---------- SAVE MODEL ----------

joblib.dump(model, "models/predictive_maintenance_model.pkl")

joblib.dump(encoder, "models/label_encoder.pkl")

print("\nModel saved to models/predictive_maintenance_model.pkl")