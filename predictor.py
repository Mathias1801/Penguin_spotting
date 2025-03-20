import json
import joblib
import requests
import datetime

# Load trained model, label encoder, and scaler
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")
scaler = joblib.load("models/penguin_scaler.pkl")

# Define the correct feature order (must match training time)
feature_order = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Fetch new penguin data from API
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract and order features safely
raw_features = [[data[feature] for feature in feature_order]]

# Scale the features using the saved scaler
scaled_features = scaler.transform(raw_features)

# Predict species
species_encoded = clf.predict(scaled_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Save prediction with all details
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    **{feature: data[feature] for feature in feature_order},
    "predicted_species": species
}

# Append prediction to file
with open("data/prediction.json", "a") as f:
    f.write(json.dumps(prediction_result, indent=4) + ",\n")

print(f"✅ Prediction saved: {prediction_result}")
