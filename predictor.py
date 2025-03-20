import json
import joblib
import requests
import datetime

# Load our model, label encoder, and scaler
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")
scaler = joblib.load("models/penguin_scaler.pkl")

# API endpoint where we fetch data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract features
raw_features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

# Scale the features before prediction
scaled_features = scaler.transform(raw_features)

# Predict species
species_encoded = clf.predict(scaled_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Save the prediction as JSON so we can gather data over time
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

# Save predictions (append-friendly format)
with open("data/prediction.json", "a") as f:
    f.write(json.dumps(prediction_result, indent=4) + ",\n")

print(f"Prediction saved: {prediction_result}")
