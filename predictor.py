import json
import csv
import os
import datetime
import joblib
import requests

# Load model and label encoder
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")

# Get data from API
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Prepare features and make prediction
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

species_encoded = clf.predict(features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Build prediction result
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

# 🔹 1. TEMP: Save latest prediction to JSON (overwrite each time)
with open("data/latest_prediction.json", "w") as f:
    json.dump(prediction_result, f, indent=4)

# 🔹 2. LOG: Append prediction to CSV file
csv_file = "data/predictions.csv"
file_exists = os.path.exists(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=prediction_result.keys())
    if not file_exists:
        writer.writeheader()
    writer.writerow(prediction_result)

print(f"Prediction saved to JSON and logged to CSV:\n{prediction_result}")
