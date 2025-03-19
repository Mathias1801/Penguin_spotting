import json
import joblib
import requests
import datetime
import os
import csv
from datetime import datetime as dt

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Load our model
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")

# Fetch new data from the API
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract features for prediction
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

# Predict species
species_encoded = clf.predict(features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Prepare prediction result
timestamp = datetime.datetime.utcnow().isoformat()

prediction_result = {
    "timestamp": timestamp,
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

# ✅ Append to JSON file
with open("data/predictions.json", "a") as f:
    json.dump(prediction_result, f)
    f.write("\n")

# ✅ Append to CSV file
csv_path = "data/predictions.csv"
write_header = not os.path.exists(csv_path)

with open(csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "timestamp",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "predicted_species"
        ])
    writer.writerow([
        timestamp,
        data["bill_length_mm"],
        data["bill_depth_mm"],
        data["flipper_length_mm"],
        data["body_mass_g"],
        species
    ])

print("✅ Prediction saved to both JSON and CSV.")
