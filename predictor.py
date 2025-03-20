import csv
import json
import joblib
import requests
import datetime
import os

# Load model, scaler, and label encoder
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")
scaler = joblib.load("models/penguin_scaler.pkl")

# Define the correct order of features (same as training)
feature_order = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Fetch penguin data from API
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract features in correct order
raw_features = [[data[feature] for feature in feature_order]]

# Scale the features
scaled_features = scaler.transform(raw_features)

# Predict species
species_encoded = clf.predict(scaled_features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Create result object
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    **{feature: data[feature] for feature in feature_order},
    "predicted_species": species
}

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Save to JSON
prediction_file_path = "data/prediction.json"
predictions = []

if os.path.exists(prediction_file_path):
    try:
        with open(prediction_file_path, "r") as f:
            predictions = json.load(f)
    except json.JSONDecodeError:
        print("⚠️ Existing prediction file is invalid or empty. Starting fresh.")

predictions.append(prediction_result)

with open(prediction_file_path, "w") as f:
    json.dump(predictions, f, indent=4)

# ✅ Save to CSV
def append_to_csv(prediction, csv_path="data/predictions.csv"):
    fieldnames = ['timestamp'] + feature_order + ['predicted_species']
    write_header = False

    # Check if header is already there
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        write_header = True

    try:
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({field: prediction.get(field, "") for field in fieldnames})
    except Exception as e:
        print(f"❌ Error writing to CSV: {e}")

# Append the result to CSV
append_to_csv(prediction_result)

print(f"✅ Prediction saved: {prediction_result}")
