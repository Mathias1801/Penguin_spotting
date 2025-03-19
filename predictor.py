import json
import joblib
import requests
import datetime
import sqlite3
from datetime import datetime as dt

# Load our model
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")

# API endpoint where we fetch data
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Extract features
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

# Predict species
species_encoded = clf.predict(features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Save to JSON file
prediction_result = {
    "timestamp": datetime.datetime.utcnow().isoformat(),
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

with open("predictions.json", "a") as f:
    json.dump(prediction_result, f)
    f.write("\n")

# === SQLite Logging ===
conn = sqlite3.connect("data/predictions.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        features TEXT,
        prediction TEXT
    )
""")

cursor.execute(
    "INSERT INTO predictions (timestamp, features, prediction) VALUES (?, ?, ?)",
    (datetime.datetime.now().isoformat(), json.dumps(features), json.dumps(species))
)

conn.commit()
conn.close()
