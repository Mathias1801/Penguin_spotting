import json
import joblib
import requests
import datetime
import os
import sqlite3
from datetime import datetime as dt

# Load model and label encoder
clf = joblib.load("models/penguin_classifier_model.pkl")
label_encoder = joblib.load("models/penguin_label_encoder.pkl")

# Fetch data from API
url = "http://130.225.39.127:8000/new_penguin/"
response = requests.get(url)
data = response.json()

# Prepare features and predict species
features = [[
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"]
]]

species_encoded = clf.predict(features)[0]
species = label_encoder.inverse_transform([species_encoded])[0]

# Prepare the prediction result
timestamp = datetime.datetime.utcnow().isoformat()

prediction_result = {
    "timestamp": timestamp,
    "bill_length_mm": data["bill_length_mm"],
    "bill_depth_mm": data["bill_depth_mm"],
    "flipper_length_mm": data["flipper_length_mm"],
    "body_mass_g": data["body_mass_g"],
    "predicted_species": species
}

# ✅ Overwrite JSON file with latest prediction
with open("data/predictions.json", "w") as f:
    json.dump(prediction_result, f)

# ✅ Append to SQLite DB (grow over time)
db_path = "data/predictions.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        bill_length_mm REAL,
        bill_depth_mm REAL,
        flipper_length_mm REAL,
        body_mass_g REAL,
        predicted_species TEXT
    )
""")

# Insert today's prediction
cursor.execute("""
    INSERT INTO predictions (
        timestamp,
        bill_length_mm,
        bill_depth_mm,
        flipper_length_mm,
        body_mass_g,
        predicted_species
    ) VALUES (?, ?, ?, ?, ?, ?)
""", (
    timestamp,
    data["bill_length_mm"],
    data["bill_depth_mm"],
    data["flipper_length_mm"],
    data["body_mass_g"],
    species
))

conn.commit()
conn.close()

print("✅ JSON overwritten and prediction saved to SQLite.")
