# predictor.py

import pandas as pd
import sqlite3
import joblib
import os
import requests
from datetime import datetime

print("📦 Loading model, scaler, encoder...")
model = joblib.load("model/penguin_classifier_model.pkl")
scaler = joblib.load("model/penguin_scaler.pkl")
label_encoder = joblib.load("model/penguin_label_encoder.pkl")

# === Step 1: Download JSON from API
api_url = "http://130.225.39.127:8000/new_penguin/"
print(f"🌐 Fetching data from {api_url}")
response = requests.get(api_url)

if response.status_code != 200:
    raise Exception(f"❌ Failed to retrieve data: HTTP {response.status_code}")

data = response.json()

# Make sure this works even if the response is a single object (not a list yet)
if isinstance(data, dict):
    data = [data]

df = pd.DataFrame(data)

# === Step 2: Select Features
features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
df = df[features]

# === Step 3: Preprocess
df_scaled = scaler.transform(df)

# === Step 4: Predict
pred_encoded = model.predict(df_scaled)
df["predicted_species"] = label_encoder.inverse_transform(pred_encoded)
df["prediction_timestamp"] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# === Step 5: Add GIF logic
df['celebration'] = df['predicted_species'].apply(
    lambda x: '<img src="https://media.giphy.com/media/26BRrSvJUa0crqw4E/giphy.gif" width="60">' if x == "Adelie"
    else '<img src="https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif" width="60">'
)

# === Step 6: Save to SQLite
os.makedirs("output", exist_ok=True)
db_path = os.path.join("output", "predictions.db")
conn = sqlite3.connect(db_path)
df.to_sql("predictions", conn, if_exists="append", index=False)
conn.close()
print("✅ Saved to SQLite DB")

# === Step 7: Export to HTML
html_path = os.path.join("output", "predictions.html")
df.to_html(html_path, index=False, escape=False)

# === Step 8: Create index.html for GitHub Pages
with open("output/index.html", "w") as f:
    f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Penguin Predictions</title>
    <style>
        body {{ font-family: Arial; background: #f4f4f4; padding: 2em; }}
        h1 {{ text-align: center; color: #2c3e50; }}
        iframe {{ width: 100%; height: 600px; border: none; }}
    </style>
</head>
<body>
    <h1>🐧 Daily Penguin Predictions</h1>
    <iframe src="predictions.html"></iframe>
</body>
</html>
    """)
