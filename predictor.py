import pandas as pd
import sqlite3
import joblib
import os
from datetime import datetime

#Load the model, scaler, and label encoder
model = joblib.load("model/penguin_classifier_model.pkl")
scaler = joblib.load("model/penguin_scaler.pkl")
label_encoder = joblib.load("model/penguin_label_encoder.pkl")

url = "https://raw.githubusercontent.com/<your-org-or-username>/<repo>/main/data.csv"
df = pd.read_csv(url)

features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
df = df[features]

df_scaled = scaler.transform(df)

# Predict
pred_encoded = model.predict(df_scaled)
df['predicted_species'] = label_encoder.inverse_transform(pred_encoded)
df['prediction_timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

# Save to SQLite Database 
os.makedirs("output", exist_ok=True)
db_path = "output/predictions.db"

conn = sqlite3.connect(db_path)
df.to_sql("predictions", conn, if_exists="append", index=False)
conn.close()
