import schedule
import time
import subprocess

def run_prediction():
    subprocess.run(["python", "/app/predictor.py"])  # Changed from scripts path

schedule.every().day.at("07:30").do(run_prediction)
schedule.every().day.at("18:20").do(run_prediction)

while True:
    schedule.run_pending()
    time.sleep(1)
