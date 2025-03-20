# 🐧 MLOps Assignment 1 - Penguin Classifier 🐧

This repository contains my submission for the MLOps Assignment 1, where the goal is to build a machine learning pipeline for classifying penguin species — with a focus on daily webscraping!
This project is an end-to-end machine learning pipeline and MLOps workflow for classifying penguin species based on given features. It includes model training, prediction scripts, CI/CD setup using GitHub Actions, and a playground for testing and colculating models.

---

## 📦 Project Structure

```
.
├── predictor.py                 # The script that implements the trained model and predicts species
├── playground.ipynb            # Jupyter Notebook where fetures are selected and models are downloaded
├── requirements.txt            # list of dependencies 
├── index.html                  # HTML Report
├── data/
│   └── prediction.json         # Sample input data for predictions
├── models/
│   ├── penguin_classifier_model.pkl    # Trained classifier
│   ├── penguin_label_encoder.pkl       # Label encoder
│   ├── penguin_scaler.pkl              # Scaler used for feature normalization
└── .github/
    └── workflows/
        └── run_predictions.yml # GitHub Actions workflow for automated predictions
```

---

## How to use the repo

### 1. Clone the repository

```bash
git clone <repo-url>
cd Mathias1801-MLOps_Assignment_1
```

### 2. Create a virtual environment

#### On macOS/Linux
```bash
python -m venv venv
source venv/bin/activate
```

#### On Windows
```bash
python -m venv venv
venv\Scripts\activate
```
```
### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Predicting species

You can make predictions by running the `predictor.py` script. The script is deendant on the placement of files in specific folders, so keep that in mind. Predictor directly uses classifier, scaler and encoder in the models folder, and it also uses predictions.json for features that needs to be predicted:

```bash
python predictor.py
```

Make sure `data/prediction.json` exists and contains the input data in the correct format.

Alternatively, explore and manipulate predictions in `playground.ipynb`.

---

## CI/CD

This project includes a workflow file at `.github/workflows/run_predictions.yml` to automate the prediction process when new data is pushed or changes occur. This is done through Github Actions and it is made with the goal of scraping data everyday at 07:30 am (UTC/GMT +1 hour).
