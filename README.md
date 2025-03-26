# 🐧 MLOps Assignment 1 - Penguin Classifier 🐧

This repository contains my submission for the MLOps Assignment 1, where the goal is to build a machine learning pipeline for classifying penguin species — with a focus on daily webscraping!
This project is an end-to-end machine learning pipeline and MLOps workflow for classifying penguin species based on given features. It includes model training, prediction scripts, CI/CD setup using GitHub Actions, and a playground for testing and colculating models.

The projects is deployed in github pages: https://mathias1801.github.io/Penguin_spotting/ 

---

## Project Structure

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

## Feature Selection & Modeling Process

For this assignment, relevant feature selection was performed based on exploratory data analysis.  
The following features were selected due to their strong correlation with species classification:

- `bill_length_mm`
- `bill_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

These features were chosen because they represent key anatomical traits that vary between penguin species, as observed in the dataset.  

The data was then split into training and testing sets, and standard scaling was applied to numeric features. The prediction report indicated that all the trained model had 100% accuracy but in the playgroundfolder more detail-focused plots show how the different methods result in diffent approaches to feature selection. I chose to work with the Wrapper method focusing in recursive feature eliminationwhich was saved in the repository under `/models`.

---

## Note

Although the assignment specified using a SQL database, I encountered some challenges working with SQL in this context.  
As a workaround, I adapted the workflow to use a structured JSON format instead, which allowed for easier data handling and integration into the pipeline.

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

---

## NB
The prediction this repo deploys is wrong most of the time. The Seaborn dataset on penguins is very limit in the data availiable across seasons. The file called 'playground' creates a "perfect" prediction, but it is wrong, because the weight of the penguins vary dependant on season, but the data does not take this into account.
