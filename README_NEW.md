# 🐧 MLOps Assignment 1 - Penguin Classifier

This project demonstrates an end-to-end machine learning pipeline and MLOps workflow for classifying penguin species based on given features. It includes model training, prediction scripts, CI/CD setup using GitHub Actions, and a playground for testing.

---

## 📦 Project Structure

```
.
├── predictor.py                 # Prediction script using pre-trained model
├── playground.ipynb            # Interactive Jupyter notebook to test predictions
├── requirements.txt            # Dependencies list
├── index.html                  # HTML Report or interface (optional)
├── data/
│   └── prediction.json         # Sample input data for predictions
├── models/
│   ├── penguin_classifier_model.pkl    # Trained classifier
│   ├── penguin_label_encoder.pkl       # Label encoder used in preprocessing
│   ├── penguin_scaler.pkl              # Scaler used for feature normalization
│   └── t.txt                           # Misc file (can be removed or renamed)
└── .github/
    └── workflows/
        └── run_predictions.yml # GitHub Actions workflow for automated predictions
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd Mathias1801-MLOps_Assignment_1
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Making Predictions

You can make predictions by running the `predictor.py` script:

```bash
python predictor.py
```

Make sure `data/prediction.json` exists and contains the input data in the correct format.

Alternatively, explore predictions in `playground.ipynb`.

---

## ⚙️ CI/CD with GitHub Actions

This project includes a workflow file at `.github/workflows/run_predictions.yml` to automate the prediction process when new data is pushed or changes occur.

---

## 📊 Input Format Example (`prediction.json`)

```json
[
  {
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "MALE"
  }
]
```

---

## 📄 License

MIT License

---

## ✍️ Author

Mathias1801
