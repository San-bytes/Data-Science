# Customer Churn Prediction — MCA DS Project

Predict whether a telecom customer will churn using Python, scikit-learn, and SMOTE.

## Project Structure
```
churn-project/
├── churn_prediction.ipynb   # Main notebook (exploration, modeling, evaluation)
├── churn_rf_smote.pkl       # Saved best model (created after running notebook)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── Telco-Customer-Churn.csv # Dataset (place this file here — not included)
```

## Dataset
Uses the Telco Customer Churn dataset (≈7,000 customers).  
Target column: `Churn` (Yes/No).

Place `Telco-Customer-Churn.csv` in this folder before running the notebook.

## How to Run
```bash
# 1. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook churn_prediction.ipynb
```

Run all cells top to bottom.

## What the Notebook Does
1. Loads and cleans the Telco churn data (`TotalCharges` → numeric, drops missing, removes `customerID`)
2. Encodes categorical variables with one-hot encoding
3. Splits data using **stratified sampling** (keeps churn ratio consistent)
4. Handles class imbalance with **SMOTE** on training data only
5. Trains Decision Tree, Random Forest, and Random Forest + SMOTE
6. Evaluates with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix
7. Saves the best model as `churn_rf_smote.pkl`

## Key Result
Random Forest with SMOTE improved recall for the churn class (caught more at-risk customers) compared to the baseline models, even though overall accuracy was slightly lower. For churn retention, recall matters more than raw accuracy.

## Author
MCA Student — Data Science Internship Prep
