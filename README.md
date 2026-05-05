# Customer Churn Prediction

## Business Problem
A telecom company loses significant revenue every time a customer cancels 
their subscription. Acquiring a new customer costs 5 to 7 times more than 
retaining an existing one. This project builds a machine learning model to 
identify customers at risk of churning, enabling the business to intervene 
proactively with targeted retention offers.

## Dataset
- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No)
- **Class imbalance:** 73% No Churn / 27% Churn

## Approach
1. Exploratory data analysis to understand feature distributions and churn patterns
2. Data cleaning and preprocessing including encoding categorical variables
3. Handled class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
4. Trained an XGBoost classifier
5. Evaluated using ROC AUC, Precision, Recall, and F1 Score

## Results
| Metric | Score |
|---|---|
| ROC AUC | 0.83 |
| Precision (Churn) | 0.55 |
| Recall (Churn) | 0.64 |
| F1 Score (Churn) | 0.59 |

Recall was prioritized over precision because missing a churner costs 
the business a full customer, while a false alarm only costs a small 
retention offer.

## Key Findings
- **Payment method** is the strongest churn predictor. Customers paying 
  by electronic check churn at significantly higher rates.
- **Fiber optic internet** customers churn more, likely due to higher 
  monthly costs and stronger competition in that segment.
- **Paperless billing** customers are more digitally engaged and more 
  likely to shop around for better deals.

## Tech Stack
- Python 3.9
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- matplotlib, seaborn

## Project Structure
churn-prediction/
├── data/
│   └── telco_churn.csv       # Raw dataset
├── models/
│   ├── churn_model.pkl       # Saved XGBoost model
│   └── scaler.pkl            # Saved scaler
├── notebooks/
│   └── 01_churn_prediction.ipynb  # Full analysis
├── requirements.txt          # Dependencies
└── README.md                 # Project overview

## How to Run
```bash
# Clone the repo
git clone https://github.com/philipatosam/churn-prediction.git
cd churn-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## What I Would Do Next
- Tune the classification threshold to improve recall further
- Try Logistic Regression and Random Forest as baseline comparisons
- Build a simple scoring script to predict churn on new customers
- Deploy the model as an API using FastAPI on Azure