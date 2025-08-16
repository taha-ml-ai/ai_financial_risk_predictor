
"""
Evaluate the AI Financial Risk Prediction model.
"""

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import os


def load_model(filename="financial_risk_model.pkl"):
    # Build absolute path based on this script location
    model_path = os.path.join(os.path.dirname(__file__), filename)
    return joblib.load(model_path)


def load_test_data(filename='synthetic_financial_data.csv'):
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    df = pd.read_csv(data_path)
    X_test = df.drop('default_risk', axis=1)
    y_test = df['default_risk']
    return X_test, y_test


if __name__ == "__main__":
    model = load_model()
    X_test, y_test = load_test_data()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
