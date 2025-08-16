
"""
Train the AI model for Financial Risk Prediction.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os  

def load_data(path='../data/synthetic_financial_data.csv'):
    # Resolve absolute path based on script location
    path = os.path.join(os.path.dirname(__file__), path)
    df = pd.read_csv(path)
    X = df.drop('default_risk', axis=1)
    y = df['default_risk']
    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == "__main__":
    X, y = load_data()
    model, X_test, y_test = train_model(X, y)
    
    # Save model inside model/ folder
    model_dir = os.path.dirname(__file__)  # points to model/
    model_path = os.path.join(model_dir, "financial_risk_model.pkl")
    joblib.dump(model, model_path)
    
    print(f"Model trained and saved as {model_path}")

