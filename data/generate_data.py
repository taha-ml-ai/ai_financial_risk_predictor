
"""
Script to generate or load financial dataset for AI Financial Risk Predictor project.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os


def generate_synthetic_data(n_samples=1000, random_state=42):
    """
    Generates a synthetic dataset for financial risk prediction.
    Returns a pandas DataFrame.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4,
        n_redundant=0, n_classes=2, random_state=random_state
    )
    df = pd.DataFrame(X, columns=[
        'credit_score','income','loan_amount','dti','employment_years','past_defaults'
    ])
    df['default_risk'] = y
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    # Save directly inside the data folder
    output_path = os.path.join(os.path.dirname(__file__), "synthetic_financial_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset created: {output_path}")