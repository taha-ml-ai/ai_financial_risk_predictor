

"""
Utility functions for the AI Financial Risk Predictor dashboard.
"""

import pandas as pd

def preprocess_input(input_dict):
    """
    Convert input dictionary to DataFrame and apply any preprocessing if needed.
    """
    df = pd.DataFrame([input_dict])
    return df
