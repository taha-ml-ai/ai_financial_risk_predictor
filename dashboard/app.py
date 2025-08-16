
"""
Streamlit app for AI Financial Risk Predictor.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import utils

import streamlit as st
import joblib
import pandas as pd





st.title("AI Financial Risk Predictor")

# Load trained model



# Get absolute path to model
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "financial_risk_model.pkl")
model_path = os.path.abspath(model_path)

# Load the model
model = joblib.load(model_path)


st.sidebar.header("User Input Features")
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
income = st.sidebar.number_input("Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0, 100, 30)
employment_years = st.sidebar.slider("Employment Years", 0, 50, 5)
past_defaults = st.sidebar.slider("Past Defaults", 0, 10, 0)

input_df = pd.DataFrame({
    'credit_score':[credit_score],
    'income':[income],
    'loan_amount':[loan_amount],
    'dti':[dti],
    'employment_years':[employment_years],
    'past_defaults':[past_defaults]
})

if st.button("Predict Risk"):
    prediction = model.predict(input_df)
    st.write("Predicted Risk:", "High" if prediction[0]==1 else "Low")
