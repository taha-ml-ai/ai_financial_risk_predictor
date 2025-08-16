# AI Financial Risk Predictor

---

## **Project Overview**

The AI Financial Risk Predictor is a Python-based machine learning project that predicts financial risk (default/non-default) using structured financial data. The project includes:

- **Synthetic dataset generation** (`data/generate_data.py`)  
- **ML model training** (`model/train_model.py`)  
- **Model evaluation** (`model/evaluate_model.py`)  
- **Interactive dashboard** using Streamlit (`dashboard/app.py`)  

**Key Skills Demonstrated:**

- Python ML (classification, Random Forest)
- Data preprocessing & EDA
- Model evaluation (accuracy, precision, recall, ROC-AUC)
- Interactive dashboards with Streamlit
- Windows-friendly path handling

---

## **Folder Structure**

ai_financial_risk_predictor/
│
├── data/
│ ├── generate_data.py
│ └── synthetic_financial_data.csv
│
├── model/
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── financial_risk_model.pkl
│
├── notebooks/
│ └── eda_model.ipynb
│
├── dashboard/
│ ├── app.py
│ └── utils.py
│
├── requirements.txt
└── README.md


---

## **Setup Instructions**

1. **Clone the repository** (or place your local project here):

```bash
git clone https://github.com/taha-ml-ai/ai_financial_risk_predictor.git
cd ai_financial_risk_predictor

conda create -n ai_financial_risk python=3.11 -y
conda activate ai_financial_risk
pip install -r requirements.txt
python data\generate_data.py
python model\train_model.py
python model\evaluate_model.py


Run the Dashboard

Start the Streamlit dashboard to make interactive predictions:

streamlit run dashboard\app.py


Open the provided URL in your browser (usually http://localhost:8501).

Use the sliders and input options to test financial risk predictions.

