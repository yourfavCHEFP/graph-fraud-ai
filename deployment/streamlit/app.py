import os

import requests
import streamlit as st

from utils import predict_transaction

st.set_page_config(page_title="Graph Fraud AI", page_icon="🛡️", layout="wide")
st.title("Graph Fraud AI")
st.caption("Production-style investigation interface for the GraphSAGE fraud detector.")

api_url = os.getenv("API_URL", "http://localhost:8000/predict")
st.sidebar.info(f"Prediction API: {api_url}")

transaction_id = st.number_input("Transaction ID", min_value=0, value=0, step=1)

if st.button("Analyze transaction", type="primary"):
    try:
        result = predict_transaction(transaction_id, api_url)
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", result["prediction"].upper())
        col2.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        col3.metric("Threshold", f"{result['threshold']:.2%}")
        st.subheader("Explanation")
        st.json(result["explanation"])
    except requests.exceptions.RequestException as exc:
        st.error(f"Prediction API unavailable: {exc}")
    except Exception as exc:
        st.error(str(exc))
