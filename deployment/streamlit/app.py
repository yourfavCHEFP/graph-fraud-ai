import streamlit as st
from utils import predict_transaction

st.title("Graph Fraud AI Investigation Dashboard")

transaction_id = st.number_input(
    "Transaction ID",
    min_value=0,
    value=0,
)

if st.button("Analyze"):
    result = predict_transaction(transaction_id)
    st.json(result)
