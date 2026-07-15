import streamlit as st
import requests
import numpy as np
import os

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title(" AI Credit Card Fraud Detection System")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("ROC-AUC", "0.98")
st.sidebar.metric("Recall (Fraud)", "92%")
st.sidebar.metric("Precision", "85%")
st.sidebar.metric("F1-Score", "88%")

with st.form("fraud_form"):
    st.subheader("Enter Transaction Features")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount ($)", value=100.0)
        time = st.number_input("Time (seconds)", value=50000)
    with col2:
        v1 = st.slider("V1", -5.0, 5.0, 0.0, 0.1)
        v2 = st.slider("V2", -5.0, 5.0, 0.0, 0.1)
        v3 = st.slider("V3", -5.0, 5.0, 0.0, 0.1)
        v4 = st.slider("V4", -5.0, 5.0, 0.0, 0.1)
        v5 = st.slider("V5", -5.0, 5.0, 0.0, 0.1)
        v6 = st.slider("V6", -5.0, 5.0, 0.0, 0.1)
        v7 = st.slider("V7", -5.0, 5.0, 0.0, 0.1)
        v8 = st.slider("V8", -5.0, 5.0, 0.0, 0.1)
        v9 = st.slider("V9", -5.0, 5.0, 0.0, 0.1)
        v10 = st.slider("V10", -5.0, 5.0, 0.0, 0.1)
    submitted = st.form_submit_button("Analyze")

if submitted:
    features = (
        [float(time), v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]
        + [0.0] * 18
        + [float(amount)]
    )
    # That gives time + V1..V10 (11) + 18 zeros (V11..V28) + amount = 30
    try:
        resp = requests.post(f"{API_URL}/predict", json={"features": features})
        if resp.status_code == 200:
            res = resp.json()
            if res["is_fraud"]:
                st.error(
                    f"⚠️ FRAUD DETECTED – Probability: {res['probability']*100:.2f}%"
                )
            else:
                st.success(
                    f"✅ Legitimate – Probability: {res['probability']*100:.2f}%"
                )
            st.progress(res["probability"])
        else:
            st.error(f"API error: {resp.status_code}")
    except:
        st.error("❌ Cannot connect to API. Is FastAPI running?")
