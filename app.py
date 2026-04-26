import joblib
import pandas as pd
import streamlit as st

# Load model and scaler
model = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Customer Segmentation", page_icon="📊")

# Title
st.title("📊 Customer Segmentation App")
st.markdown("Enter customer details below to predict their segment.")

# Input section
st.subheader("📝 Customer Details")

col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input("💰 Annual Income", 0, 1000000, 50000)

with col2:
    purchase_amount = st.number_input("🛒 Purchase Amount", 0, 10000, 300)

purchase_frequency = st.number_input("📅 Purchase Frequency (per year)", 0, 100, 15)
loyalty_score = st.slider("⭐ Loyalty Score", 0.0, 10.0, 5.0)

# Prediction
if st.button("🔍 Predict Segment"):
    data = pd.DataFrame([[ 
        annual_income,
        purchase_amount,
        purchase_frequency,
        loyalty_score
    ]], columns=[
        "annual_income",
        "purchase_amount",
        "purchase_frequency",
        "loyalty_score"
    ])

    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    st.success(f"🎯 Predicted Customer Segment: **{prediction[0]}**")

# Footer
st.markdown("---")
st.caption("🚀 Built using Machine Learning & Streamlit")