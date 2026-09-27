import streamlit as st
import pickle
import numpy as np

# Load model and scaler
kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Buyer Segmentation App")
st.write("Age aur Satisfaction Score dalo, cluster pata karo")

age = st.slider("Age", 18, 70, 30)
satisfaction = st.slider("Satisfaction Score", 1, 5, 3)

if st.button("Predict Cluster"):
    data_scaled = scaler.transform([[age, satisfaction]])
    cluster = kmeans.predict(data_scaled)[0]

    cluster_names = {
        0: "Price-Sensitive (C1)",
        1: "At-Risk (C2)",
        2: "Young Achievers (C3)",
        3: "Premium Loyalists (C4)"
    }
    st.success(f"Buyer is in: **{cluster_names[cluster]}**")
