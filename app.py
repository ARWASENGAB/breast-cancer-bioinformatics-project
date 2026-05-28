
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

st.title("Breast Cancer Subtype Prediction")
st.write("Upload a CSV file containing selected gene expression values.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    input_data = input_data[selected_features]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    prediction_label = encoder.inverse_transform(prediction)

    st.subheader("Prediction Result")
    st.write(prediction_label)
