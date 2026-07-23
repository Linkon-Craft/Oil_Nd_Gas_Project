import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/oil_nd_gas.pkl")


st.title("Test Oil and Gas CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload Test Oil Dataset",
    type="csv"
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)



predictions = model.predict(test_df)



test_df["Prediction"] = predictions



st.dataframe(test_df["Prediction"])





csv = test_df.to_csv(index=False)

st.download_button(
    "Download Predictions",
    csv,
    file_name="oil_predictions.csv"
)