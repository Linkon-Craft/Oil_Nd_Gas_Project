import streamlit as st
import pandas as pd
import joblib


model = joblib.load("models/oil_nd_gas.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(
        "train_oil.csv",
        engine="python",
        on_bad_lines="skip"
    )

    return df[df["Onshore/Offshore"] != "ONSHORE-OFFSHORE"]

train_oil = load_data()

st.header("Single Predict Oil Field Location")

with st.form("prediction_form"):

    country = st.selectbox(
    "Country",
    sorted(train_oil["Country"].dropna().unique())
    )

    region = st.selectbox(
    "Region",
    sorted(train_oil["Region"].dropna().unique())
    )

    basin = st.selectbox(
    "Basin Name",
    sorted(train_oil["Basin name"].dropna().unique())
    )

    Operator_Company = st.selectbox(
    "Operator company",
    sorted(train_oil["Operator company"].dropna().unique())
    )

    Hydrocarbon_type = st.selectbox(
    "Hydrocarbon type",
    sorted(train_oil["Hydrocarbon type"].dropna().unique())
    )


    Depth = st.selectbox(
    "Depth",
    sorted(train_oil["Depth"].dropna().unique())
    )

    Field_Name = st.selectbox(
    "Field name",
    sorted(train_oil["Field name"].dropna().unique())
    )

    Latitude = st.selectbox(
    "Latitude",
    sorted(train_oil["Latitude"].dropna().unique())
    )


    Longitude = st.selectbox(
    "Longitude",
    sorted(train_oil["Longitude"].dropna().unique())
    )

    Reservoir = st.selectbox(
    "Reservoir unit",
    sorted(train_oil["Reservoir unit"].dropna().unique())
    )

    Lithology = st.selectbox(
    "Lithology",
    sorted(train_oil["Lithology"].dropna().unique())
    )

    Porosity = st.selectbox(
        "Porosity",
        sorted(train_oil["Porosity"].dropna().unique())
    )

    Tectonic_regime = st.selectbox(
        "Tectonic regime",
        sorted(train_oil["Tectonic regime"].dropna().unique())
    )

    Reservoir_status = st.selectbox(
        "Reservoir status",
        sorted(train_oil["Reservoir status"].dropna().unique())
    )

    Structural_setting = st.selectbox(
        "Structural setting",
        sorted(train_oil["Structural setting"].dropna().unique())
    )

    Reservoir_period = st.selectbox(
        "Reservoir period",
        sorted(train_oil["Reservoir period"].dropna().unique())
    )

    Thickness_gross = st.selectbox(
        "Thickness (gross average ft)",
        sorted(train_oil["Thickness (gross average ft)"].dropna().unique())
    )

    Thickness_net = st.selectbox(
        "Thickness (net pay average ft)",
        sorted(train_oil["Thickness (net pay average ft)"].dropna().unique())
    )

    Permeability = st.selectbox(
        "Permeability",
        sorted(train_oil["Permeability"].dropna().unique())
    )

    submit = st.form_submit_button("Predict")

if submit:

    sample = pd.DataFrame({
        "Country": [country],
        "Region": [region],
        "Basin name": [basin],
        "Hydrocarbon type": [Hydrocarbon_type],
        "Operator company": [Operator_Company],
        "Depth": [Depth],
        "Field name": [Field_Name],
        "Latitude": [Latitude],
        "Longitude": [Longitude],
        "Reservoir unit": [Reservoir],
        "Lithology": [Lithology],
        "Porosity": [Porosity],
        "Tectonic regime": [Tectonic_regime],
        "Reservoir status": [Reservoir_status],
        "Structural setting": [Structural_setting],
        "Reservoir period": [Reservoir_period],
        "Thickness (gross average ft)": [Thickness_gross],
        "Thickness (net pay average ft)": [Thickness_net],
        "Permeability": [Permeability]
    })

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample).max()

    if prediction == "ONSHORE":
        st.success(f"🌍 Predicted Class: {prediction}")
    else:
        st.info(f"🌊 Predicted Class: {prediction}")

    st.metric(
        "Confidence",
        f"{probability:.2%}"
    )