import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction App")

st.write("Enter patient medical details")

pregnancies = st.number_input("Pregnancies", min_value=0.0)
glucose = st.number_input("Glucose Level", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin Level", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0.0)

if st.button("Predict"):

    data = pd.DataFrame(
        [[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    prediction = model.predict(data)[0]

    if prediction == 0:
        st.success("✅The person is NOT diabetic")
    else:
        st.error(" ⚠️The person is DIABETIC")