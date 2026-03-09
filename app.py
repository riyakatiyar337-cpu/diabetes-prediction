import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Diabetes Prediction App")

st.write("Enter patient details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0.0)
bp = st.number_input("Blood Pressure", min_value=0.0)
skin = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0, step=1)

# Prediction button
if st.button("Predict"):

    data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ Patient is likely to have Diabetes")
    else:
        st.success("✅ Patient is unlikely to have Diabetes")