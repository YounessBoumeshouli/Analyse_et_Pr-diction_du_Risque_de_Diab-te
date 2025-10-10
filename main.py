import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

st.title("Predict Diabetes Risk")

Pregnancies = st.number_input("Pregnancies", 0.0, 20.0, 0.0, step=1.0, key="pregnancies")
Glucose = st.number_input("Glucose", 0.0, 200.0, 100.0, step=1.0, key="glucose")
BloodPressure = st.number_input("Blood Pressure", 0.0, 150.0, 70.0, step=1.0, key="bp")
SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0, step=1.0, key="skin")
Insulin = st.number_input("Insulin", 0.0, 900.0, 80.0, step=1.0, key="insulin")
BMI = st.number_input("BMI", 0.0, 70.0, 25.0, step=0.1, key="bmi")
DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01, key="dpf")
Age = st.number_input("Age", 0.0, 120.0, 30.0, step=1.0, key="age")

if st.button("Predict"):
    features = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]]
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame(features, columns=columns)

    model_path = os.path.join(os.getcwd(), 'model.joblib')
    scaler_path = os.path.join(os.getcwd(), 'scaler.joblib')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load(model_path)
        scaler = load(scaler_path)

        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Risk: {pred:.2f}")
    else:
        st.error("❌ Model or scaler file not found. Make sure both 'model.joblib' and 'scaler.joblib' are in the same directory.")
