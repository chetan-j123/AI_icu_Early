import streamlit as st
import pandas as pd
import numpy as np
from predict import predict_single

st.set_page_config(page_title="ICU Early Deterioration Detector", layout="wide")

st.title("ICU Early Warning System (1-Hour Deterioration Prediction)")
st.write("Enter patient vitals and lab values to predict clinical deterioration risk.")

# -------------------------------------
# INPUT FORM
# -------------------------------------
with st.form("icu_form"):

    st.subheader("Patient Vitals & Labs")

    col1, col2, col3 = st.columns(3)

    # ------------------ COLUMN 1 ---------------------
    with col1:
        heart_rate = st.number_input("Heart Rate", 30, 220, 90)
        systolic_bp = st.number_input("Systolic BP", 50, 200, 120)
        diastolic_bp = st.number_input("Diastolic BP", 30, 130, 80)
        respiratory_rate = st.number_input("Respiratory Rate", 5, 60, 18)
        temperature_c = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0)
        hour_from_admission = st.number_input("Hours Since Admission", 0, 200, 12)

    # ------------------ COLUMN 2 ---------------------
    with col2:
        spo2_pct = st.number_input("SpO₂ (%)", 50, 100, 97)
        oxygen_flow = st.number_input("Oxygen Flow (L/min)", 0, 20, 0)
        oxygen_device = st.selectbox(
            "Oxygen Device",
            ["none", "room_air", "nasal_cannula", "mask",
             "non_rebreather", "bipap", "cpap", "ventilator"]
        )
        nurse_alert = st.selectbox("Nurse Alert", [0, 1])
        mobility_score = st.selectbox("Mobility Score (0–1)", [0, 1])

    # ------------------ COLUMN 3 ---------------------
    with col3:
        wbc_count = st.number_input("WBC Count", 2000, 30000, 8000)
        lactate = st.number_input("Lactate", 0.1, 10.0, 1.2)
        creatinine = st.number_input("Creatinine", 0.1, 10.0, 1.0)
        crp_level = st.number_input("CRP Level", 0.1, 50.0, 5.0)
        hemoglobin = st.number_input("Hemoglobin", 5.0, 20.0, 13.0)

    st.subheader("Patient Info")
    col4, col5 = st.columns(2)

    with col4:
        age = st.number_input("Age", 1, 120, 45)
        gender = st.selectbox("Gender", ["M", "F"])

    with col5:
        sepsis_risk_score = st.number_input("Sepsis Risk Score", 0, 10, 1)
        comorbidity_index = st.number_input("Comorbidity Index", 0, 10, 0)
        admission_type = st.selectbox(
            "Admission Type",
            ["emergency", "normal", "urgent", "elective", "other"]
        )

    submitted = st.form_submit_button("Predict")

# -------------------------------------
# PREDICTION RESULT
# -------------------------------------
if submitted:

    # -------------------------------------------
    # RAW INPUTS EXACTLY IN TRAINING SEQUENCE
    # -------------------------------------------
    input_dict = {
        "heart_rate": heart_rate,
        "spo2_pct": spo2_pct,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "respiratory_rate": respiratory_rate,
        "temperature_c": temperature_c,
        "oxygen_flow": oxygen_flow,
        "mobility_score": mobility_score,
        "nurse_alert": nurse_alert,
        "wbc_count": wbc_count,
        "lactate": lactate,
        "creatinine": creatinine,
        "crp_level": crp_level,
        "hemoglobin": hemoglobin,
        "sepsis_risk_score": sepsis_risk_score,
        "age": age,
        "comorbidity_index": comorbidity_index,
        "hour_from_admission": hour_from_admission,
        "gender": gender,
        "oxygen_device": oxygen_device,
        "admission_type": admission_type
    }

    # -------------------------------------------
    # CALL MODEL
    # -------------------------------------------
    result = predict_single(input_dict)

    # -------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------
    st.subheader("Prediction Summary")


    st.write(f"Logistic Probability: **{result['logistic_prob']:.3f}**")
    st.write(f"Random Forest Probability: **{result['rf_prob']:.3f}**")
    st.write(f"Final Ensemble Probability: **{result['final_prob']:.3f}**")

    if result["final_pred"] == 1:
        st.error("HIGH RISK: Patient may deteriorate in the next 1 hour.")
    else:
        st.success("LOW RISK: Patient stable for the next 1 hour.")
