import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

# Import the prediction logic
# NOTE: Ensure your original predict.py is in the same directory
from predict import predict_single

def create_gauge_chart(probability):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability*100,
        number={'valueformat': '.2f'},
        title={'text': "Deterioration Probability"},
        gauge={
            'axis': {
               #'range': [0, 1],
                'range': [0, 1],
                'tickvals': [0, 0.25, 0.5, 0.75, 1],
                'ticktext': ['0', '0.25', '0.50', '0.75', '1.0']
            },
            'bar': {'color': 'rgba(0,0,0,0)'},
            'steps': [
                {'range': [0.0, 0.25], 'color': "#66bb6a"},
                {'range': [0.25, 0.5], 'color': "#ffa726"},
                {'range': [0.5, 1.0], 'color': "#ef5350"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.8,
                'value': probability*100
            }
        }
    ))

    fig.update_layout(height=320, margin=dict(t=60, b=10))
    return fig


# --- CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI ICU Early Deterioration Detector🏥",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #cc0000;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .risk-banner-high {
        padding: 20px;
        background-color: #ffebee;
        color: #c62828;
        border-left: 10px solid #c62828;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .risk-banner-med {
        padding: 20px;
        background-color: #fff3e0;
        color: #ef6c00;
        border-left: 10px solid #ef6c00;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .risk-banner-low {
        padding: 20px;
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 10px solid #2e7d32;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def create_gauge_chart(probability):
    """Creates a speedometer-style gauge for risk probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round((probability*100),1),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Deterioration Probability (%)"},
        gauge ={
            'axis': {'range': [0, 1],'tickvals': [0, 0.25, 0.5, 0.75, 1],'ticktext': ['0%', '25%', '50%', '75%', '100%']},

            'bar': {'color': "rgba(0,0,0,0)"}, # Hide default bar, use threshold/steps
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "yellow",
            'steps': [
                {'range': [0, 24], 'color': "#181e18"},   # Green
                {'range': [24, 50], 'color': "#ffa726"},  # Orange
                {'range': [50, 100], 'color': "#ef5350"}  # Red
            ],
        
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': probability
                #'value':probability
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# --- INITIALIZATION ---
if "run_prediction" not in st.session_state:
    st.session_state.run_prediction = False

# Load Animation (Heartbeat/Medical)
lottie_medical = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_5njp3vgg.json")

# --- HEADER SECTION ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("🏥 ICU Early Warning System")
    st.markdown("**1-Hour Clinical Deterioration Prediction**")
    st.caption("AI-powered assessment of vital signs and lab values to predict patient stability.")
with col_head2:
    if lottie_medical:
        st_lottie(lottie_medical, height=100, key="medical_anim")

st.divider()

# --- MAIN FORM ---
with st.form("icu_form"):
    
    # Using tabs for cleaner mobile organization, but accessible all at once
    tab1, tab2, tab3 = st.tabs(["🫀 Vitals", "🧪 Labs & Devices", "👤 Patient Profile"])

    # ------------------ TAB 1: VITALS ---------------------
    with tab1:
        st.subheader("Physiological Vitals")
        col1, col2 = st.columns(2)
        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 220, 90, help="Normal resting: 60-100 bpm")
            systolic_bp = st.number_input("Systolic BP (mmHg)", 50, 200, 120, help="Normal: <120 mmHg")
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 30, 130, 80, help="Normal: <80 mmHg")
        with col2:
            respiratory_rate = st.number_input("Respiratory Rate (breaths/min)", 5, 60, 18, help="Normal: 12-20 breaths/min")
            temperature_c = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0, step=0.1)
            spo2_pct = st.number_input("SpO₂ (%)", 50, 100, 97, help="Oxygen saturation. Target usually >94%")

    # ------------------ TAB 2: LABS & OXYGEN ---------------------
    with tab2:
        st.subheader("Labs & Intervention")
        col3, col4 = st.columns(2)
        with col3:
            oxygen_flow = st.number_input("Oxygen Flow (L/min)", 0, 20, 0)
            oxygen_device = st.selectbox(
                "Oxygen Device",
                ["none", "room_air", "nasal_cannula", "mask", "non_rebreather", "bipap", "cpap", "ventilator"],
                help="Type of respiratory support currently in use"
            )
            nurse_alert = st.selectbox("Nurse Alert", [0, 1], help="1 if nurse has flagged concern, 0 otherwise")
            mobility_score = st.selectbox("Mobility Score (0–1)", [0, 1], help="Clinical assessment of patient mobility")

        with col4:
            wbc_count = st.number_input("WBC Count", 2000, 30000, 8000)
            lactate = st.number_input("Lactate (mmol/L)", 0.1, 10.0, 1.2, step=0.1, help="Elevated levels may indicate sepsis or hypoxia")
            creatinine = st.number_input("Creatinine (mg/dL)", 0.1, 10.0, 1.0, step=0.1)
            crp_level = st.number_input("CRP Level (mg/L)", 0.1, 50.0, 5.0, step=0.1, help="Marker of inflammation")
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 13.0, step=0.1)

    # ------------------ TAB 3: DEMOGRAPHICS ---------------------
    with tab3:
        st.subheader("Demographics & Risk Factors")
        col5, col6 = st.columns(2)
        with col5:
            age = st.number_input("Age", 1, 120, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            hour_from_admission = st.number_input("Hours Since Admission", 0, 500, 12)
        
        with col6:
            sepsis_risk_score = st.number_input("Sepsis Risk Score", 0, 10, 1)
            comorbidity_index = st.number_input("Comorbidity Index", 0, 10, 0, help="Charlson Comorbidity Index or similar")
            admission_type = st.selectbox(
                "Admission Type",
                ["emergency", "normal", "urgent", "elective", "other"]
            )

    st.markdown("---")
    submitted = st.form_submit_button(
        "⚡ Analyze Risk Prediction",
        on_click=lambda: st.session_state.update({"run_prediction": True})
    )

# --- SIDEBAR SUMMARY ---
with st.sidebar:
    st.header("Patient Summary")
    st.info("Input data in the main form to update risk assessment.")
    if st.session_state.run_prediction:
        st.write(f"**Patient Age:** {age}")
        st.write(f"**Admission:** {admission_type.title()}")
        st.write(f"**Time in ICU:** {hour_from_admission} hrs")
        
        # Mini metrics
        m1, m2 = st.columns(2)
        m1.metric("HR", f"{heart_rate}", "bpm")
        m2.metric("SpO2", f"{spo2_pct}%", "")


# -------------------------------------
# PREDICTION LOGIC & RESULTS
# -------------------------------------
if st.session_state.run_prediction:
    
    # 1. Loading Animation
    with st.spinner('Processing vitals and calculating risk score...'):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01) # Simulate computation time for UX
            progress_bar.progress(i + 1)
    
    # 2. Prepare Data
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

    # 3. Get Prediction
    try:
        result = predict_single(input_dict)
        prob = result["final_prob"]
        final_pred = result["final_pred"]
    except Exception as e:
        st.error(f"Error in prediction module: {e}")
        st.stop()

    # 4. Display Results
    st.subheader("🔍 Prediction Results")
    
    # Layout: Gauge chart on left, Text Details on right
    res_col1, res_col2 = st.columns([1, 1.5])

    with res_col1:
        st.plotly_chart(create_gauge_chart(prob), use_container_width=True)

    with res_col2:
        #st.markdown(f"### Probability of Deterioration: **{prob:.1f}%**")
        
        # High Risk Logic
        if final_pred == 1:
            st.markdown(
                """
                <div class="risk-banner-high">
                    <h3>⚠️ HIGH RISK ALERT</h3>
                    <p>Patient predicted to deteriorate within the next 1 hour.</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if prob <= 50*1.5:
                 st.warning("Observation: Patient is in a critical condition range.")
            elif prob < 80*1.5:
                 st.error("ACTION REQUIRED: Very high risk. Immediate clinical review recommended.")
            else:
                 st.error("CRITICAL: Extreme risk of immediate deterioration.")

        # Low Risk Logic
        elif final_pred == 0:
            
            if prob <= 10*1.5:
                st.markdown(
                    """
                    <div class="risk-banner-low">
                        <h3>✅ STABLE</h3>
                        <p>Vitals do not indicate critical condition in the next 1h.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.info("Suggestion: Continue standard monitoring protocols.")

            elif prob < 24*1.5:
                st.markdown(
                    """
                    <div class="risk-banner-med">
                        <h3>⚠️ BORDERLINE RISK</h3>
                        <p>Potential for light deterioration.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.warning("Suggestion: Increase monitoring frequency.")
            else:
                 st.success("Patient currently stable, but continue to monitor trends.")

    # Reset state to prevent auto-rerun issues
    st.session_state.run_prediction = False