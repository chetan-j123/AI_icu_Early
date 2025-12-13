import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import os

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ICU Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ICU Monitor Look
st.markdown("""
    <style>
    /* Dark Theme Adjustment for Charts */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* ICU Card Styling */
    .metric-card {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
    }
    
    /* Risk Alert Boxes */
    .risk-low {
        background-color: #064e3b; /* Dark Green */
        color: #34d399;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #34d399;
    }
    .risk-moderate {
        background-color: #78350f; /* Dark Orange */
        color: #fbbf24;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #fbbf24;
    }
    .risk-high {
        background-color: #7f1d1d; /* Dark Red */
        color: #f87171;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #f87171;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(248, 113, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING & MOCKING
# -----------------------------------------------------------------------------
class MockModel:
    """Fallback model if model.pkl is not found."""
    def predict(self, X):
        # Mock logic: High HR or Low SpO2 triggers deterioration
        # X order assumed: [HR, Resp, SpO2, Temp, SBP, DBP]
        risk = []
        for row in X:
            score = 0
            if row[0] > 110 or row[0] < 50: score += 1 # HR
            if row[2] < 92: score += 2 # SpO2
            if row[4] < 90: score += 1 # SBP
            risk.append(1 if score >= 2 else 0)
        return np.array(risk)

    def predict_proba(self, X):
        # Generate fake probabilities based on the simple logic above
        probs = []
        for row in X:
            score = 0
            if row[0] > 100: score += 0.2
            if row[1] > 25: score += 0.2
            if row[2] < 95: score += 0.3
            if row[4] < 100: score += 0.2
            base_prob = 0.1 + score
            base_prob = min(base_prob, 0.99)
            probs.append([1-base_prob, base_prob]) # [Class 0, Class 1]
        return np.array(probs)

@st.cache_resource
def load_model():
    """Loads the ML model or returns a mock if file missing."""
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return MockModel(), False
    else:
        return MockModel(), False

model, is_real_model = load_model()

def predict_icu_risk(features):
    """
    Predicts risk using the loaded model.
    features: [HR, RespRate, SpO2, Temp, SBP, DBP]
    """
    # Ensure 2D array
    features_arr = np.array([features])
    pred = model.predict(features_arr)[0]
    prob = model.predict_proba(features_arr)[0][1]
    return pred, prob

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk", "risk-low", "✅"
    elif prob < 0.7:
        return "Moderate Risk", "risk-moderate", "⚠️"
    else:
        return "High Risk", "risk-high", "🚨"

def generate_explanation(vitals_dict):
    """
    Generates rule-based explanations for the user based on vitals.
    """
    reasons = []
    
    if vitals_dict['SpO2'] < 92:
        reasons.append(f"**Hypoxia detected**: SpO2 is critically low ({vitals_dict['SpO2']}%)")
    elif vitals_dict['SpO2'] < 95:
        reasons.append(f"SpO2 is suboptimal ({vitals_dict['SpO2']}%)")
        
    if vitals_dict['Heart Rate'] > 100:
        reasons.append(f"**Tachycardia**: Heart Rate is elevated ({vitals_dict['Heart Rate']} bpm)")
    elif vitals_dict['Heart Rate'] < 60:
        reasons.append(f"**Bradycardia**: Heart Rate is low ({vitals_dict['Heart Rate']} bpm)")
        
    if vitals_dict['Systolic BP'] < 90:
        reasons.append(f"**Hypotension**: Systolic BP is dangerously low ({vitals_dict['Systolic BP']} mmHg)")
    
    if vitals_dict['Resp Rate'] > 25:
        reasons.append(f"**Tachypnea**: Respiratory rate is high ({vitals_dict['Resp Rate']} bpm)")

    if not reasons:
        return "All vitals are currently within stable ranges. Model predicts stability."
    return " | ".join(reasons)

# -----------------------------------------------------------------------------
# 4. UI LAYOUT
# -----------------------------------------------------------------------------

# Header
st.title("🏥 ICU Early Warning System")
st.markdown("**Real-time Patient Deterioration Prediction**")
st.markdown("---")

# Layout: Left (Controls) vs Right (Dashboard)
left_col, right_col = st.columns([1, 3])

with left_col:
    st.subheader("⚙️ Patient Vitals Input")
    st.info("Adjust sliders to simulate patient condition or start auto-simulation.")
    
    # Simulation Mode Toggle
    sim_mode = st.checkbox("🔄 Auto-Simulation Mode", value=False)
    
    # Input Controls
    # Using session state to allow updates from simulation loop if needed, 
    # but for simplicity, we read sliders directly or generate randoms.
    
    hr_val = st.slider("Heart Rate (bpm)", 40, 180, 80)
    resp_val = st.slider("Resp. Rate (breaths/min)", 8, 40, 18)
    spo2_val = st.slider("SpO2 (%)", 70, 100, 98)
    temp_val = st.slider("Temperature (°C)", 35.0, 42.0, 37.0)
    sbp_val = st.slider("Systolic BP (mmHg)", 70, 200, 120)
    dbp_val = st.slider("Diastolic BP (mmHg)", 40, 120, 80)

    start_btn = st.button("▶️ Start Monitoring", type="primary")
    stop_btn = st.button("⏹️ Stop")

    st.markdown("---")
    st.caption(f"Model Loaded: {'✅ Real (model.pkl)' if is_real_model else '⚠️ Mock (Simulation)'}")

# -----------------------------------------------------------------------------
# 5. REAL-TIME DASHBOARD LOGIC
# -----------------------------------------------------------------------------

with right_col:
    # Container for the dashboard
    dashboard_container = st.container()
    
    # Container for Prediction
    prediction_container = st.container()

    # Placeholders for the 6 Vitals
    # We create a 2x3 grid logic using columns inside the container
    with dashboard_container:
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        # Initialize placeholders for metrics and charts
        # Row 1
        p_hr_metric = row1_col1.empty()
        p_hr_chart = row1_col1.empty()
        
        p_resp_metric = row1_col2.empty()
        p_resp_chart = row1_col2.empty()
        
        p_spo2_metric = row1_col3.empty()
        p_spo2_chart = row1_col3.empty()

        # Row 2
        p_temp_metric = row2_col1.empty()
        p_temp_chart = row2_col1.empty()
        
        p_sbp_metric = row2_col2.empty()
        p_sbp_chart = row2_col2.empty()
        
        p_dbp_metric = row2_col3.empty()
        p_dbp_chart = row2_col3.empty()

    # Prediction Placeholder
    with prediction_container:
        st.markdown("### 🧠 AI Deterioration Risk Assessment")
        p_risk_alert = st.empty()
        p_explanation = st.empty()


# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION LOOP
# -----------------------------------------------------------------------------
if start_btn:
    # Initialize sliding window history
    history_len = 50
    history = {
        'HR': [hr_val] * history_len,
        'Resp': [resp_val] * history_len,
        'SpO2': [spo2_val] * history_len,
        'Temp': [temp_val] * history_len,
        'SBP': [sbp_val] * history_len,
        'DBP': [dbp_val] * history_len
    }

    # Helper to render a specific vital
    def render_vital(metric_ph, chart_ph, label, value, history_data, color, unit):
        # Update Metric Text
        metric_ph.markdown(
            f"""<div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value' style='color:{color}'>{value} <span style='font-size:12px'>{unit}</span></div>
            </div>""", 
            unsafe_allow_html=True
        )
        # Update Chart using Streamlit native line chart (fastest for loops)
        # We assume history_data is a list. st.line_chart needs a list or df.
        chart_ph.line_chart(history_data, height=100)

    # Main Loop (Simulation)
    # Running for 200 iterations as requested, or until stopped (simulated by break logic)
    for i in range(200):
        # 1. Get New Data
        if sim_mode:
            # Random Walk Simulation
            current_hr = int(history['HR'][-1] + np.random.randint(-2, 3))
            current_resp = int(history['Resp'][-1] + np.random.randint(-1, 2))
            current_spo2 = min(100, int(history['SpO2'][-1] + np.random.randint(-1, 2)))
            current_temp = round(history['Temp'][-1] + np.random.uniform(-0.1, 0.1), 1)
            current_sbp = int(history['SBP'][-1] + np.random.randint(-2, 3))
            current_dbp = int(history['DBP'][-1] + np.random.randint(-2, 3))
            
            # Constrain physics
            current_hr = max(40, min(180, current_hr))
            current_spo2 = max(70, min(100, current_spo2))
        else:
            # Use Manual Inputs (from sliders)
            # Note: Sliders won't update *during* the loop in standard Streamlit unless we rerun,
            # but this variable reading works if we assume the loop runs once per state.
            # However, for a "monitoring" effect without page reload, we use the initial values 
            # plus slight noise to show the chart moving.
            noise = np.random.normal(0, 0.5)
            current_hr = int(hr_val + noise)
            current_resp = int(resp_val + noise)
            current_spo2 = int(spo2_val) # SpO2 usually stable
            current_temp = temp_val
            current_sbp = int(sbp_val + noise)
            current_dbp = int(dbp_val + noise)

        # 2. Append to History (Sliding Window)
        history['HR'].append(current_hr)
        history['HR'] = history['HR'][-history_len:]
        
        history['Resp'].append(current_resp)
        history['Resp'] = history['Resp'][-history_len:]
        
        history['SpO2'].append(current_spo2)
        history['SpO2'] = history['SpO2'][-history_len:]
        
        history['Temp'].append(current_temp)
        history['Temp'] = history['Temp'][-history_len:]
        
        history['SBP'].append(current_sbp)
        history['SBP'] = history['SBP'][-history_len:]
        
        history['DBP'].append(current_dbp)
        history['DBP'] = history['DBP'][-history_len:]

        # 3. Update UI Charts
        # Heart Rate (Green)
        render_vital(p_hr_metric, p_hr_chart, "Heart Rate", current_hr, history['HR'], "#4ade80", "bpm")
        # Resp Rate (Blue)
        render_vital(p_resp_metric, p_resp_chart, "Resp. Rate", current_resp, history['Resp'], "#60a5fa", "bpm")
        # SpO2 (Yellow/Cyan)
        render_vital(p_spo2_metric, p_spo2_chart, "SpO2", current_spo2, history['SpO2'], "#22d3ee", "%")
        # Temp (Orange)
        render_vital(p_temp_metric, p_temp_chart, "Temperature", current_temp, history['Temp'], "#fbbf24", "°C")
        # SBP (Red)
        render_vital(p_sbp_metric, p_sbp_chart, "Systolic BP", current_sbp, history['SBP'], "#f87171", "mmHg")
        # DBP (Red)
        render_vital(p_dbp_metric, p_dbp_chart, "Diastolic BP", current_dbp, history['DBP'], "#f87171", "mmHg")

        # 4. Prediction
        # Prepare vector: [HR, Resp, SpO2, Temp, SBP, DBP]
        feature_vector = [current_hr, current_resp, current_spo2, current_temp, current_sbp, current_dbp]
        pred_class, pred_prob = predict_icu_risk(feature_vector)
        
        risk_text, risk_class, icon = get_risk_level(pred_prob)
        
        # 5. Update Prediction UI
        p_risk_alert.markdown(
            f"""<div class='{risk_class}'>
                <h2>{icon} {risk_text}</h2>
                <p>Deterioration Probability: {pred_prob:.1%}</p>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # 6. Feature Explanation
        vitals_dict = {
            'Heart Rate': current_hr, 'Resp Rate': current_resp, 
            'SpO2': current_spo2, 'Temp': current_temp, 
            'Systolic BP': current_sbp, 'Diastolic BP': current_dbp
        }
        explanation_text = generate_explanation(vitals_dict)
        
        p_explanation.info(f"**Clinical Context:** {explanation_text}")

        # 7. Sleep to simulate 1-second interval
        time.sleep(0.8) # Slightly faster than 1s to account for processing time

elif not start_btn:
    # Initial Placeholder State
    st.info("👋 System Ready. Click 'Start Monitoring' on the left to begin.")