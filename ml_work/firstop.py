import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ----------------------------------------------------------------------------- 
# 1. PAGE CONFIGURATION & THEME SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ICU Neural Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------------- 
# 2. MEDICAL CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
VITALS_CONFIG = {
    "Heart Rate": {
        "unit": "bpm",
        "normal": (60, 100),
        "critical": (50, 120),
        "color": "#FF4B4B"
    },
    "Sys BP": {
        "unit": "mmHg",
        "normal": (90, 140),
        "critical": (80, 180),
        "color": "#FFA15A"
    },
    "Dia BP": {
        "unit": "mmHg",
        "normal": (60, 90),
        "critical": (50, 100),
        "color": "#FFA15A"
    },
    "SpO2": {
        "unit": "%",
        "normal": (95, 100),
        "critical": (90, 100),
        "color": "#00ADB5"
    },
    "Resp Rate": {
        "unit": "bpm",
        "normal": (12, 20),
        "critical": (8, 30),
        "color": "#F9F871"
    },
    "Temp": {
        "unit": "°C",
        "normal": (36.5, 37.5),
        "critical": (36.0, 38.0),
        "color": "#B39CD0"
    },
}

# ----------------------------------------------------------------------------- 
# 3. ADVANCED CSS INJECTION
# -----------------------------------------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
       .stApp { background-color: #0e1117; }
       .vital-card {
            background-color: #1E1E1E;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            border-left: 6px solid #444;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
       .metric-label { font-size: 14px; font-weight: 600; color: #a0a0a0; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 5px; }
       .metric-value { font-size: 38px; font-weight: 700; color: #ffffff; font-family: 'Roboto Mono', monospace; }
       .metric-unit { font-size: 16px; font-weight: 400; color: #666; margin-left: 5px; }
       @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); } 70% { box-shadow: 0 0 0 12px rgba(255, 75, 75, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); } }
       @keyframes pulse-yellow { 0% { box-shadow: 0 0 0 0 rgba(255, 161, 90, 0.7); } 70% { box-shadow: 0 0 0 12px rgba(255, 161, 90, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 161, 90, 0); } }
       .status-normal { border-left-color: #00ADB5; }
       .status-warning { border-left-color: #FFA15A; animation: pulse-yellow 2s infinite; background-color: #2a2018; }
       .status-critical { border-left-color: #FF4B4B; animation: pulse-red 1.2s infinite; background-color: #2a1010; border: 1px solid #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------- 
# 4. BACKEND LOGIC: PATIENT SIMULATOR & STATE MANAGEMENT
# -----------------------------------------------------------------------------
class PatientMonitor:
    """
    Manages a sliding window buffer in Streamlit session state and produces
    synthetic vitals using a Random Walk with optional sepsis drift.
    """
    def __init__(self, seed_rows: int = 10):
        if 'patient_data' not in st.session_state:
            cols = list(VITALS_CONFIG.keys()) + ["Timestamp", "Risk_Score"]
            st.session_state.patient_data = pd.DataFrame(columns=cols)
            # Seed baseline rows
            initial_data = []
            base_time = datetime.now() - timedelta(seconds=seed_rows)
            for i in range(seed_rows):
                row = {
                    "Heart Rate": round(80 + np.random.normal(0, 2), 1),
                    "Sys BP": round(115 + np.random.normal(0, 3), 1),
                    "Dia BP": round(75 + np.random.normal(0, 2), 1),
                    "SpO2": round(98 + np.random.normal(0, 0.5), 1),
                    "Resp Rate": round(16 + np.random.normal(0, 1), 1),
                    "Temp": round(37.0 + np.random.normal(0, 0.1), 1),
                    "Timestamp": base_time + timedelta(seconds=i),
                    "Risk_Score": 0.1
                }
                initial_data.append(row)
            st.session_state.patient_data = pd.DataFrame(initial_data)

    def generate_vitals(self, sepsis_mode: bool = False):
        # Ensure there is at least one row (defensive)
        if st.session_state.patient_data.empty:
            self.__init__(seed_rows=10)

        last_row = st.session_state.patient_data.iloc[-1]
        new_row = {"Timestamp": datetime.now()}

        # Drift applied when sepsis_mode True
        drift = {
            "Heart Rate": 1.0 if sepsis_mode else 0.0,
            "Sys BP": -1.5 if sepsis_mode else 0.0,
            "Dia BP": -0.8 if sepsis_mode else 0.0,
            "SpO2": -0.3 if sepsis_mode else 0.0,
            "Resp Rate": 0.5 if sepsis_mode else 0.0,
            "Temp": 0.1 if sepsis_mode else 0.0
        }

        # Random walk step per vital
        for vital in VITALS_CONFIG.keys():
            prev = float(last_row[vital])
            noise_sigma = 0.5 if vital != "Temp" else 0.05
            val = prev + drift.get(vital, 0.0) + np.random.normal(0, noise_sigma)

            # Physiological clamping
            if vital == "SpO2":
                val = min(100.0, max(50.0, val))
            if vital == "Temp":
                val = min(43.0, max(30.0, val))
            if vital == "Heart Rate":
                val = max(0.0, val)

            new_row[vital] = round(val, 1)

        # ML / Risk simulation (Shock Index + qSOFA)
        hr = new_row["Heart Rate"]
        sbp = new_row["Sys BP"]
        rr = new_row["Resp Rate"]

        shock_index = hr / sbp if sbp > 0 else 0.0

        qsofa = 0
        if rr >= 22:
            qsofa += 1
        if sbp <= 100:
            qsofa += 1

        # Compose a bounded risk score (0.01 - 0.99)
        risk_score = 0.1 + (0.5 * shock_index) + (0.2 * qsofa)
        # Add small random jitter to avoid flat-lines
        risk_score += np.random.normal(0, 0.02)
        risk_score = float(min(0.99, max(0.01, risk_score)))

        new_row["Risk_Score"] = round(risk_score, 3)

        # Append safely to session_state DataFrame
        new_df = pd.DataFrame([new_row])
        st.session_state.patient_data = pd.concat([st.session_state.patient_data, new_df], ignore_index=True)

        # Keep sliding window (last 300 rows to be generous)
        max_rows = 300
        if len(st.session_state.patient_data) > max_rows:
            st.session_state.patient_data = st.session_state.patient_data.iloc[-max_rows:].reset_index(drop=True)

        return new_row

    def determine_status(self, vital: str, value: float) -> str:
        crit_low, crit_high = VITALS_CONFIG[vital]["critical"]
        norm_low, norm_high = VITALS_CONFIG[vital]["normal"]
        if value < crit_low or value > crit_high:
            return "critical"
        if value < norm_low or value > norm_high:
            return "warning"
        return "normal"

# ----------------------------------------------------------------------------- 
# 5. UI COMPONENTS
# -----------------------------------------------------------------------------
def render_vital_card(label: str, value, unit: str, status: str, color_code: str):
    display_value = value if value is not None else "—"
    html = f"""
    <div class="vital-card status-{status}">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color_code if status == 'normal' else '#FFF'}">
            {display_value} <span class="metric-unit">{unit}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_sparkline(df: pd.DataFrame, vital_name: str):
    # Defensive: ensure Timestamp exists and is datetime
    if "Timestamp" in df.columns:
        df_chart = df[[vital_name, "Timestamp"]].dropna().copy()
        df_chart = df_chart.set_index(pd.to_datetime(df_chart["Timestamp"]))
        df_chart = df_chart[vital_name].tail(60)  # last 60 points
        if not df_chart.empty:
            st.line_chart(df_chart, height=100)
        else:
            st.write("No data yet")
    else:
        st.write("No history available")

# ----------------------------------------------------------------------------- 
# 6. MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    inject_custom_css()

    # Sidebar controls
    with st.sidebar:
        st.title("ICU Controls")
        st.markdown("---")
        st.caption("PATIENT DETAILS")
        st.info("**ID:** PT-2024-X92\n\n**Age:** 64\n\n**Dx:** Pneumonia")
        st.markdown("---")
        st.caption("SIMULATION PARAMETERS")
        sepsis_mode = st.checkbox("⚠️ Simulate Sepsis Decompensation", value=False)
        refresh_rate = st.slider("Refresh Rate (seconds)", 0.5, 5.0, 1.0)
        st.markdown("---")
        st.write("System Status: **ONLINE**")

    st.title("ICU Early Warning System (EWS)")
    st.markdown("Real-time telemetry and hemodynamic monitoring.")

    # Backend monitor
    monitor = PatientMonitor()

    # Real-time fragment: only re-runs this section at the refresh_rate
    @st.fragment(run_every=refresh_rate)
    def dashboard_fragment():
        # A. Generate new vitals and get history
        current_data = monitor.generate_vitals(sepsis_mode)
        df_history = st.session_state.patient_data.copy()

        # Extract numbers safely
        hr = current_data.get("Heart Rate", None)
        sbp = current_data.get("Sys BP", None)
        dbp = current_data.get("Dia BP", None)
        spo2 = current_data.get("SpO2", None)
        rr = current_data.get("Resp Rate", None)
        temp = current_data.get("Temp", None)
        risk = current_data.get("Risk_Score", 0.0)

        # B. Global alert banner
        if risk > 0.8:
            st.error(f"🚨 CRITICAL SEPSIS WARNING - RISK SCORE: {risk:.2f}")
        elif risk > 0.5:
            st.warning(f"⚠️ ELEVATED RISK - MONITOR CLOSELY - SCORE: {risk:.2f}")
        else:
            st.success(f"Risk Score: {risk:.2f}")

        # C. Layout grid
        col1, col2, col3 = st.columns(3)
        with col1:
            stat = monitor.determine_status("Heart Rate", hr)
            render_vital_card("Heart Rate", hr, VITALS_CONFIG["Heart Rate"]["unit"], stat, VITALS_CONFIG["Heart Rate"]["color"])
            render_sparkline(df_history, "Heart Rate")

        with col2:
            stat_s = monitor.determine_status("Sys BP", sbp)
            stat_d = monitor.determine_status("Dia BP", dbp)
            final_stat = "critical" if "critical" in [stat_s, stat_d] else ("warning" if "warning" in [stat_s, stat_d] else "normal")
            bp_text = f"{sbp}/{dbp}" if (sbp is not None and dbp is not None) else "—"
            render_vital_card("Blood Pressure", bp_text, VITALS_CONFIG["Sys BP"]["unit"], final_stat, VITALS_CONFIG["Sys BP"]["color"])
            render_sparkline(df_history, "Sys BP")

        with col3:
            stat = monitor.determine_status("SpO2", spo2)
            render_vital_card("SpO2", spo2, VITALS_CONFIG["SpO2"]["unit"], stat, VITALS_CONFIG["SpO2"]["color"])
            render_sparkline(df_history, "SpO2")

        col4, col5, col6 = st.columns(3)
        with col4:
            stat = monitor.determine_status("Resp Rate", rr)
            render_vital_card("Resp Rate", rr, VITALS_CONFIG["Resp Rate"]["unit"], stat, VITALS_CONFIG["Resp Rate"]["color"])
            render_sparkline(df_history, "Resp Rate")

        with col5:
            stat = monitor.determine_status("Temp", temp)
            render_vital_card("Temperature", temp, VITALS_CONFIG["Temp"]["unit"], stat, VITALS_CONFIG["Temp"]["color"])
            render_sparkline(df_history, "Temp")

        with col6:
            st.markdown("### 🧠 AI Analysis")
            st.caption("Feature Contribution to Risk Score (mock)")
            # Mock SHAP using deviations from normal ranges
            shap_mock = {
                "High HR": round(max(0, (hr - VITALS_CONFIG["Heart Rate"]["normal"][1]) / 50) if hr is not None else 0, 3),
                "Low BP": round(max(0, (VITALS_CONFIG["Sys BP"]["normal"][0] - sbp) / 40) if sbp is not None else 0, 3),
                "Rapid Breathing": round(max(0, (rr - VITALS_CONFIG["Resp Rate"]["normal"][1]) / 15) if rr is not None else 0, 3)
            }
            shap_df = pd.DataFrame(list(shap_mock.items()), columns=["Feature", "Impact"]).set_index("Feature")
            st.bar_chart(shap_df)

    # Run the fragment and show historical logs (outside fragment)
    dashboard_fragment()

    st.markdown("---")
    with st.expander("📋 View Full Data Logs (Click to Expand)"):
        # Sort history by Timestamp for readable table
        df_display = st.session_state.patient_data.copy()
        if "Timestamp" in df_display.columns:
            df_display = df_display.sort_values("Timestamp", ascending=False)
        st.dataframe(df_display.reset_index(drop=True), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

    """ python -m streamlit run app.py"""