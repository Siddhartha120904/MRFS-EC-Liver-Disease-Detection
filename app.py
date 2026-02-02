import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load your model and scaler
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except:
        return None, None

model, scaler = load_assets()

# --- PAGE CONFIG ---
st.set_page_config(page_title="LiverCare AI - Diagnostic Portal", layout="wide", page_icon="🏥")

# --- ADVANCED CSS FOR UI/UX ---
st.markdown("""
    <style>
    /* Main background with a subtle medical gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Professional Hospital Header */
    .header-banner {
        background-color: #004a99;
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }

    /* Glassmorphism Effect for Input Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }

    /* Result Card Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-box {
        animation: fadeIn 0.8s ease-out;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 30px;
        color: white;
        font-size: 28px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Styled Button */
    .stButton>button {
        background-color: #004a99;
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #002d5e;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("""
    <div class="header-banner">
        <h1>🏥 LIVERCARE AI-DIAGNOSTICS</h1>
        <p>Advanced Clinical Intelligence for Hepatology Analysis</p>
    </div>
    """, unsafe_allow_html=True)



# --- MAIN INTERFACE ---
col_img, col_form = st.columns([1, 2])

with col_img:
    st.image("https://cdn-icons-png.flaticon.com/512/508/508791.png", width=200) # Liver Icon
    st.markdown("### Clinical Indicators")
    st.info("""
    **Patient Evaluation Guide:**
    - High Bilirubin (>1.5) indicates jaundice risk.
    - Low Albumin (<3.5) indicates synthetic failure.
    - AST/ALT ratio is critical for alcoholic vs non-alcoholic diagnosis.
    """)
    

with col_form:
    st.subheader("Laboratory Results Entry")
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Patient Age", 1, 100, 45)
        gender = st.selectbox("Biological Sex", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin", 0.1, 75.0, 1.0)
        db = st.number_input("Direct Bilirubin", 0.1, 20.0, 0.3)
        alkphos = st.number_input("Alkaline Phosphatase", 10.0, 3000.0, 150.0)
    
    with c2:
        tp = st.number_input("Total Proteins", 1.0, 10.0, 6.5)
        alb = st.number_input("Albumin Level", 1.0, 6.0, 3.5)
        ag_ratio = st.number_input("A/G Ratio", 0.1, 3.0, 1.0)
        sgpt = st.number_input("SGPT (ALT)", 1.0, 2000.0, 40.0)
        sgot = st.number_input("SGOT (AST)", 1.0, 2000.0, 40.0)

# --- PREDICTION ---
if st.button("RUN AI DIAGNOSTIC ANALYSIS", use_container_width=True):
    if model and scaler:
        # Engineering
        gender_val = 1 if gender == "Male" else 0
        db_tb = db / (tb + 1e-8)
        ast_alt = sgot / (sgpt + 1e-8)
        alp_alt = alkphos / (sgpt + 1e-8)
        alb_tp = alb / (tp + 1e-8)
        age_enzyme = age * (sgot + sgpt + alkphos)
        
        # Array (Matches your 16-feature model_trainer.py)
        features = np.array([[
            age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio,
            1, db_tb, ast_alt, alp_alt, alb_tp, age_enzyme
        ]])
        
        # Process
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # UI Results
        severity = {0: "MILD (Normal)", 1: "MODERATE (Observe)", 2: "SEVERE (Urgent)"}
        colors = {0: "#28a745", 1: "#f39c12", 2: "#e74c3c"}
        
        st.markdown(f"""
            <div class="result-box" style="background-color: {colors[prediction]};">
                DIAGNOSIS: {severity[prediction]}
            </div>
            """, unsafe_allow_html=True)
        
        if prediction == 2:
            st.warning("⚠️ High Clinical Alert: Immediate specialist consultation recommended.")
        elif prediction == 1:
            st.info("ℹ️ Follow-up tests required within 14 days.")
        else:
            st.success("✅ Results within acceptable clinical parameters.")
    else:
        st.error("Model assets not loaded.")
        