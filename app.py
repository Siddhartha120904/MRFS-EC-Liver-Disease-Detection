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

# --- ADVANCED CSS FOR STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    .header-banner {
        background-color: #004a99;
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    }
    .result-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        color: white;
        font-size: 26px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .stTable {
        background-color: white;
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
    <div class="header-banner">
        <h1>🏥 LIVERCARE AI-DIAGNOSTICS</h1>
        <p>Advanced Clinical Intelligence for Hepatology Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# --- HOSPITAL IMAGE SECTION ---
st.image("https://images.unsplash.com/photo-1586773860418-d3b97978c65c?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
         caption="City Central Hospital Integrated AI Lab", use_container_width=True)



# --- MAIN INTERFACE ---
col_img, col_form = st.columns([1, 2])

with col_img:
    st.image("https://cdn-icons-png.flaticon.com/512/508/508791.png", width=180) # Liver Icon
    st.markdown("### Clinical Evaluation")
    st.info("""
    **Hepatology Metrics:**
    - AI analyzes 16 unique data points.
    - Focuses on Bilirubin/Albumin balance.
    - Evaluates AST/ALT enzyme patterns.
    """)
    # Liver Anatomy Image
    st.image("https://images.unsplash.com/photo-1576086213369-97a306d36557?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80", 
             caption="Liver Anatomy Structure", width=300)



with col_form:
    st.subheader("Laboratory Data Entry")
    c1, c2 = st.columns(2)
    
    with c1:
        age = st.number_input("Patient Age", 1, 100, 45)
        gender = st.selectbox("Biological Sex", ["Male", "Female"])
        tb = st.number_input("Total Bilirubin (mg/dL)", 0.1, 75.0, 1.0)
        db = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 20.0, 0.3)
        alkphos = st.number_input("Alkaline Phosphatase (U/L)", 10.0, 3000.0, 150.0)
    
    with c2:
        tp = st.number_input("Total Proteins (g/dL)", 1.0, 10.0, 6.5)
        alb = st.number_input("Albumin Level (g/dL)", 1.0, 6.0, 3.5)
        ag_ratio = st.number_input("A/G Ratio", 0.1, 3.0, 1.0)
        sgpt = st.number_input("SGPT (ALT) (U/L)", 1.0, 2000.0, 40.0)
        sgot = st.number_input("SGOT (AST) (U/L)", 1.0, 2000.0, 40.0)

# --- PREDICTION AND TABLE ---
if st.button("RUN AI DIAGNOSTIC ANALYSIS", use_container_width=True):
    if model and scaler:
        # Engineering (16 features)
        gender_val = 1 if gender == "Male" else 0
        db_tb = db / (tb + 1e-8)
        ast_alt = sgot / (sgpt + 1e-8)
        alp_alt = alkphos / (sgpt + 1e-8)
        alb_tp = alb / (tp + 1e-8)
        age_enzyme = age * (sgot + sgpt + alkphos)
        
        features = np.array([[
            age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio,
            1, db_tb, ast_alt, alp_alt, alb_tp, age_enzyme
        ]])
        
        # Scaling and Prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # UI Setup
        severity = {0: "MILD (Normal)", 1: "MODERATE (Observe)", 2: "SEVERE (Urgent)"}
        colors = {0: "#28a745", 1: "#f39c12", 2: "#e74c3c"}
        
        # 1. Result Diagnosis Card
        st.markdown(f"""
            <div class="result-box" style="background-color: {colors[prediction]};">
                DIAGNOSIS: {severity[prediction]}
            </div>
            """, unsafe_allow_html=True)
        
        # 2. Result Summary Table
        st.markdown("### 📊 Clinical Feature Summary")
        
        summary_table = pd.DataFrame({
            "Medical Parameter": ["AST/ALT Ratio", "Bilirubin Index", "Synthetic Marker", "Enzyme Load"],
            "Calculated Value": [f"{ast_alt:.2f}", f"{db_tb:.2f}", f"{alb_tp:.2f}", f"{age_enzyme:.1f}"],
            "Clinical Target": ["Cellular Injury", "Biliary Function", "Protein Synthesis", "Metabolic Load"]
        })
        
        st.table(summary_table) # Displaying the table with column names

        # 3. Clinical Recommendation
        if prediction == 2:
            st.error("🚨 **Immediate Attention:** Critical levels detected. Refer to Hepatology Department immediately.")
        elif prediction == 1:
            st.warning("⚠️ **Clinical Advisory:** Moderate markers detected. Suggest blood test repetition in 7 days.")
        else:
            st.success("✅ **Patient Stable:** Results are within acceptable clinical parameters.")
    else:
        st.error("Model assets not loaded. Please ensure model.joblib and scaler.joblib are in the directory.")
