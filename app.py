import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.joblib")
        scaler = joblib.load("scaler.joblib")
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_assets()

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="LiverCare AI Diagnostics",
    page_icon="🏥",
    layout="wide"
)

# =====================================================
# PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>

/* ===============================
   GOOGLE FONT
================================= */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

/* ===============================
   APP BACKGROUND
================================= */

.stApp{
    background:#F4F8FC;
}

/* ===============================
   HEADER
================================= */

.header{
    background:linear-gradient(90deg,#2F80ED,#56CCF2);
    padding:28px;
    border-radius:18px;
    text-align:center;
    color:white;
    margin-bottom:25px;
    box-shadow:0px 8px 25px rgba(0,0,0,.15);
}

.header h1{
    color:white;
    font-size:42px;
    font-weight:700;
    margin:0;
}

.header p{
    color:#F8F9FA;
    font-size:18px;
    margin-top:10px;
}

/* ===============================
   CARD
================================= */

.card{
    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 6px 18px rgba(0,0,0,.08);
    margin-bottom:20px;
}

/* ===============================
   BUTTON
================================= */

.stButton>button{

    width:100%;
    height:55px;

    background:linear-gradient(90deg,#2F80ED,#4F9DDE);

    color:white !important;

    border:none;

    border-radius:10px;

    font-size:18px;

    font-weight:600;

    transition:0.3s;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#1F6FD0,#2F80ED);

    color:white !important;

    transform:scale(1.02);

}

/* ===============================
   LABELS
================================= */

label{

    color:#2C3E50 !important;

    font-weight:600 !important;

}

/* ===============================
   NUMBER INPUT
================================= */

.stNumberInput input{

    background:#FFFFFF !important;

    color:#000000 !important;

    -webkit-text-fill-color:#000000 !important;

    border:1px solid #D6E4F0 !important;

    border-radius:10px !important;

    font-size:16px !important;

}

/* ===============================
   TEXT INPUT
================================= */

.stTextInput input{

    background:#FFFFFF !important;

    color:#000000 !important;

    -webkit-text-fill-color:#000000 !important;

}

/* ===============================
   SELECT BOX
================================= */

.stSelectbox div[data-baseweb="select"]{

    background:#FFFFFF !important;

    color:#000000 !important;

    border-radius:10px;

    border:1px solid #D6E4F0 !important;

}

.stSelectbox div[data-baseweb="select"] *{

    color:#000000 !important;

}

/* ===============================
   FORCE INPUT TEXT BLACK
================================= */

input,
textarea,
select{

    color:#000000 !important;

    -webkit-text-fill-color:#000000 !important;

}

[data-baseweb="input"] input{

    color:#000000 !important;

    -webkit-text-fill-color:#000000 !important;

}

[data-baseweb="base-input"] input{

    color:#000000 !important;

    -webkit-text-fill-color:#000000 !important;

}

/* ===============================
   METRICS
================================= */

[data-testid="stMetric"]{

    background:white;

    border-radius:15px;

    padding:15px;

    box-shadow:0px 5px 15px rgba(0,0,0,.08);

}

/* ===============================
   RESULT CARD
================================= */

.result-box{

    padding:30px;

    border-radius:18px;

    color:white;

    font-size:30px;

    font-weight:bold;

    text-align:center;

    box-shadow:0px 8px 25px rgba(0,0,0,.15);

}

/* ===============================
   DATAFRAME
================================= */

[data-testid="stDataFrame"]{

    background:white;

    border-radius:15px;

}

thead tr th{

    background:#2F80ED !important;

    color:white !important;

}

tbody tr td{

    background:white !important;

    color:#2C3E50 !important;

}

/* ===============================
   SUCCESS / INFO / WARNING
================================= */

[data-testid="stSuccess"]{

    border-left:6px solid #27AE60;

}

[data-testid="stWarning"]{

    border-left:6px solid #F2994A;

}

[data-testid="stError"]{

    border-left:6px solid #EB5757;

}

[data-testid="stInfo"]{

    border-left:6px solid #2F80ED;

}

/* ===============================
   IMAGES
================================= */

img{

    border-radius:15px;

}

/* ===============================
   REMOVE STREAMLIT MENU
================================= */

#MainMenu{

    visibility:hidden;

}

footer{

    visibility:hidden;

}

header{

    visibility:hidden;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""

<div class="header">

<h1>🏥 LiverCare AI Diagnostics</h1>

<p>
AI Powered Clinical Decision Support System for Liver Disease Assessment
</p>

</div>

""", unsafe_allow_html=True)

# =====================================================
# HOSPITAL IMAGE
# =====================================================

st.image(
    "https://images.unsplash.com/photo-1586773860418-d3b97978c65c?auto=format&fit=crop&w=1400&q=80",
    use_container_width=True,
)

# =====================================================
# MAIN LAYOUT
# =====================================================

left, right = st.columns([1,2])

# =====================================================
# LEFT PANEL
# =====================================================

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(
        "https://t3.ftcdn.net/jpg/07/69/55/88/240_F_769558846_l3UvsH8MGsMTuvCY3FiF1Z1AzCuvAEfx.jpg",
        width=220
    )

    st.markdown("## Clinical Evaluation")

    st.info("""

### AI evaluates

✔ Bilirubin Balance

✔ Albumin Production

✔ AST / ALT Ratio

✔ Liver Function

✔ Protein Synthesis

✔ Metabolic Load

""")

    st.image(
        "https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&w=600&q=80",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# RIGHT PANEL
# =====================================================

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("🧪 Laboratory Test Inputs")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Patient Age",
            1,
            100,
            45
        )

        gender = st.selectbox(
            "Biological Sex",
            ["Male","Female"]
        )

        tb = st.number_input(
            "Total Bilirubin (mg/dL)",
            0.1,
            75.0,
            1.0
        )

        db = st.number_input(
            "Direct Bilirubin (mg/dL)",
            0.1,
            20.0,
            0.3
        )

        alkphos = st.number_input(
            "Alkaline Phosphatase (U/L)",
            10.0,
            3000.0,
            150.0
        )
    with col2:

        tp = st.number_input(
            "Total Proteins (g/dL)",
            1.0,
            10.0,
            6.5
        )

        alb = st.number_input(
            "Albumin Level (g/dL)",
            1.0,
            6.0,
            3.5
        )

        ag_ratio = st.number_input(
            "A/G Ratio",
            0.1,
            3.0,
            1.0
        )

        sgpt = st.number_input(
            "SGPT (ALT) (U/L)",
            1.0,
            2000.0,
            40.0
        )

        sgot = st.number_input(
            "SGOT (AST) (U/L)",
            1.0,
            2000.0,
            40.0
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION BUTTON
# =====================================================

predict = st.button(
    "🔬 RUN AI DIAGNOSTIC ANALYSIS",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    if model is None or scaler is None:

        st.error(
            "❌ model.joblib or scaler.joblib not found."
        )

    else:

        gender_val = 1 if gender == "Male" else 0

        db_tb = db / (tb + 1e-8)

        ast_alt = sgot / (sgpt + 1e-8)

        alp_alt = alkphos / (sgpt + 1e-8)

        alb_tp = alb / (tp + 1e-8)

        age_enzyme = age * (
            sgot +
            sgpt +
            alkphos
        )

        features = np.array([[
            age,
            gender_val,
            tb,
            db,
            alkphos,
            sgpt,
            sgot,
            tp,
            alb,
            ag_ratio,
            1,
            db_tb,
            ast_alt,
            alp_alt,
            alb_tp,
            age_enzyme
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

# =====================================================
# RESULT COLORS
# =====================================================

        severity = {

            0: "MILD (Normal)",

            1: "MODERATE (Observe)",

            2: "SEVERE (Urgent)"

        }

        colors = {

            0: "#27AE60",

            1: "#F2994A",

            2: "#EB5757"

        }

# =====================================================
# RESULT CARD
# =====================================================

        st.markdown(

            f"""

            <div class="result-box"

            style="background:{colors[prediction]};">

            🏥 AI DIAGNOSIS

            <br><br>

            {severity[prediction]}

            </div>

            """,

            unsafe_allow_html=True

        )

# =====================================================
# DASHBOARD METRICS
# =====================================================

        st.write("")

        st.subheader("📈 Clinical Dashboard")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(

                "AST / ALT Ratio",

                f"{ast_alt:.2f}"

            )

        with m2:

            st.metric(

                "Total Bilirubin",

                f"{tb:.2f} mg/dL"

            )

        with m3:

            st.metric(

                "Albumin",

                f"{alb:.2f} g/dL"

            )
# =====================================================
# CLINICAL SUMMARY
# =====================================================

        st.write("")

        st.subheader("📊 Clinical Feature Summary")

        summary_table = pd.DataFrame({

            "Medical Parameter": [

                "AST / ALT Ratio",

                "Direct / Total Bilirubin",

                "Albumin / Total Protein",

                "ALP / ALT Ratio",

                "Age Enzyme Load"

            ],

            "Calculated Value": [

                f"{ast_alt:.2f}",

                f"{db_tb:.2f}",

                f"{alb_tp:.2f}",

                f"{alp_alt:.2f}",

                f"{age_enzyme:.1f}"

            ],

            "Clinical Target": [

                "Hepatocellular Injury",

                "Biliary Function",

                "Protein Synthesis",

                "Liver Enzyme Activity",

                "Overall Metabolic Load"

            ]

        })

        st.dataframe(

            summary_table,

            hide_index=True,

            use_container_width=True

        )

# =====================================================
# CLINICAL INTERPRETATION
# =====================================================

        st.write("")

        st.subheader("🩺 AI Clinical Interpretation")

        if prediction == 2:

            st.error("""

### 🚨 High Risk Detected

The laboratory values indicate a significant possibility of liver dysfunction.

**Recommended Actions**

• Immediate consultation with a Hepatologist

• Repeat Liver Function Test (LFT)

• Complete Blood Count (CBC)

• Ultrasound Abdomen

• Viral Hepatitis Screening

• Clinical Examination

""")

        elif prediction == 1:

            st.warning("""

### ⚠ Moderate Risk

Some liver biomarkers are outside the normal range.

**Suggested Follow-up**

• Repeat Liver Function Test within 7 days

• Maintain hydration

• Avoid alcohol

• Follow a healthy low-fat diet

• Consult physician if symptoms persist

""")

        else:

            st.success("""

### ✅ Low Risk

Current laboratory findings are within acceptable limits.

**Recommendations**

• Continue healthy lifestyle

• Routine annual health check-up

• Balanced diet

• Regular exercise

• Stay hydrated

""")

# =====================================================
# DASHBOARD DETAILS
# =====================================================

        st.write("")

        st.subheader("📋 Laboratory Overview")

        c1, c2 = st.columns(2)

        with c1:

            st.info(f"""

### Patient Information

**Age:** {age} Years

**Gender:** {gender}

**Total Protein:** {tp:.2f} g/dL

**Albumin:** {alb:.2f} g/dL

**A/G Ratio:** {ag_ratio:.2f}

""")

        with c2:

            st.info(f"""

### Liver Enzymes

**Total Bilirubin:** {tb:.2f}

**Direct Bilirubin:** {db:.2f}

**ALP:** {alkphos:.2f}

**SGPT (ALT):** {sgpt:.2f}

**SGOT (AST):** {sgot:.2f}

""")
# =====================================================
# AI HEALTH SCORE
# =====================================================

        st.write("")

        st.subheader("📈 AI Health Score")

        if prediction == 0:
            score = 92
            color = "green"

        elif prediction == 1:
            score = 63
            color = "orange"

        else:
            score = 28
            color = "red"

        st.progress(score / 100)

        st.markdown(
            f"""
            <h3 style="text-align:center;color:{color};">
            Overall Liver Health Score : {score}/100
            </h3>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# ADDITIONAL RECOMMENDATIONS
# =====================================================

        st.write("")
        st.subheader("💡 Lifestyle Recommendations")

        colA, colB, colC = st.columns(3)

        with colA:

            st.success("""
### 🥗 Diet

✔ Fresh Fruits

✔ Green Vegetables

✔ High Fiber Food

✔ Low Fat Diet

✔ Adequate Protein
""")

        with colB:

            st.info("""
### 🚶 Lifestyle

✔ Daily Walking

✔ Exercise

✔ Good Sleep

✔ Stress Management

✔ Stay Hydrated
""")

        with colC:

            st.warning("""
### 🚫 Avoid

✖ Alcohol

✖ Smoking

✖ Junk Food

✖ Self Medication

✖ Sugary Drinks
""")

# =====================================================
# HOSPITAL CONTACT
# =====================================================

        st.write("")
        st.markdown("---")

        st.markdown(
            """
<div class="card">

<h3 style="color:#2F80ED;">
🏥 LiverCare AI Diagnostic Center
</h3>

<b>Services</b>

<ul>

<li>AI Liver Disease Screening</li>

<li>Clinical Decision Support</li>

<li>Laboratory Analysis</li>

<li>Risk Assessment</li>

<li>Patient Monitoring</li>

</ul>

</div>
""",
            unsafe_allow_html=True
        )

# =====================================================
# DISCLAIMER
# =====================================================

st.markdown("---")

st.info(
"""
### ⚕ Medical Disclaimer

This AI application is intended only for educational and decision-support
purposes.

It **does not replace** a qualified physician, hepatologist,
or laboratory diagnosis.

Always consult a licensed medical professional before making
any healthcare decisions.
"""
)

# =====================================================
# FOOTER
# =====================================================

st.markdown(
"""
<div style='text-align:center;
padding:20px;
color:#6c757d;
font-size:16px;'>

Made with ❤️ using
<b>Streamlit</b>,
<b>Machine Learning</b>,
<b>NumPy</b>,
<b>Pandas</b>

<br><br>

© 2026 LiverCare AI Diagnostics

</div>
""",
unsafe_allow_html=True
)
