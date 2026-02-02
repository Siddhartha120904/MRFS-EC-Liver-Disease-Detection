# --- PREDICTION ---
if st.button("RUN AI DIAGNOSTIC ANALYSIS", use_container_width=True):
    if model and scaler:
        # 1. Feature Engineering
        gender_val = 1 if gender == "Male" else 0
        db_tb = db / (tb + 1e-8)
        ast_alt = sgot / (sgpt + 1e-8)
        alp_alt = alkphos / (sgpt + 1e-8)
        alb_tp = alb / (tp + 1e-8)
        age_enzyme = age * (sgot + sgpt + alkphos)
        
        # 2. Arrange features (Matches your 16-feature model)
        features = np.array([[
            age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio,
            1, db_tb, ast_alt, alp_alt, alb_tp, age_enzyme
        ]])
        
        # 3. Model Prediction
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        # 4. Display Severity Card
        severity = {0: "MILD (Normal)", 1: "MODERATE (Observe)", 2: "SEVERE (Urgent)"}
        colors = {0: "#28a745", 1: "#f39c12", 2: "#e74c3c"}
        
        st.markdown(f"""
            <div class="result-box" style="background-color: {colors[prediction]};">
                DIAGNOSIS: {severity[prediction]}
            </div>
            """, unsafe_allow_html=True)

        # 5. NEW: PROFESSIONAL SUMMARY TABLE
        st.markdown("### 📊 Clinical Feature Summary")
        
        # Create a dictionary for the table
        summary_data = {
            "Medical Parameter": [
                "Bilirubin Ratio (DB/TB)", 
                "De Ritis Ratio (AST/ALT)", 
                "Synthetic Function (Alb/TP)", 
                "Age-Enzyme Index", 
                "A/G Ratio"
            ],
            "Calculated Value": [
                f"{db_tb:.3f}", 
                f"{ast_alt:.3f}", 
                f"{alb_tp:.3f}", 
                f"{age_enzyme:.1f}", 
                f"{ag_ratio:.2f}"
            ],
            "Clinical Significance": [
                "Biliary obstruction marker",
                "Liver injury patterns",
                "Protein synthesis health",
                "Enzyme-age correlation",
                "Protein balance"
            ]
        }
        
        # Convert to DataFrame and display
        df_summary = pd.DataFrame(summary_data)
        
        # Styling the table with Streamlit
        st.table(df_summary)

        # 6. Clinical Guidance
        if prediction == 2:
            st.error("🚨 **High Alert:** Markers indicate significant liver dysfunction. Immediate medical intervention is advised.")
        elif prediction == 1:
            st.warning("⚠️ **Moderate Risk:** Significant elevations noted. Schedule a follow-up with a hepatologist.")
        else:
            st.success("✅ **Stable:** Clinical markers are currently within low-risk thresholds.")
            
    else:
        st.error("Model assets not loaded. Check model.joblib and scaler.joblib.")
