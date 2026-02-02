import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def build_model():
    # 1. Load Data
    df = pd.read_csv("ILPD.csv")
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Cleaning & Encoding
    df['gender'] = LabelEncoder().fit_transform(df['gender'].astype(str))
    df.fillna(df.median(numeric_only=True), inplace=True)

    # 3. Medical Ratios (Matches your app logic)
    df["DB_TB_Ratio"] = df["direct_bilirubin"] / (df["tot_bilirubin"] + 1e-8)
    df["AST_ALT_Ratio"] = df["sgot"] / (df["sgpt"] + 1e-8)
    df["ALP_ALT_Ratio"] = df["alkphos"] / (df["sgpt"] + 1e-8)
    df["Albumin_TP_Ratio"] = df["albumin"] / (df["tot_proteins"] + 1e-8)
    df["Age_Enzyme_Index"] = df["age"] * (df["sgot"] + df["sgpt"] + df["alkphos"])

    # 4. Severity Labeling
    severity = []
    for _, r in df.iterrows():
        if r["tot_bilirubin"] > 3.0 or r["albumin"] < 3.0 or (r["sgot"] > 200 and r["sgpt"] > 200):
            severity.append(2) # Severe
        elif r["tot_bilirubin"] > 1.5 or r["albumin"] < 3.5 or (r["sgot"] > 100 or r["sgpt"] > 100):
            severity.append(1) # Moderate
        else:
            severity.append(0) # Mild
    df["Severity"] = severity

    # 5. Training Setup
    X = df.drop(columns=["Severity"])
    y = df["Severity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Scaling (This creates the 'scaler' variable)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 7. Balancing & Stacking Model
    sm = SMOTEENN(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    
    model = StackingClassifier(
        estimators=[('rf', RandomForestClassifier()), ('knn', KNeighborsClassifier())],
        final_estimator=LogisticRegression()
    )
    model.fit(X_res, y_res)

    # 8. EXPORT FILES
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("Success: 'model.joblib' and 'scaler.joblib' have been created!")

if __name__ == "__main__":
    build_model()