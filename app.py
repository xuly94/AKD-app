import streamlit as st
import lightgbm as lgb
import numpy as np

# Load the AKD and CKD models
akd_model = lgb.Booster(model_file='akd_model.txt')
ckd_model = lgb.Booster(model_file='ckd_model.txt')

# Mapping for Urine_protein and AKIGrade values
Urine_protein_mapping = {"-": 0, "1+": 1, "2+": 2, "3+": 3}
AKIGrade_mapping = {"Stage 0": 0, "Stage 1": 1, "Stage 2": 2, "Stage 3": 3}
Pathology_mapping = {"Benign": 0, "Malignant (Non-Clear)": 1, "Clear Cell": 2}
AKI_and_AKD_mapping = {"NKD": 0, "subacute AKD": 1, "AKI recover": 2,"AKD with AKI": 3}




def predict_akd_probability(features):
    akd_prob = akd_model.predict(features)
    return akd_prob[0]

def predict_ckd_probability(features):
    ckd_prob = ckd_model.predict(features)
    return ckd_prob[0]


def main():
    st.title('AKD and CKD Probability Prediction')

    # User selects prediction type (AKD or CKD)
    prediction_type = st.radio("Select Prediction Type", ("AKD Prediction", "CKD Prediction"))
    # Feature input
    features = []

    if prediction_type == "AKD Prediction":
        st.subheader("AKD Features")

        Operativeduration = st.number_input("Operative Duration (Hours)", value=0.0, format="%.2f") 
        Hb = st.number_input("Hb (g/L)", value=0, format="%d")
        Bloodloss = st.number_input("Blood Loss (ML)", value=0, format="%d")
        Urine_protein = st.selectbox("Urine Protein", ["-", "1+", "2+", "3+"])
        AKIGrade = st.selectbox("AKI Grade", ["Stage 0", "Stage 1", "Stage 2", "Stage 3"])
        Hct = st.number_input("Hct (%)", value=0.0, format="%.2f")
        ALB = st.number_input("ALB (g/L)", value=0.0, format="%.2f")
        SBP = st.number_input("SBP (mmHg)", value=0, format="%d")
        BaslineeGFR = st.number_input("Baseline eGFR (ml/min/1.73 m²)", value=0.0, format="%.2f")
        Anion_gap = st.number_input("Anion Gap (mmol/L)", value=0.0, format="%.2f")

        # Map AKIGrade back to 0, 1, 2, 3 for prediction
        Urine_protein_encoded = Urine_protein_mapping[Urine_protein]
        AKIGrade_encoded = AKIGrade_mapping[AKIGrade]

        features.extend([Operativeduration, Hb, Bloodloss, Urine_protein_encoded,AKIGrade_encoded, Hct, ALB, SBP, BaslineeGFR, Anion_gap])
    else:
        st.subheader("CKD Features")

        BaslineeGFR = st.number_input("Basline eGFR (ml/min/1.73 m²)", value=0.0, format="%.2f")
        Age = st.number_input("Age (Years)", value=0, format="%d")
        Pathology = st.selectbox("Pathology", ["Benign", "Malignant (Non-Clear)", "Clear Cell"])
        AKI_and_AKD = st.selectbox("Trajectories of renal function", ["NKD" ,"subacute AKD", "AKI recover","AKD with AKI"])
        TBIL = st.number_input("TBIL (umol/L)", value=0.0, format="%.2f")
        Mg = st.number_input("Mg (mmol/L)", value=0.0, format="%.2f")
        Scr = st.number_input("Scr (umol/L)", value=0.0, format="%.2f")
        TG = st.number_input("TG (mmol/L)", value=0.0, format="%.2f")
        WBC = st.number_input("WBC (10^9/L)", value=0.0, format="%.2f")
        Urine_protein = st.selectbox("Urine Protein", ["-", "1+", "2+", "3+"])
        
        Pathology_encoded = Pathology_mapping[Pathology]
        AKI_and_AKD_encoded = AKI_and_AKD_mapping[AKI_and_AKD]
        Urine_protein_encoded = Urine_protein_mapping[Urine_protein]

        features.extend([BaslineeGFR, Age, Pathology_encoded, AKI_and_AKD_encoded, TBIL, Mg, Scr, TG, WBC,Urine_protein_encoded])

    # Create a button to make predictions
    if st.button('Predict'):
        features_array = np.array(features).reshape(1, -1)

        if prediction_type == "AKD Prediction":
            akd_probability = predict_akd_probability(features_array)
            st.write(f'AKD Probability: {akd_probability:.2f}')
        else:
            ckd_probability = predict_ckd_probability(features_array)
            st.write(f'CKD Probability: {ckd_probability:.2f}')

if __name__ == '__main__':
    main()
