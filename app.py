import streamlit as st
import lightgbm as lgb
import numpy as np

# Load the AKD and CKD models
akd_model = lgb.Booster(model_file='akd_model.txt')
ckd_model = lgb.Booster(model_file='ckd_model.txt')

def predict_akd_probability(features):
    akd_prob = akd_model.predict([features])
    return akd_prob[0]

def predict_ckd_probability(features):
    ckd_prob = ckd_model.predict([features])
    return ckd_prob[0]

def main():
    st.title('AKD and CKD Probability Prediction')

    # User selects prediction type (AKD or CKD)
    prediction_type = st.radio("Select Prediction Type", ("AKD", "CKD"))

    # Feature input
    features = []

    if prediction_type == "AKD":
        st.subheader("AKD Features")

        urine_protein = st.selectbox("Urine Protein", [0, 1, 2, 3])
        aki_grade = st.selectbox("AKI Grade", [0, 1, 2, 3])
        hb = st.slider("Hb (integer)", 0, 100, step=1)
        bloodloss = st.slider("Blood Loss (integer)", 0, 100, step=1)
        sbp = st.slider("SBP (integer)", 0, 200, step=1)
        operativeduration = st.number_input("Operative Duration", value=0.0, format="%.2f")
        hct = st.number_input("Hct", value=0.0, format="%.2f")
        alb = st.number_input("ALB", value=0.0, format="%.2f")
        baseline_egfr = st.number_input("Baseline eGFR", value=0.0, format="%.2f")
        anion_gap = st.number_input("Anion Gap", value=0.0, format="%.2f")

        features.extend([operativeduration, hb, bloodloss, aki_grade, hct, alb, sbp, baseline_egfr, anion_gap])
    else:
        st.subheader("CKD Features")

        urine_protein = st.selectbox("Urine Protein", [0, 1, 2, 3])
        aki_and_akd = st.selectbox("AKI and AKD", [0, 1, 2, 3])
        pathology = st.selectbox("Pathology", [0, 1, 2])
        age = st.slider("Age (integer)", 0, 120, step=1)
        baseline_egfr = st.number_input("Baseline eGFR", value=0.0, format="%.2f")
        tbil = st.number_input("TBIL", value=0.0, format="%.2f")
        mg = st.number_input("Mg", value=0.0, format="%.2f")
        scr = st.number_input("Scr", value=0.0, format="%.2f")
        tg = st.number_input("TG", value=0.0, format="%.2f")
        wbc = st.number_input("WBC", value=0.0, format="%.2f")

        features.extend([baseline_egfr, age, pathology, aki_and_akd, tbil, mg, scr, tg, wbc])

    # Create a button to make predictions
    if st.button('Predict'):
        features_array = np.array(features).reshape(1, -1)

        if prediction_type == "AKD":
            akd_probability = predict_akd_probability(features_array)
            st.write(f'AKD Probability: {akd_probability:.2f}')
        else:
            ckd_probability = predict_ckd_probability(features_array)
            st.write(f'CKD Probability: {ckd_probability:.2f}')

if __name__ == '__main__':
    main()
