import lightgbm as lgb
import streamlit as st
import numpy as np

# 加载AKD Prediction Model的LightGBM模型
akd_model = lgb.Booster(model_file='akd_model.txt')

# 加载CKD Prediction Model的LightGBM模型
ckd_model = lgb.Booster(model_file='ckd_model.txt')

# 添加应用程序标题
st.title("Prediction Model for Nephrectomy")

# 用户选择预测 AKD 还是 CKD
prediction_type = st.radio("选择预测类型", ("AKD", "CKD"))

# 根据预测类型指定特征列
if prediction_type == "AKD":
    feature_columns = ['Operativeduration', 'Hb', 'Bloodloss', 'AKIGrade', 'Hct', 'ALB', 'SBP', 'BaslineeGFR', 'Anion_gap']
else:
    feature_columns = ['BaslineeGFR', 'Age', 'Pathology', 'AKI_and_AKD', 'TBIL', 'Mg', 'Scr', 'TG', 'WBC']

# 创建输入特征的输入框
features = []
for feature_name in feature_columns:
    feature_value = st.number_input(f'Enter {feature_name}', min_value=0.0, max_value=100.0, step=0.1)
    features.append(feature_value)

# 创建一个按钮用于进行预测
if st.button('Predict'):
    features_array = np.array(features).reshape(1, -1)
    
    if prediction_type == "AKD":
        akd_probability = akd_model.predict(features_array)
        st.write(f'AKD Probability: {akd_probability[1]:.2f}')
    else:
        ckd_probability = ckd_model.predict(features_array)
        st.write(f'CKD Probability: {ckd_probability[1]:.2f}')
