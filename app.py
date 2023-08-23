import lightgbm as lgb
import streamlit as st
import numpy as np

# 加载AKD Prediction Model的LightGBM模型
akd_model = lgb.Booster(model_file='model.txt')

# 加载CKD Prediction Model的LightGBM模型
ckd_model = lgb.Booster(model_file='ckd_model.txt')

# 添加应用程序标题
st.title("Prediction Model for Nephrectomy")

# 用户选择预测 AKD 还是 CKD
prediction_type = st.radio("选择预测类型", ("AKD", "CKD"))

# 根据预测类型指定特征列
if prediction_type == "AKD":
    feature_columns = ['Operativeduration', 'Hb', 'Bloodloss', 'AKIGrade', 'Hct', 'ALB', 'SBP', 'BaslineeGFR', 'Anion_gap']
    feature_types = {'Operativeduration': 'number', 'Hb': 'number', 'Bloodloss': 'number', 'AKIGrade': 'selectbox',
                     'Hct': 'number', 'ALB': 'number', 'SBP': 'number', 'BaslineeGFR': 'number', 'Anion_gap': 'number'}
else:
    feature_columns = ['BaslineeGFR', 'Age', 'Pathology', 'AKI_and_AKD', 'TBIL', 'Mg', 'Scr', 'TG', 'WBC']
    feature_types = {'BaslineeGFR': 'number', 'Age': 'number', 'Pathology': 'selectbox', 'AKI_and_AKD': 'selectbox',
                     'TBIL': 'number', 'Mg': 'number', 'Scr': 'number', 'TG': 'number', 'WBC': 'number'}

# 收集输入特征值
input_features = {}
for column in feature_columns:
    if feature_types[column] == 'selectbox':
        feature_value = st.selectbox(column, [0, 1, 2, 3])
    else:
        feature_value = st.number_input(column, value=0.0 if feature_types[column] == 'number' else 0, step=0.01 if feature_types[column] == 'number' else 1, format="%.2f" if feature_types[column] == 'number' else None)
    input_features[column] = feature_value

# 创建一个按钮，当用户点击时进行预测
if st.button("预测"):
    # 将输入的特征值转换为数组
    features = np.array([[input_features[column] for column in feature_columns]])

    if prediction_type == "AKD":
        # 使用 AKD 模型进行预测
        prediction = akd_model.predict(features)
    else:
        # 使用 CKD 模型进行预测
        prediction = ckd_model.predict(features)
    
    st.write(f"预测结果: {prediction[1]}")
