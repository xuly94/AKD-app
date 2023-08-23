import lightgbm as lgb
import pandas as pd
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
    feature_columns = ['Operativeduration','Hb','Bloodloss','Urine_protein','AKIGrade','Hct','ALB','SBP','BaslineeGFR','Anion_gap']
else:
    feature_columns = ['BaslineeGFR','Age','Pathology','AKI_and_AKD','TBIL','Mg','Scr','TG','WBC','Urine_protein']

# 添加特征输入小部件
input_features = []

for column in feature_columns:
    feature_value = st.number_input(column, value=0.0)
    input_features.append(feature_value)

# 创建一个按钮，当用户点击时进行预测
if st.button("预测"):
    # 将输入的特征值转换为数组
    features = np.array([input_features])

    if prediction_type == "AKD":
        # 使用 AKD 模型进行预测
        prediction = akd_model.predict(features)
    else:
        # 使用 CKD 模型进行预测
        prediction = ckd_model.predict(features)

    # 显示预测结果
    st.write(f"预测结果: {prediction[0]}")
