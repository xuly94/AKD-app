import lightgbm as lgb
import pandas as pd
import streamlit as st
import numpy as np


# 加载训练好的LightGBM模型
model = lgb.Booster(model_file='model.txt')

# 添加应用程序标题
st.title("AKD Prediction Model for Nephrectomy")

# 读取特征列名（假设您有一个存储特征列名的文件或通过其他方式获取）
# 在这里我将特征列名直接写在代码中，您可以根据需要进行修改
feature_columns = ['Operativeduration','Hb','Bloodloss','Urine_protein','AKIGrade','Hct','ALB','SBP','BaslineeGFR','Anion_gap']

# 添加特征输入小部件
input_features = []

for column in feature_columns:
    feature_value = st.number_input(column, value=0.0)
    input_features.append(feature_value)

# 创建一个按钮，当用户点击时进行预测
if st.button("预测"):
    # 将输入的特征值转换为数组
    features = np.array([input_features])

    # 使用模型进行预测
    prediction = model.predict(features)

    # 显示预测结果
    st.write(f"预测结果: {prediction[0]}")