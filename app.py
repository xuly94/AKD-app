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
    feature_constraints = {
        'Operativeduration': (0, float('inf')),
        'Hb': (0, float('inf')),
        'Bloodloss': (0, float('inf')),
        'AKIGrade': (0, 3),
        'Hct': (0, float('inf')),
        'ALB': (0, float('inf')),
        'SBP': (0, float('inf')),
        'BaslineeGFR': (0, float('inf')),
        'Anion_gap': (0, float('inf'))
    }
else:
    feature_columns = ['BaslineeGFR', 'Age', 'Pathology', 'AKI_and_AKD', 'TBIL', 'Mg', 'Scr', 'TG', 'WBC']
    feature_constraints = {
        'BaslineeGFR': (0, float('inf')),
        'Age': (0, float('inf')),
        'Pathology': (0, 2),
        'AKI_and_AKD': (0, 3),
        'TBIL': (0, float('inf')),
        'Mg': (0, float('inf')),
        'Scr': (0, float('inf')),
        'TG': (0, float('inf')),
        'WBC': (0, float('inf'))
    }

# 收集输入特征值
input_features = {}
for column in feature_columns:
    min_value, max_value = feature_constraints[column]
    if column in ['Hb', 'Bloodloss', 'SBP', 'Age']:
        feature_value = st.number_input(column, value=0, min_value=int(min_value), max_value=int(max_value), step=1)
    elif column in ['Urine_protein', 'AKIGrade', 'AKI_and_AKD', 'Pathology']:
        feature_value = st.selectbox(column, [0, 1, 2, 3])
    else:
        feature_value = st.number_input(column, value=0.0, min_value=float(min_value), max_value=float(max_value), step=0.01, format="%.2f")
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
