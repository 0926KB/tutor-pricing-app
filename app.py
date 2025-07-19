# app.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import streamlit as st

# 테스트용 샘플 데이터 (survey_clean.csv 대신)
data = {
    'experience_label': ['A', 'B', 'C', 'A'],
    'major_match_label': ['A', 'B', 'C', 'D'],
    'is_primary_job_label': ['A', 'B', 'A', 'B'],
    'teaching_mode_label': ['A', 'B', 'A', 'B'],
    'material_label': ['A', 'B', 'A', 'B'],
    'students_total_label': [30, 20, 50, 15],
    'univ_tier_numeric': [3, 2, 1, 0],
    'age_final': [1, 0, 0, 1],
    'age_adult': [0, 1, 0, 0],
    'age_high': [1, 1, 0, 0],
    'age_middle': [0, 0, 1, 0],
    'age_elementary': [0, 0, 1, 1],
    'log_hourly_won_clean': [np.log1p(40000), np.log1p(35000), np.log1p(50000), np.log1p(30000)]
}
df = pd.DataFrame(data)

# 피처 분류
categorical = ["experience_label", "major_match_label", "is_primary_job_label", "teaching_mode_label", "material_label"]
numeric = ["students_total_label", "univ_tier_numeric"]
multi = ["age_final", "age_adult", "age_high", "age_middle", "age_elementary"]
target = "log_hourly_won_clean"

# 파이프라인 정의
X = df[categorical + numeric + multi]
y = df[target]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", SimpleImputer(strategy="mean"), numeric)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
pipeline.fit(X, y)

# Streamlit App
st.title("🧠 시급 예측 MVP")

exp = st.selectbox("경력", ["A", "B", "C", "D"])
match = st.selectbox("전공 연관성", ["A", "B", "C", "D", "E"])
jobtype = st.selectbox("전업/부업", ["A", "B"])
mode = st.selectbox("수업 방식", ["A", "B"])  # "대면", "비대면" 대신 임시 label 사용
material = st.selectbox("자료 제공 유형", ["A", "B"])
students = st.number_input("누적 학생 수", min_value=0, step=1)
tier = st.slider("대학 티어 (3=A, 0=D)", 0, 3)

st.write("학생 연령대")
age_final = st.checkbox("입시 준비생")
age_adult = st.checkbox("성인")
age_high = st.checkbox("고등학생")
age_middle = st.checkbox("중학생")
age_elementary = st.checkbox("초등학생")

if st.button("예측 실행"):
    input_data = pd.DataFrame([{
        "experience_label": exp,
        "major_match_label": match,
        "is_primary_job_label": jobtype,
        "teaching_mode_label": mode,
        "material_label": material,
        "students_total_label": students,
        "univ_tier_numeric": tier,
        "age_final": int(age_final),
        "age_adult": int(age_adult),
        "age_high": int(age_high),
        "age_middle": int(age_middle),
        "age_elementary": int(age_elementary)
    }])
    pred_log = pipeline.predict(input_data)[0]
    pred_won = int(np.expm1(pred_log))
    st.success(f"💰 예측된 시급: ₩{pred_won:,}")
