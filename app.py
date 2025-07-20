# app.py

import pandas as pd
import numpy as np
import re
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =============================
# STEP 1: 데이터 정제 및 저장
# =============================

RAW_FILE = "survey.csv"

# =============================
# 전역 유틸 함수 (대학 이름 정규화 및 티어 분류)
# =============================
UNIV_KEYWORDS = {
    "서울대학교": ["서울대학교", "서울대", "snu"],
    "연세대학교": ["연세대학교", "연세대", "yonsei"],
    "고려대학교": ["고려대", "고려대학교"],
    "성균관대학교": ["성균관대학교", "성균관대"],
    "한양대학교": ["한양대", "한양대학교"],
    "경희대학교": ["경희대", "경희대학교"],
    "이화여자대학교": ["이화여자대학교", "이화여대"],
    "기타": []
}

TIER_MAP = {
    "서울대학교": 3, "연세대학교": 3, "고려대학교": 3,
    "성균관대학교": 2, "한양대학교": 2, "경희대학교": 2, "이화여자대학교": 2,
    "기타": 0
}

def normalize_univ_name(raw):
    if not raw: return "기타"
    text = re.sub(r"[^\w\s]", "", str(raw).lower())
    for std, kws in UNIV_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text:
                return std
    return "기타"

def tier_from_univ(name):
    if name in ["서울대학교", "연세대학교", "고려대학교"]: return "A"
    elif name in ["성균관대학교", "한양대학교", "경희대학교", "이화여자대학교"]: return "B"
    else: return "D"

@st.cache_data
def clean_data():
    df = pd.read_csv(RAW_FILE)

    def clean_hourly_won(text):
        if pd.isna(text): return np.nan
        text = str(text).strip()
        if text.replace(".", "", 1).isdigit():
            val = float(text)
            if val <= 10: return int(val * 10000)
            elif val < 300: return int(val * 100)
            else: return int(val)
        numbers = [int(n.replace(",", "")) for n in re.findall(r"\d{2,3},?\d{3}", text)]
        numbers = [n for n in numbers if n >= 1000]
        return int(np.mean(numbers)) if numbers else np.nan

    df["hourly_won_clean"] = df["본인의 시급을 입력해주세요. (원)"].apply(clean_hourly_won)
    df["log_hourly_won_clean"] = df["hourly_won_clean"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

    score_to_label = {"1": "E", "2": "D", "3": "C", "4": "B", "5": "A"}
    df["major_match_label"] = df["전공과 가르치는 과목이 얼마나 관련이 있나요?"].astype(str).map(score_to_label)

    def normalize_experience(val):
        if pd.isna(val): return None
        s = val.strip().replace(" ", "")
        if "1년이하" in s: return "1년 이하"
        if "1~2년" in s or "1년~2년" in s: return "1~2년"
        if "3~5년" in s or "3-5년" in s: return "3~5년"
        if "6년이상" in s: return "6년 이상"
        return s

    df["experience_years_normalized"] = df["과외 경력은 얼마나 되시나요?"].apply(normalize_experience)
    df["experience_label"] = df["experience_years_normalized"].map({
        "6년 이상": "A", "3~5년": "B", "1~2년": "C", "1년 이하": "D"
    })

    df["is_primary_job_label"] = df["현재 과외 활동은 본업인가요, 부업인가요?"].map({
        "전업 (과외가 주 수입원)": "A", "부업 (본업이 따로 있음)": "B"
    })

    df["teaching_mode_label"] = df["수업 방식은 어떻게 진행하시나요?"].map({
        "대면": "A", "대면과 비대면 모두 가능": "A", "비대면": "B"
    })

    def label_material_type(cell):
        if pd.isna(cell): return "B"
        return "A" if "자체 제작" in str(cell) else "B"

    df["material_label"] = df["수업 중 제공하는 자료 유형을 모두 선택해주세요."].apply(label_material_type)

    def clean_student_count(text):
        if pd.isna(text): return np.nan
        cleaned = re.sub(r"[^\d]", "", str(text))
        return int(cleaned) if cleaned else np.nan

    df["students_total_label"] = df["지금까지 가르친 학생 수는 몇 명인가요? (숫자만 입력해주세요)"].apply(clean_student_count)

    df["univ_normalized"] = df["학교명을 기입해주세요.(재학중 포함)"].apply(normalize_univ_name)
    df["univ_tier"] = df["univ_normalized"].apply(tier_from_univ)
    df["univ_tier_numeric"] = df["univ_tier"].map({"A":3,"B":2,"C":1,"D":0})

    AGE_LABELS = {
        "파이널 입시 준비생": "age_final",
        "성인":               "age_adult",
        "고등학생":           "age_high",
        "중학생":             "age_middle",
        "초등학생":           "age_elementary"
    }

    def multihot(cell):
        if pd.isna(cell): return []
        items = str(cell).split(", ")
        return [AGE_LABELS[i] for i in items if i in AGE_LABELS]

    df["student_age_label"] = df["가르치는 학생 연령대를 선택해주세요."].apply(multihot)
    for lbl in AGE_LABELS.values():
        df[lbl] = df["student_age_label"].apply(lambda lst: int(lbl in lst))

    df.to_csv("survey_clean.csv", index=False)
    return df

df = clean_data()
df = df.dropna(subset=["log_hourly_won_clean"])

# =============================
# STEP 2: 모델 학습
# =============================
categorical = ["experience_label", "major_match_label", "is_primary_job_label", "teaching_mode_label", "material_label"]
numeric = ["students_total_label", "univ_tier_numeric"]
multi = ["age_final", "age_adult", "age_high", "age_middle", "age_elementary"]
target = "log_hourly_won_clean"

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

# =============================
# STEP 3: Streamlit UI
# =============================
st.title("📊 프리랜서 강사 시급 예측기")

# 사용자에게 보일 선택지 (한국어)
# 1. 경력 (경력 년수 → A~D)
exp_options = {
    "6년 이상": "A",
    "3~5년": "B",
    "1~2년": "C",
    "1년 이하": "D"
}
exp_kor = st.selectbox("과외 경력", list(exp_options.keys()))
exp = exp_options[exp_kor]

# 2. 전공 연관성 (매우 높음~전혀 없음 → A~E)
match_options = {
    "매우 높음": "A",
    "높음": "B",
    "보통": "C",
    "낮음": "D",
    "전혀 없음": "E"
}
match_kor = st.selectbox("전공과 과외 과목의 연관성", list(match_options.keys()))
match = match_options[match_kor]

# 3. 전업/부업
jobtype_options = {
    "전업 (과외가 주 수입원)": "A",
    "부업 (본업이 따로 있음)": "B"
}
jobtype_kor = st.selectbox("과외 활동 유형", list(jobtype_options.keys()))
jobtype = jobtype_options[jobtype_kor]

# 4. 수업 방식
mode_options = {
    "대면 / 대면+비대면 모두 가능": "A",
    "비대면만 가능": "B"
}
mode_kor = st.selectbox("수업 방식", list(mode_options.keys()))
mode = mode_options[mode_kor]

# 5. 자료 제공 유형
material_options = {
    "자체 제작 자료 제공": "A",
    "기성 자료 또는 미제공": "B"
}
material_kor = st.selectbox("자료 유형", list(material_options.keys()))
material = material_options[material_kor]

# 6. 누적 학생 수
students = st.number_input("누적 학생 수", min_value=0, step=1)

# 7. 대학 티어 (슬라이더 그대로 유지)
school_input = st.text_input("학교명을 입력하세요 (예: 서울대, 연세대학교 등)")
normalized_name = normalize_univ_name(school_input)
tier = TIER_MAP.get(normalized_name, 0)
st.write(f"🎓 분류된 대학 티어: {tier} (학교명: {normalized_name})")

# 8. 연령대 (체크박스 그대로 유지)
st.write("학생 연령대")
age_final = st.checkbox("입시 준비생")
age_adult = st.checkbox("성인")
age_high = st.checkbox("고등학생")
age_middle = st.checkbox("중학생")
age_elementary = st.checkbox("초등학생")

# 9. 실제 받고 있는 시급 입력 (선택)
actual_hourly = st.number_input("현재 받고 있는 시급을 입력해주세요 (선택)", min_value=0, step=1000)


if st.button("💡 예측 실행"):
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
    st.success(f"💰 예측된 시급: \u20a9{pred_won:,}")

    # ✅ 유저 입력 저장 (CSV 방식)
    user_data_row = [
        exp_kor,
        match_kor,
        jobtype_kor,
        mode_kor,
        material_kor,
        students,
        normalized_name,
        tier,
        int(age_final),
        int(age_adult),
        int(age_high),
        int(age_middle),
        int(age_elementary),
        actual_hourly,
        pred_won  # 예측 결과도 같이 저장하면 좋음
    ]
    
    # ✅ Google Sheets에 저장
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gcp_service_account"], 
        scope
    )
    client = gspread.authorize(creds)

    sheet = client.open("Tutorpays Submissions").sheet1  # 구글 시트 이름
    sheet.append_row(user_data_row)

    import csv
    with open("form_submissions.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(user_data_row)
