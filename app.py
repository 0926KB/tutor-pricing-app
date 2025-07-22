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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import csv
from datetime import datetime

# =============================
# Streamlit 기본 설정
# =============================
st.set_page_config(page_title="프리랜서 강사 시급 예측기", page_icon="💰", layout="centered")

RAW_FILE = "survey.csv"

# =============================
# 대학 분류 테이블 & 함수
# =============================
UNIV_TABLE = {
    # ---- Tier A (3점) ----
    "서울대학교": {"keywords": ["서울대학교", "서울대", "snu"], "tier": "A", "score": 3},
    "연세대학교": {"keywords": ["연세대학교", "연세대", "yonsei"], "tier": "A", "score": 3},
    "고려대학교": {"keywords": ["고려대학교", "고려대"], "tier": "A", "score": 3},
    # ---- Tier B (2점) ----
    "성균관대학교": {"keywords": ["성균관대학교", "성균관대"], "tier": "B", "score": 2},
    "한양대학교": {"keywords": ["한양대학교", "한양대"], "tier": "B", "score": 2},
    "경희대학교": {"keywords": ["경희대학교", "경희대"], "tier": "B", "score": 2},
    "이화여자대학교": {"keywords": ["이화여자대학교", "이화여대"], "tier": "B", "score": 2},
    # ---- Tier C (1점) : 예시로 일부 추가, 필요시 확장 ----
    "서강대학교": {"keywords": ["서강대학교", "서강대"], "tier": "C", "score": 1},
    "중앙대학교": {"keywords": ["중앙대학교", "중앙대"], "tier": "C", "score": 1},
    "한국외국어대학교": {"keywords": ["한국외국어대학교", "한국외대", "hankuk"], "tier": "C", "score": 1},
    "동국대학교": {"keywords": ["동국대학교", "동국대"], "tier": "C", "score": 1},
    "홍익대학교": {"keywords": ["홍익대학교", "홍익대"], "tier": "C", "score": 1},
}

DEFAULT_UNIV = {"name": "기타", "tier": "D", "score": 0}

def classify_univ(raw_name: str):
    """raw_name -> (표준화된 학교명, 티어문자, 점수)"""
    if not raw_name:
        return DEFAULT_UNIV["name"], DEFAULT_UNIV["tier"], DEFAULT_UNIV["score"]
    text = re.sub(r"[^\w\s]", "", str(raw_name).lower())
    for uni_name, info in UNIV_TABLE.items():
        for kw in info["keywords"]:
            if kw.lower() in text:
                return uni_name, info["tier"], info["score"]
    return DEFAULT_UNIV["name"], DEFAULT_UNIV["tier"], DEFAULT_UNIV["score"]

# =============================
# 데이터 정제
# =============================
@st.cache_data
def clean_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # 시급 정제
    def clean_hourly_won(text):
        if pd.isna(text):
            return np.nan
        text = str(text).strip()
        if text.replace(".", "", 1).isdigit():
            val = float(text)
            if val <= 10:
                return int(val * 10000)
            elif val < 300:
                return int(val * 100)
            else:
                return int(val)
        numbers = [int(n.replace(",", "")) for n in re.findall(r"\d{2,3},?\d{3}", text)]
        numbers = [n for n in numbers if n >= 1000]
        return int(np.mean(numbers)) if numbers else np.nan

    df["hourly_won_clean"] = df["본인의 시급을 입력해주세요. (원)"].apply(clean_hourly_won)
    df["log_hourly_won_clean"] = df["hourly_won_clean"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

    # 전공 연관성 라벨
    score_to_label = {"1": "E", "2": "D", "3": "C", "4": "B", "5": "A"}
    df["major_match_label"] = df["전공과 가르치는 과목이 얼마나 관련이 있나요?"].astype(str).map(score_to_label)

    # 경력 라벨
    def normalize_experience(val):
        if pd.isna(val):
            return None
        s = str(val).strip().replace(" ", "")
        if "1년이하" in s:
            return "1년 이하"
        if "1~2년" in s or "1년~2년" in s:
            return "1~2년"
        if "3~5년" in s or "3-5년" in s:
            return "3~5년"
        if "6년이상" in s:
            return "6년 이상"
        return s

    df["experience_years_normalized"] = df["과외 경력은 얼마나 되시나요?"].apply(normalize_experience)
    df["experience_label"] = df["experience_years_normalized"].map({
        "6년 이상": "A", "3~5년": "B", "1~2년": "C", "1년 이하": "D"
    })

    # 전업/부업 라벨
    df["is_primary_job_label"] = df["현재 과외 활동은 본업인가요, 부업인가요?"].map({
        "전업 (과외가 주 수입원)": "A", "부업 (본업이 따로 있음)": "B"
    })

    # 수업 방식 라벨
    df["teaching_mode_label"] = df["수업 방식은 어떻게 진행하시나요?"].map({
        "대면": "A",
        "대면과 비대면 모두 가능": "A",
        "비대면": "B"
    })

    # 자료 유형 라벨
    def label_material_type(cell):
        if pd.isna(cell):
            return "B"
        return "A" if "자체 제작" in str(cell) else "B"

    df["material_label"] = df["수업 중 제공하는 자료 유형을 모두 선택해주세요."].apply(label_material_type)

    # 학생 수 정수화
    def clean_student_count(text):
        if pd.isna(text):
            return np.nan
        cleaned = re.sub(r"[^\d]", "", str(text))
        return int(cleaned) if cleaned else np.nan

    df["students_total_label"] = df["지금까지 가르친 학생 수는 몇 명인가요? (숫자만 입력해주세요)"].apply(clean_student_count)

    # 대학 분류 (이제 하나의 함수 사용)
    df[["univ_normalized", "univ_tier", "univ_tier_score"]] = df["학교명을 기입해주세요.(재학중 포함)"].apply(
        lambda x: pd.Series(classify_univ(x))
    )

    # 가르치는 연령대 멀티핫
    AGE_LABELS = {
        "파이널 입시 준비생": "age_final",
        "성인": "age_adult",
        "고등학생": "age_high",
        "중학생": "age_middle",
        "초등학생": "age_elementary"
    }

    def multihot(cell):
        if pd.isna(cell):
            return []
        items = [x.strip() for x in str(cell).split(",")]
        return [AGE_LABELS[i] for i in items if i in AGE_LABELS]

    df["student_age_label"] = df["가르치는 학생 연령대를 선택해주세요."].apply(multihot)
    for lbl in AGE_LABELS.values():
        df[lbl] = df["student_age_label"].apply(lambda lst: int(lbl in lst))

    df.to_csv("survey_clean.csv", index=False)
    return df

# =============================
# 모델 학습 (Train/Test 분리, 캐시)
# =============================
@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.dropna(subset=["log_hourly_won_clean"])

    # --- Feature sets ---
    categorical = [
        "experience_label",
        "major_match_label",
        "is_primary_job_label",
        "teaching_mode_label",
        "material_label",
        "univ_tier"  # ← 숫자(점수) 대신 범주형으로 사용
    ]
    numeric = ["students_total_label"]
    multi = ["age_final", "age_adult", "age_high", "age_middle", "age_elementary"]
    target = "log_hourly_won_clean"

    X = df[categorical + numeric + multi]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", SimpleImputer(strategy="mean"), numeric)
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    # 평가
    y_pred_test = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    r2 = float(r2_score(y_test, y_pred_test))
    metrics = {"rmse": rmse, "r2": r2, "trained_at": datetime.now().isoformat()}

    return pipeline, metrics, categorical, numeric, multi

# =============================
# MAIN
# =============================
df = clean_data(RAW_FILE)
pipeline, metrics, CAT_COLS, NUM_COLS, MULTI_COLS = train_model(df)

# =============================
# UI
# =============================
st.title("📊 프리랜서 강사 시급 예측기")
st.caption("UI 입력값은 학습에 사용되지 않고, **이미 학습된 모델로 추론만** 수행합니다.")

with st.expander("📈 현재 모델 테스트 성능", expanded=False):
    st.write(f"**RMSE**: {metrics['rmse']:,.2f}")
    st.write(f"**R²**: {metrics['r2']:.4f}")
    st.write(f"Trained at: {metrics['trained_at']}")

# 1. 경력
exp_options = {"6년 이상": "A", "3~5년": "B", "1~2년": "C", "1년 이하": "D"}
exp_kor = st.selectbox("과외 경력", list(exp_options.keys()))
exp = exp_options[exp_kor]

# 2. 전공 연관성
match_options = {"매우 높음": "A", "높음": "B", "보통": "C", "낮음": "D", "전혀 없음": "E"}
match_kor = st.selectbox("전공과 과외 과목의 연관성", list(match_options.keys()))
match = match_options[match_kor]

# 3. 전업/부업
jobtype_options = {"전업 (과외가 주 수입원)": "A", "부업 (본업이 따로 있음)": "B"}
jobtype_kor = st.selectbox("과외 활동 유형", list(jobtype_options.keys()))
jobtype = jobtype_options[jobtype_kor]

# 4. 수업 방식
mode_options = {"대면 / 대면+비대면 모두 가능": "A", "비대면만 가능": "B"}
mode_kor = st.selectbox("수업 방식", list(mode_options.keys()))
mode = mode_options[mode_kor]

# 5. 자료 제공 유형
material_options = {"자체 제작 자료 제공": "A", "기성 자료 또는 미제공": "B"}
material_kor = st.selectbox("자료 유형", list(material_options.keys()))
material = material_options[material_kor]

# 6. 누적 학생 수
students = st.number_input("누적 학생 수", min_value=0, step=1)

# 7. 대학 입력 & 분류
raw_school = st.text_input("학교명을 입력하세요 (예: 서울대, 연세대학교 등)")
normalized_name, tier_letter, tier_score = classify_univ(raw_school)
st.write(f"🎓 분류: {normalized_name} / 티어 {tier_letter} / 점수 {tier_score}")

# 8. 연령대
st.write("학생 연령대 (복수 선택 가능)")
age_final = st.checkbox("입시 준비생")
age_adult = st.checkbox("성인")
age_high = st.checkbox("고등학생")
age_middle = st.checkbox("중학생")
age_elementary = st.checkbox("초등학생")

# 9. 실제 시급(선택)
actual_hourly = st.number_input("현재 받고 있는 시급을 입력해주세요 (선택)", min_value=0, step=1000)

# 예측
if st.button("💡 예측 실행"):
    input_df = pd.DataFrame([{
        "experience_label": exp,
        "major_match_label": match,
        "is_primary_job_label": jobtype,
        "teaching_mode_label": mode,
        "material_label": material,
        "univ_tier": tier_letter,
        "students_total_label": students,
        "age_final": int(age_final),
        "age_adult": int(age_adult),
        "age_high": int(age_high),
        "age_middle": int(age_middle),
        "age_elementary": int(age_elementary)
    }])

    pred_log = pipeline.predict(input_df)[0]
    pred_won = int(np.expm1(pred_log))
    st.success(f"💰 예측된 시급: ₩{pred_won:,}")

    # 저장할 데이터 (원하면 헤더 행 미리 만들어두는 걸 권장)
    user_data_row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        raw_school,
        normalized_name,
        tier_letter,
        tier_score,
        exp_kor, exp,
        match_kor, match,
        jobtype_kor, jobtype,
        mode_kor, mode,
        material_kor, material,
        students,
        int(age_final), int(age_adult), int(age_high), int(age_middle), int(age_elementary),
        actual_hourly,
        float(pred_log),
        pred_won
    ]

    # Google Sheets 저장
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
        client = gspread.authorize(creds)
        sheet = client.open("Tutorpays Submissions").sheet1
        sheet.append_row(user_data_row)
        st.info("✅ Google Sheets 저장 완료")
    except Exception as e:
        st.warning(f"Google Sheets 저장 실패: {e}")

    # 로컬 CSV 저장
    try:
        with open("form_submissions.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(user_data_row)
        st.info("✅ 로컬 CSV 저장 완료")
    except Exception as e:
        st.warning(f"CSV 저장 실패: {e}")
