import os

# Cấu hình đường dẫn thư mục chuẩn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")

MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

# Đường dẫn file dữ liệu đã xử lý và tập Train / Test tách biệt
COHORT_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cohort_day28.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "features_day28.csv")
TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_day28.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_day28.csv")

# Tự động tạo các thư mục nếu chưa có
for d in [PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Khóa cấu hình thực nghiệm
SEED = 42
CUTOFF_DAY = 28
PRIMARY_KEY = ["code_module", "code_presentation", "id_student"]
TARGET_COL = "target"

# --- NHÓM 1: 7 ĐẶC TRƯNG PHÂN LOẠI (NHÂN KHẨU HỌC & MÔN HỌC) ---
CATEGORICAL_FEATURES = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability"
]

# --- NHÓM 2: 12 ĐẶC TRƯNG SỐ CƠ BẢN (ĐĂNG KÝ & TƯƠNG TÁC NGÀY 0-28) ---
BASE_NUMERICAL_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "days_since_registration",
    "total_clicks", "active_days", "unique_sites", "recency_days",
    "mean_clicks_active_day", "clicks_d0_6", "clicks_d7_13",
    "clicks_d14_20", "clicks_d21_28"
]

# --- NHÓM 3: 20 ĐẶC TRƯNG CLICK THEO LOẠI TÀI NGUYÊN VLE ---
ACTIVITY_TYPES = [
    "dataplus", "dualpane", "externalquiz", "folder", "forumng", "glossary",
    "homepage", "htmlactivity", "oucollaborate", "oucontent", "ouelluminate",
    "ouwiki", "page", "questionnaire", "quiz", "repeatactivity", "resource",
    "sharedsubpage", "subpage", "url"
]
ACTIVITY_FEATURES = [f"activity_clicks_{act}" for act in ACTIVITY_TYPES]

# --- NHÓM 4 (CẢI TIẾN): 11 ĐẶC TRƯNG NÂNG CAO (ASSESSMENTS + PRE-COURSE + TREND) ---
ASSESSMENT_FEATURES = [
    "num_submitted_d28",
    "mean_score_d28",
    "missing_assessments_d28",
    "avg_submission_delay_d28"
]

PRECOURSE_FEATURES = [
    "pre_course_clicks",
    "pre_course_active_days"
]

TREND_FEATURES = [
    "click_trend_w4_vs_w1",
    "click_trend_w4_vs_w3",
    "zero_click_weeks",
    "trailing_zero_weeks",
    "click_std_4w"
]

ENHANCED_NUMERICAL_FEATURES = ASSESSMENT_FEATURES + PRECOURSE_FEATURES + TREND_FEATURES

# Tổng hợp toàn bộ đặc trưng đầu vào (7 phân loại + 43 biến số = 50 đặc trưng)
NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ACTIVITY_FEATURES + ENHANCED_NUMERICAL_FEATURES
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES