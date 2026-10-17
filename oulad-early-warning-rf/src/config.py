import os

# Cấu hình đường dẫn thư mục chuẩn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")

MODELS_DIR = os.path.join(BASE_DIR, "results", "models")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")

# Tự động tạo các thư mục nếu chưa có
for d in [PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Khóa cấu hình thực nghiệm
SEED = 42
CUTOFF_DAY = 28
PRIMARY_KEY = ["code_module", "code_presentation", "id_student"]
TARGET_COL = "target"

# --- DANH SÁCH 39 ĐẶC TRƯNG ---
# 1. 7 biến phân loại (Categorical)
CATEGORICAL_FEATURES = [
    "code_module", "gender", "region", "highest_education",
    "imd_band", "age_band", "disability"
]

# 2. 12 biến số nền & tương tác tổng quát (Numerical)
BASE_NUMERICAL_FEATURES = [
    "num_of_prev_attempts", "studied_credits", "days_since_registration",
    "total_clicks", "active_days", "unique_sites", "recency_days",
    "mean_clicks_active_day", "clicks_d0_6", "clicks_d7_13",
    "clicks_d14_20", "clicks_d21_28"
]

# 3. 20 biến tương tác theo loại tài nguyên VLE
ACTIVITY_TYPES = [
    "dataplus", "dualpane", "externalquiz", "folder", "forumng", "glossary",
    "homepage", "htmlactivity", "oucollaborate", "oucontent", "ouelluminate",
    "ouwiki", "page", "questionnaire", "quiz", "repeatactivity", "resource",
    "sharedsubpage", "subpage", "url"
]
ACTIVITY_FEATURES = [f"activity_clicks_{act}" for act in ACTIVITY_TYPES]

# Tổng hợp
NUMERICAL_FEATURES = BASE_NUMERICAL_FEATURES + ACTIVITY_FEATURES
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES