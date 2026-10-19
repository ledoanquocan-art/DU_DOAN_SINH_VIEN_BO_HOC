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
 
# =====================================================================
# [MAIN MODEL & PREDICTOR] - Random Forest, Permutation Importance, Demo
# =====================================================================
 
# --- Chia train/test ---
# "temporal": train trên các khóa cũ, test trên TEST_PRESENTATION (sát thực tế, tránh rò rỉ thời gian)
# "random"  : chia ngẫu nhiên có phân tầng theo nhãn
SPLIT_MODE = "temporal"
TEST_PRESENTATION = "2014J"
RANDOM_TEST_SIZE = 0.2
 
# --- Siêu tham số Random Forest (mô hình chính) ---
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 16,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": SEED,
}
 
# --- Tinh chỉnh siêu tham số (RandomizedSearchCV, dùng khi chạy với --tune) ---
# Khóa có tiền tố "rf__" vì bước Random Forest trong Pipeline tên là "rf"
TUNE_PARAM_GRID = {
    "rf__n_estimators": [200, 300, 500],
    "rf__max_depth": [10, 16, 24, None],
    "rf__min_samples_leaf": [3, 5, 10],
    "rf__max_features": ["sqrt", 0.3, 0.5],
}
TUNE_N_ITER = 10
TUNE_CV = 3
TUNE_SCORING = "roc_auc"
 
# --- Permutation Importance (tính trên tập test) ---
PERM_N_REPEATS = 10
PERM_SCORING = "roc_auc"
PERM_TOP_N = 15
 
# --- Ngưỡng dự đoán & phân mức rủi ro cho Predictor ---
DECISION_THRESHOLD = 0.5
RISK_LOW_MAX = 0.33      # xác suất < 0.33          -> "Thấp"
RISK_HIGH_MIN = 0.66     # xác suất >= 0.66         -> "Cao" (còn lại: "Trung bình")
RISK_LABELS = ("Thấp", "Trung bình", "Cao")
 
# --- Đường dẫn đầu ra của mô hình chính ---
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
RF_METRICS_PATH = os.path.join(METRICS_DIR, "random_forest_metrics.json")
PERM_IMPORTANCE_CSV = os.path.join(METRICS_DIR, "permutation_importance.csv")
PERM_IMPORTANCE_FIG = os.path.join(FIGURES_DIR, "fig_permutation_importance.png")
RF_PREDICTIONS_CSV = os.path.join(PREDICTIONS_DIR, "random_forest_predictions.csv")
 
# --- Số sinh viên hiển thị trong demo ---
DEMO_N_STUDENTS = 10
 




