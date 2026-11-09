import os
import sys
import glob
import joblib
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


def print_progress_bar(percent, status_text="", bar_length=32):
    """Hiển thị thanh tiến trình % trực tiếp trên Terminal."""
    percent = max(0, min(100, int(percent)))
    filled_length = int(round(bar_length * percent / 100.0))
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    sys.stdout.write(f"\r  ⏳ [{bar}] {percent:>3}% | {status_text:<48}")
    sys.stdout.flush()
    if percent >= 100:
        sys.stdout.write("\n")
        sys.stdout.flush()


def get_user_param(param_name, default_value, param_type, valid_options=None, description=""):
    """Yêu cầu người dùng nhập từng tham số trước khi huấn luyện."""
    opt_str = f" | Lựa chọn: {valid_options}" if valid_options else ""
    prompt = f"  👉 Nhập [{param_name}] {description}{opt_str} (Nhấn Enter = {default_value}): "
    while True:
        raw = input(prompt).strip()
        if raw == "":
            print(f"     ✔ Đã chọn (mặc định): {param_name} = {default_value}")
            return default_value

        if raw.lower() == "none":
            if valid_options and (None not in valid_options and "None" not in valid_options):
                print(f"     ❌ Tham số [{param_name}] không cho phép giá trị None!")
                continue
            print(f"     ✔ Đã chọn: {param_name} = None")
            return None

        try:
            val = param_type(raw)
            if valid_options and val not in valid_options:
                print(f"     ❌ Giá trị không hợp lệ! Chỉ chấp nhận: {valid_options}")
                continue
            print(f"     ✔ Đã chọn: {param_name} = {val}")
            return val
        except ValueError:
            print(f"     ❌ Sai định dạng! Vui lòng nhập kiểu {param_type.__name__}.")


def format_model_identity(model_key, model_name, params):
    """Tạo tên hiển thị và mã file duy nhất dựa trên loại mô hình + tham số."""
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    model_label = f"{model_name} [{param_str}]"

    slug_parts = []
    for k, v in params.items():
        val_clean = str(v).replace(".", "p").replace("-", "m").replace(" ", "")
        slug_parts.append(f"{k}-{val_clean}")
    model_id = f"{model_key}__" + "__".join(slug_parts)
    return model_id, model_label


def load_and_split_data(save_split_files=True):
    """
    Đọc features_day28.csv, chia Train (80%) / Test (20%) theo id_student,
    lưu thành 2 file train_day28.csv và test_day28.csv và trả về dữ liệu.
    """
    feat_path = config.FEATURES_DATA_PATH
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            "Chưa có file features_day28.csv! Vui lòng chạy [Mục 1. Chạy tiền xử lý dữ liệu] trước."
        )

    df = pd.read_csv(feat_path)
    X = df[config.ALL_FEATURES]
    y = df[config.TARGET_COL]
    groups = df["id_student"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    train_idx, test_idx = next(gss.split(X, y, groups))
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])), "Rò rỉ id_student!"

    # Xuất trực tiếp 2 file train_day28.csv và test_day28.csv ra ổ cứng nếu chưa có hoặc được yêu cầu
    if save_split_files or not (os.path.exists(config.TRAIN_DATA_PATH) and os.path.exists(config.TEST_DATA_PATH)):
        df.iloc[train_idx].to_csv(config.TRAIN_DATA_PATH, index=False)
        df.iloc[test_idx].to_csv(config.TEST_DATA_PATH, index=False)

    X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    cv = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config.SEED).split(
            X_train, y_train, groups_train
        )
    )
    return df, X_train, y_train, X_test, y_test, cv, test_idx


def build_preprocessor(scale_numerical=False):
    """Tạo ColumnTransformer cho 7 biến phân loại và 32 biến số."""
    cat_pipe = Pipeline([
        ("missing", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))
    ])
    num_steps = [("missing", SimpleImputer(strategy="median"))]
    if scale_numerical:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    return ColumnTransformer([
        ("cat", cat_pipe, config.CATEGORICAL_FEATURES),
        ("num", num_pipe, config.NUMERICAL_FEATURES)
    ])


def train_and_save_model(model_key, model_name, estimator, params, scale_numerical=False, progress_callback=None):
    """
    Huấn luyện mô hình có thanh tiến trình % theo từng bước (Fold 1..5 + Fit tổng),
    lưu model độc lập và in rõ đường dẫn tập Train / tập Test.
    """
    def update_prog(pct, msg):
        print_progress_bar(pct, msg)
        if progress_callback:
            progress_callback(pct, msg)

    print(f"\n🚀 Bắt đầu tiến trình huấn luyện: {model_name}")
    update_prog(5, "Đang nạp và tách tập dữ liệu Train / Test...")
    _, X_train, y_train, X_test, _, cv, _ = load_and_split_data(save_split_files=True)

    model_id, model_label = format_model_identity(model_key, model_name, params)
    base_pipe = Pipeline([
        ("preprocess", build_preprocessor(scale_numerical=scale_numerical)),
        ("model", estimator)
    ])

    fold_accs = []
    fold_f1s = []
    n_folds = len(cv)

    # Chạy từng Fold để cập nhật % tiến trình theo thời gian thực
    for idx, (f_train_idx, f_val_idx) in enumerate(cv, 1):
        start_pct = 10 + int((idx - 1) * (70 / n_folds))
        update_prog(start_pct, f"Đang huấn luyện Cross-Validation Fold {idx}/{n_folds}...")

        fold_pipe = clone(base_pipe)
        X_tr_fold, y_tr_fold = X_train.iloc[f_train_idx], y_train.iloc[f_train_idx]
        X_va_fold, y_va_fold = X_train.iloc[f_val_idx], y_train.iloc[f_val_idx]

        fold_pipe.fit(X_tr_fold, y_tr_fold)
        y_va_pred = fold_pipe.predict(X_va_fold)

        fold_accs.append(accuracy_score(y_va_fold, y_va_pred))
        fold_f1s.append(f1_score(y_va_fold, y_va_pred, average="macro"))

        end_pct = 10 + int(idx * (70 / n_folds))
        update_prog(end_pct, f"Hoàn tất Fold {idx}/{n_folds} (Macro-F1: {fold_f1s[-1]*100:.2f}%)")

    # Bước huấn luyện mô hình tổng trên toàn bộ tập Train (80% -> 95%)
    update_prog(85, "Đang khớp mô hình trên toàn bộ 21.956 mẫu Train...")
    base_pipe.fit(X_train, y_train)

    # Bước lưu file .joblib (95% -> 100%)
    update_prog(95, "Đang đóng gói và lưu file mô hình (.joblib)...")
    cv_acc_mean = round(float(np.mean(fold_accs)) * 100, 2)
    cv_f1_mean = round(float(np.mean(fold_f1s)) * 100, 2)
    cv_f1_std = round(float(np.std(fold_f1s)) * 100, 2)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_filename = f"{model_id}.joblib"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    artifact = {
        "model_id": model_id,
        "model_key": model_key,
        "model_name": model_name,
        "model_label": model_label,
        "params": params,
        "pipeline": base_pipe,
        "cv_accuracy": cv_acc_mean,
        "cv_macro_f1": cv_f1_mean,
        "cv_macro_f1_std": cv_f1_std,
        "test_metrics": None,
        "train_data_path": config.TRAIN_DATA_PATH,
        "test_data_path": config.TEST_DATA_PATH,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(artifact, model_path)
    update_prog(100, "Hoàn tất 100% quá trình huấn luyện và lưu trữ!")

    print("=" * 78)
    print(f"✅ ĐÃ HUẤN LUYỆN VÀ LƯU MÔ HÌNH THÀNH CÔNG!")
    print("-" * 78)
    print(f"  • Tên mô hình & tham số : {model_label}")
    print(f"  • Kết quả 5-Fold CV     : Accuracy = {cv_acc_mean}% | Macro-F1 = {cv_f1_mean}% ± {cv_f1_std}%")
    print(f"  • Tập dữ liệu Train     : {config.TRAIN_DATA_PATH} ({len(X_train):,} dòng - 80%)")
    print(f"  • Tập dữ liệu Test      : {config.TEST_DATA_PATH} ({len(X_test):,} dòng - 20%)")
    print(f"  • File mô hình đã lưu   : {model_path}")
    print("=" * 78)
    return artifact


def list_saved_models():
    """Đọc toàn bộ các file mô hình (.joblib) đã lưu trong results/models/."""
    files = sorted(glob.glob(os.path.join(config.MODELS_DIR, "*.joblib")))
    models = []
    for f in files:
        try:
            data = joblib.load(f)
            data["filepath"] = f
            if "model_label" not in data:
                _, data["model_label"] = format_model_identity(
                    data.get("model_key", "model"),
                    data.get("model_name", "Model"),
                    data.get("params", {})
                )
            models.append(data)
        except Exception:
            continue
    return models


def open_file_on_os(filepath):
    """Tự động mở file trên hệ điều hành sau khi dự đoán xong."""
    abs_path = os.path.abspath(filepath)
    try:
        if sys.platform.startswith("win"):
            os.startfile(abs_path)
        elif sys.platform.startswith("darwin"):
            subprocess.call(["open", abs_path])
        else:
            subprocess.call(["xdg-open", abs_path])
        print(f"  📂 Đã mở trực tiếp file kết quả: {abs_path}")
    except Exception as e:
        print(f"  ⚠️ Không thể mở tự động ({e}). Bạn có thể mở file tại: {abs_path}")