import os
import sys
import glob
import joblib
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import config


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
    """
    Tạo tên hiển thị và mã file duy nhất dựa trên loại mô hình + toàn bộ tham số.
    Đảm bảo 2 mô hình cùng loại nhưng khác tham số sẽ có 2 model_id khác nhau.
    """
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    model_label = f"{model_name} [{param_str}]"

    slug_parts = []
    for k, v in params.items():
        val_clean = str(v).replace(".", "p").replace("-", "m").replace(" ", "")
        slug_parts.append(f"{k}-{val_clean}")
    model_id = f"{model_key}__" + "__".join(slug_parts)
    return model_id, model_label


def load_and_split_data():
    """Đọc features_day28.csv và chia Train/Test + 5-Fold CV cố định theo id_student."""
    feat_path = os.path.join(config.PROCESSED_DATA_DIR, "features_day28.csv")
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


def train_and_save_model(model_key, model_name, estimator, params, scale_numerical=False):
    """
    Huấn luyện mô hình với tham số vừa nhập, đánh giá 5-Fold CV và lưu thành file riêng theo tham số.
    """
    _, X_train, y_train, _, _, cv, _ = load_and_split_data()

    model_id, model_label = format_model_identity(model_key, model_name, params)
    pipe = Pipeline([
        ("preprocess", build_preprocessor(scale_numerical=scale_numerical)),
        ("model", estimator)
    ])

    print(f"\n⏳ Đang huấn luyện & đánh giá 5-Fold CV cho: {model_label} ...")
    cv_res = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring={"acc": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=2
    )
    pipe.fit(X_train, y_train)

    cv_acc_mean = round(float(np.mean(cv_res["test_acc"])) * 100, 2)
    cv_f1_mean = round(float(np.mean(cv_res["test_f1_macro"])) * 100, 2)
    cv_f1_std = round(float(np.std(cv_res["test_f1_macro"])) * 100, 2)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_filename = f"{model_id}.joblib"
    model_path = os.path.join(config.MODELS_DIR, model_filename)

    artifact = {
        "model_id": model_id,
        "model_key": model_key,
        "model_name": model_name,
        "model_label": model_label,
        "params": params,
        "pipeline": pipe,
        "cv_accuracy": cv_acc_mean,
        "cv_macro_f1": cv_f1_mean,
        "cv_macro_f1_std": cv_f1_std,
        "test_metrics": None,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(artifact, model_path)

    print("=" * 65)
    print(f"✅ ĐÃ HUẤN LUYỆN VÀ LƯU MÔ HÌNH THÀNH CÔNG!")
    print(f"  • Tên mô hình & tham số : {model_label}")
    print(f"  • Kết quả 5-Fold CV     : Accuracy = {cv_acc_mean}% | Macro-F1 = {cv_f1_mean}% ± {cv_f1_std}%")
    print(f"  • File lưu độc lập      : results/models/{model_filename}")
    print("=" * 65)
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