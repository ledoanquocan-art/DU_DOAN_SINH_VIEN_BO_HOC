
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src import config


MODEL_PATH = Path(config.MODELS_DIR) / "svm_rbf.joblib"
TEST_PATH = Path(config.PROCESSED_DATA_DIR) / "svm_test_set.csv"
INFO_PATH = Path(config.METRICS_DIR) / "svm_training_info.json"


def find_feature_dataset():
    """Tìm file CSV đã có đủ đặc trưng và nhãn target."""
    processed_dir = Path(config.PROCESSED_DATA_DIR)

    for csv_path in sorted(processed_dir.glob("*.csv")):
        if csv_path.name in {
            "cohort_day28.csv",
            "svm_test_set.csv",
        }:
            continue

        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except Exception:
            continue

        required = set(config.ALL_FEATURES) | {config.TARGET_COL}
        if required.issubset(columns):
            return csv_path

    raise FileNotFoundError(
        "Không tìm thấy CSV có đủ 39 đặc trưng và cột target trong "
        f"{processed_dir}. cohort_day28.csv hiện chưa đủ đặc trưng. "
        "Hãy chạy bước feature engineering trước."
    )


def train_svm():
    print("\n=== HUẤN LUYỆN SVM (RBF) ===")

    try:
        c_value = float(input("Nhập tham số C (ví dụ 1.0): ").strip() or "1.0")
        gamma_value = input(
            "Nhập gamma (scale / auto / số, mặc định scale): "
        ).strip() or "scale"

        if c_value <= 0:
            raise ValueError("C phải lớn hơn 0.")

        if gamma_value not in {"scale", "auto"}:
            gamma_value = float(gamma_value)
            if gamma_value <= 0:
                raise ValueError("Gamma phải lớn hơn 0.")

        data_path = find_feature_dataset()
        print(f"Đọc dữ liệu: {data_path}")

        df = pd.read_csv(data_path)
        required_cols = config.ALL_FEATURES + [config.TARGET_COL]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Dữ liệu thiếu các cột: {missing}")

        df = df[required_cols].dropna(subset=[config.TARGET_COL]).copy()
        X = df[config.ALL_FEATURES]
        y = df[config.TARGET_COL]

        if y.nunique() < 2:
            raise ValueError("Cột target cần có ít nhất hai lớp.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=config.SEED,
            stratify=y,
        )

        categorical_cols = [
            col for col in config.CATEGORICAL_FEATURES
            if col in X.columns
        ]
        numerical_cols = [
            col for col in config.NUMERICAL_FEATURES
            if col in X.columns
        ]

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        preprocessor = ColumnTransformer([
            ("categorical", categorical_pipeline, categorical_cols),
            ("numerical", numerical_pipeline, numerical_cols),
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(
                kernel="rbf",
                C=c_value,
                gamma=gamma_value,
                class_weight="balanced",
                random_state=config.SEED,
            )),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        INFO_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Lưu mô hình đã bao gồm bước tiền xử lý.
        joblib.dump(model, MODEL_PATH)

        # Lưu đúng tập test để evaluator dùng lại.
        test_df = X_test.copy()
        test_df[config.TARGET_COL] = y_test
        test_df.to_csv(TEST_PATH, index=False)

        info = {
            "model": "SVM",
            "kernel": "rbf",
            "C": c_value,
            "gamma": gamma_value,
            "source_dataset": str(data_path),
            "test_size": 0.2,
            "random_state": config.SEED,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(
                f1_score(y_test, predictions, average="weighted")
            ),
            "model_path": str(MODEL_PATH),
            "test_path": str(TEST_PATH),
        }

        with open(INFO_PATH, "w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=4)

        print("\nHuấn luyện SVM hoàn tất!")
        print(f"Accuracy: {info['accuracy']:.4f}")
        print(f"F1 weighted: {info['f1_weighted']:.4f}")
        print(f"Đã lưu mô hình: {MODEL_PATH}")
        print(f"Đã lưu tập test: {TEST_PATH}")

    except (ValueError, FileNotFoundError) as exc:
        print(f"\nKhông thể huấn luyện SVM: {exc}")
    except Exception as exc:
        print(f"\nĐã xảy ra lỗi khi huấn luyện SVM: {exc}")
