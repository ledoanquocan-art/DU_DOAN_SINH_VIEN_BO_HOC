
import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from src import config


def run_saved_model_menu():
    print("\n=== ĐÁNH GIÁ MÔ HÌNH ĐÃ LƯU ===")

    models_dir = Path(config.MODELS_DIR)
    test_path = Path(config.PROCESSED_DATA_DIR) / "svm_test_set.csv"

    models = sorted(models_dir.glob("*.joblib"))

    if not models:
        print(f"Chưa có mô hình .joblib trong: {models_dir}")
        print("Hãy huấn luyện mô hình trước.")
        return

    if not test_path.exists():
        print(f"Chưa tìm thấy tập test: {test_path}")
        print("Hãy huấn luyện SVM trước để tạo tập test.")
        return

    print("\nDanh sách mô hình:")
    for i, model_path in enumerate(models, start=1):
        print(f"  {i}. {model_path.name}")
    print("  0. Quay lại")

    choice = input("Chọn mô hình cần đánh giá: ").strip()

    if choice == "0":
        return

    if not choice.isdigit() or not (1 <= int(choice) <= len(models)):
        print("Lựa chọn không hợp lệ.")
        return

    model_path = models[int(choice) - 1]

    try:
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)

        if config.TARGET_COL not in test_df.columns:
            raise ValueError(
                f"Tập test không có cột {config.TARGET_COL}."
            )

        missing = [
            col for col in config.ALL_FEATURES
            if col not in test_df.columns
        ]
        if missing:
            raise ValueError(f"Tập test thiếu đặc trưng: {missing}")

        X_test = test_df[config.ALL_FEATURES]
        y_test = test_df[config.TARGET_COL]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )

        print(f"\nMô hình: {model_path.name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nBáo cáo phân loại:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Vẽ và lưu confusion matrix.
        labels = sorted(pd.unique(pd.concat([
            y_test.reset_index(drop=True),
            pd.Series(y_pred),
        ])).tolist())

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels,
        )
        display.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_path.stem}")
        plt.tight_layout()

        figures_dir = Path(config.FIGURES_DIR)
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure_path = figures_dir / f"confusion_matrix_{model_path.stem}.png"
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

        # Xuất báo cáo JSON.
        metrics_dir = Path(config.METRICS_DIR)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = (
            metrics_dir / f"evaluation_{model_path.stem}_{timestamp}.json"
        )

        result = {
            "model_name": model_path.name,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "test_dataset": str(test_path),
            "test_samples": int(len(y_test)),
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "labels": [str(label) for label in labels],
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix_image": str(figure_path),
        }

        with open(result_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)

        print(f"\nĐã lưu biểu đồ: {figure_path}")
        print(f"Đã xuất kết quả JSON: {result_path}")

    except Exception as exc:
        print(f"\nLỗi khi đánh giá mô hình: {exc}")
